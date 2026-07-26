"""Evaluate hierarchical SE(3)-conditioned flow control on Shadow Block."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.priors import make_ddpm, sample_action_chunks
from jepa_robotics.algos.task_families.dexterous import (
    pose_cost,
    quat_exp,
    quat_log,
    quat_mul,
    quat_conjugate,
    quat_normalize,
    relative_pose_features,
    step_pose,
)
from jepa_robotics.data import Episode, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import (
    DexterousFlowPrior,
    DexterousInverseDynamics,
)
from jepa_robotics.tasks import resolve_task


class PoseFlowController:
    def __init__(
        self,
        world_model,
        normalizer,
        spec,
        flow,
        flow_config,
        device,
        action_low,
        action_high,
        *,
        candidates: int,
        sample_steps: int,
        execute: int,
        position_step: float,
        rotation_step: float,
        selection: str,
        action_scale: float,
        neutral_candidate: bool,
        stop_position: float,
        stop_rotation: float,
        slew_limit: float,
        contact_retain_weight: float,
    ) -> None:
        self.world_model = world_model
        self.normalizer = normalizer
        self.spec = spec
        self.flow = flow
        self.flow_config = flow_config
        self.device = device
        self.low, self.high = action_low, action_high
        self.candidates = candidates
        self.sample_steps = sample_steps
        self.execute = execute
        self.position_step = position_step
        self.rotation_step = rotation_step
        self.selection = selection
        self.action_scale = action_scale
        self.neutral_candidate = neutral_candidate
        self.stop_position = stop_position
        self.stop_rotation = stop_rotation
        self.slew_limit = slew_limit
        self.contact_retain_weight = contact_retain_weight
        self.horizon = int(
            flow_config.get("chunk", flow_config.get("horizon"))
        )
        self.action_dim = int(flow_config["action_dim"])
        self.ddpm = make_ddpm(
            int(flow_config.get("flow_steps", 1)), device
        )
        self.achieved = spec.obs_dim
        self.desired = spec.obs_dim + spec.goal_dim
        self.object_mean = torch.as_tensor(
            normalizer.mean[self.achieved:self.achieved + 7],
            dtype=torch.float32,
            device=device,
        )
        self.object_std = torch.as_tensor(
            normalizer.std[self.achieved:self.achieved + 7],
            dtype=torch.float32,
            device=device,
        )
        self.buffer: list[np.ndarray] = []
        self.previous = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.buffer = []
        self.previous.fill(0.0)

    def _limit(self, actions: torch.Tensor) -> torch.Tensor:
        if self.slew_limit <= 0:
            return actions
        previous = torch.as_tensor(self.previous, device=self.device)
        limited = []
        for step in range(actions.shape[1]):
            current = previous + (actions[:, step] - previous).clamp(
                -self.slew_limit, self.slew_limit
            )
            limited.append(current)
            previous = current
        return torch.stack(limited, dim=1)

    @torch.no_grad()
    def _plan(self, state: np.ndarray) -> np.ndarray:
        achieved = state[self.achieved:self.achieved + 7]
        desired = state[self.desired:self.desired + 7]
        error = relative_pose_features(achieved, desired, position_scale=1.0)
        if (
            np.linalg.norm(error[:3]) <= self.stop_position
            and np.linalg.norm(error[3:]) <= self.stop_rotation
        ):
            return np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        subgoal = step_pose(
            achieved,
            desired,
            max_position_step=self.position_step,
            max_rotation_step=self.rotation_step,
        )
        model_state = state.copy()
        if self.flow_config.get("aligned_model_goal", False):
            model_state[self.desired:self.desired + 7] = subgoal
        normalized = torch.from_numpy(
            self.normalizer.encode(model_state).astype(np.float32)
        ).unsqueeze(0).to(self.device)
        latent = self.world_model.encode(normalized)
        delta = torch.from_numpy(
            relative_pose_features(
                achieved,
                subgoal,
                position_scale=float(self.flow_config["position_scale"]),
            )
        ).unsqueeze(0).to(self.device)
        condition_parts = [latent, delta]
        if self.flow_config.get("condition_prev_action", False):
            condition_parts.append(
                torch.from_numpy(self.previous).to(self.device).unsqueeze(0)
            )
        condition = torch.cat(condition_parts, dim=-1)
        if (
            self.flow_config.get("architecture")
            == "dexterous_pose_inverse"
        ):
            sampled = self.flow(condition).squeeze(0)
        else:
            expanded_condition = condition.expand(self.candidates, -1)
            sampled = sample_action_chunks(
                self.flow,
                self.ddpm,
                expanded_condition,
                int(self.flow_config["chunk_dim"]),
                self.device,
                objective="flow",
                flow_steps=self.sample_steps,
            ).view(
                self.candidates, self.horizon, self.action_dim
            )
        if self.flow_config.get("action_representation", "absolute") == "delta":
            delta_mean = torch.as_tensor(
                self.flow_config["action_delta_mean"],
                dtype=torch.float32,
                device=self.device,
            )
            delta_std = torch.as_tensor(
                self.flow_config["action_delta_std"],
                dtype=torch.float32,
                device=self.device,
            )
            increments = (
                sampled * delta_std + delta_mean
            ) * self.action_scale
            chunks = (
                torch.from_numpy(self.previous)
                .to(self.device)
                .view(1, 1, -1)
                + torch.cumsum(increments, dim=1)
            )
        else:
            chunks = self.action_scale * sampled
        chunks = self._limit(chunks.clamp(self.low, self.high))
        if self.neutral_candidate:
            chunks[0].zero_()
        if self.selection == "trust":
            return chunks[0].cpu().numpy()
        rollout = self.world_model.predict_rollout(
            latent.expand(len(chunks), -1), chunks, self.horizon
        )
        object_prediction = (
            self.world_model.predict_object(rollout[:, -1])
            if hasattr(self.world_model, "predict_object") else None
        )
        if object_prediction is None:
            predicted = self.normalizer.decode_tensor(
                self.world_model.state_probe(rollout[:, -1])
            )[:, self.achieved:self.achieved + 7]
        else:
            predicted = object_prediction * self.object_std + self.object_mean
        position = torch.linalg.vector_norm(
            predicted[:, :3] - torch.as_tensor(subgoal[:3], device=self.device), dim=-1
        )
        quaternion = predicted[:, 3:]
        quaternion /= torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        target_q = torch.as_tensor(subgoal[3:], device=self.device)
        target_q /= torch.linalg.vector_norm(target_q).clamp_min(1e-6)
        rotation = 2.0 * torch.acos(
            (quaternion * target_q).sum(-1).abs().clamp(max=1.0)
        )
        cost = 10.0 * position + rotation
        if self.contact_retain_weight > 0 and self.world_model.contact_dims is not None:
            contact_lo, contact_hi = self.world_model.contact_dims
            contact_prediction = self.world_model.contact_consistency(rollout[:, -1])
            contact_mean = torch.as_tensor(
                self.normalizer.mean[contact_lo:contact_hi],
                dtype=torch.float32,
                device=self.device,
            )
            contact_std = torch.as_tensor(
                self.normalizer.std[contact_lo:contact_hi],
                dtype=torch.float32,
                device=self.device,
            )
            predicted_contact = contact_prediction * contact_std + contact_mean
            predicted_strength = torch.log1p(predicted_contact.clamp_min(0.0)).mean(-1)
            current_contact = torch.as_tensor(
                state[contact_lo:contact_hi], dtype=torch.float32, device=self.device
            )
            current_strength = torch.log1p(current_contact.clamp_min(0.0)).mean()
            contact_floor = 0.75 * current_strength
            cost = cost + self.contact_retain_weight * (
                contact_floor - predicted_strength
            ).clamp_min(0.0)
        return chunks[int(torch.argmin(cost))].cpu().numpy()

    def act(self, observation) -> np.ndarray:
        if not self.buffer:
            plan = self._plan(flatten_obs(observation))
            self.buffer = [
                plan[index].copy()
                for index in range(max(1, min(self.execute, len(plan))))
            ]
        action = np.clip(
            self.buffer.pop(0), self.low.cpu().numpy(), self.high.cpu().numpy()
        ).astype(np.float32)
        self.previous = action.copy()
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--flow-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--execute", type=int, default=4)
    parser.add_argument("--position-step", type=float, default=0.012)
    parser.add_argument("--rotation-step-deg", type=float, default=22.0)
    parser.add_argument("--selection", choices=["trust", "model"], default="trust")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale relative-control action proposals before model selection.",
    )
    parser.add_argument(
        "--neutral-candidate",
        action="store_true",
        help="Include an exact zero-relative-action hold chunk in model selection.",
    )
    parser.add_argument("--stop-position-cm", type=float, default=0.0)
    parser.add_argument("--stop-rotation-deg", type=float, default=0.0)
    parser.add_argument("--slew-limit", type=float, default=0.0)
    parser.add_argument(
        "--contact-retain-weight",
        type=float,
        default=0.0,
        help="Penalize candidates whose learned contact-slot prediction falls below 75% of live contact.",
    )
    parser.add_argument("--torch-seed", type=int, default=0)
    parser.add_argument(
        "--local-goal-rotation-deg",
        type=float,
        default=0.0,
        help="Replace the environment goal with a random-axis rotation this far from the start.",
    )
    parser.add_argument(
        "--local-goal-position-cm",
        type=float,
        default=0.0,
        help="Also move a synthetic local goal this many centimetres in a random xyz direction.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-episodes", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--rollout-out",
        type=Path,
        default=None,
        help="Save executed state/action trajectories for self-supervised on-policy calibration.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    world_model, normalizer, spec, _ = load_jepa_artifact(args.model_path, device)
    artifact = torch.load(args.flow_path, map_location=device, weights_only=False)
    config = artifact["config"]
    if config.get("architecture") == "dexterous_pose_inverse":
        flow = DexterousInverseDynamics(
            int(config["condition_dim"]),
            int(config["horizon"]),
            int(config["action_dim"]),
            hidden=int(config["hidden"]),
            n_blocks=int(config["blocks"]),
            heads=int(config["heads"]),
            modes=int(config["modes"]),
        ).to(device)
        flow.load_state_dict(artifact["state_dict"])
    else:
        flow = DexterousFlowPrior(
            int(config["chunk_dim"]),
            int(config["condition_dim"]),
            hidden=int(config["hidden"]),
            n_blocks=int(config["blocks"]),
            heads=int(config["heads"]),
            action_dim=int(config["action_dim"]),
        ).to(device)
        flow.load_state_dict(artifact["ema"])
    flow.eval()
    env = make_env(
        task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps
    )
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    controller = PoseFlowController(
        world_model,
        normalizer,
        spec,
        flow,
        config,
        device,
        low,
        high,
        candidates=args.candidates,
        sample_steps=args.sample_steps,
        execute=args.execute,
        position_step=args.position_step,
        rotation_step=np.radians(args.rotation_step_deg),
        selection=args.selection,
        action_scale=args.action_scale,
        neutral_candidate=args.neutral_candidate,
        stop_position=args.stop_position_cm / 100.0,
        stop_rotation=np.radians(args.stop_rotation_deg),
        slew_limit=args.slew_limit,
        contact_retain_weight=args.contact_retain_weight,
    )
    successes, costs, start_costs, best_costs = [], [], [], []
    position_errors, rotation_errors = [], []
    rollout_episodes: list[Episode] = []
    goal_rng = np.random.default_rng(args.seed + 991)
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        if args.local_goal_rotation_deg > 0 or args.local_goal_position_cm > 0:
            reset_state = flatten_obs(observation)
            current_pose = reset_state[spec.obs_dim:spec.obs_dim + 7].copy()
            axis = goal_rng.normal(size=3).astype(np.float32)
            axis /= np.linalg.norm(axis) + 1e-9
            rotation = axis * np.radians(args.local_goal_rotation_deg)
            target_q = quat_mul(quat_exp(rotation), current_pose[3:])
            direction = goal_rng.normal(size=3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-9
            target_position = (
                current_pose[:3] + direction * (args.local_goal_position_cm / 100.0)
            )
            env.unwrapped.goal = np.concatenate(
                [target_position, quat_normalize(target_q)]
            ).astype(np.float32)
            observation = env.unwrapped._get_obs()
        controller.reset()
        initial_state = flatten_obs(observation)
        rollout_states = [initial_state.copy()]
        rollout_actions = []
        start_cost = pose_cost(
            initial_state[spec.obs_dim:spec.obs_dim + 7],
            initial_state[spec.obs_dim + 7:spec.obs_dim + 14],
        )
        best_cost = start_cost
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action = controller.act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            live_state = flatten_obs(observation)
            rollout_actions.append(action.copy())
            rollout_states.append(live_state.copy())
            best_cost = min(
                best_cost,
                pose_cost(
                    live_state[spec.obs_dim:spec.obs_dim + 7],
                    live_state[spec.obs_dim + 7:spec.obs_dim + 14],
                ),
            )
        state = flatten_obs(observation)
        achieved = state[spec.obs_dim:spec.obs_dim + 7]
        desired = state[spec.obs_dim + 7:spec.obs_dim + 14]
        cost = pose_cost(
            achieved,
            desired,
        )
        q_error = quat_mul(
            quat_normalize(desired[3:]),
            quat_conjugate(quat_normalize(achieved[3:])),
        )
        successes.append(float(info.get("is_success", 0.0)))
        costs.append(cost)
        start_costs.append(start_cost)
        best_costs.append(best_cost)
        position_errors.append(float(np.linalg.norm(achieved[:3] - desired[:3])))
        rotation_errors.append(float(np.linalg.norm(quat_log(q_error))))
        rollout_episodes.append(
            Episode(
                states=np.asarray(rollout_states, dtype=np.float32),
                actions=np.asarray(rollout_actions, dtype=np.float32),
            )
        )
        if args.log_episodes:
            print(
                json.dumps(
                    {
                        "event": "dexterous_pose_flow_episode",
                        "episode": episode,
                        "success": successes[-1],
                        "start_goal_cost": round(start_cost, 4),
                        "best_goal_cost": round(best_cost, 4),
                        "final_goal_cost": round(cost, 4),
                        "final_position_cm": round(100.0 * position_errors[-1], 3),
                        "final_rotation_deg": round(
                            float(np.degrees(rotation_errors[-1])), 2
                        ),
                    }
                ),
                flush=True,
            )
    env.close()
    result = {
        "event": "dexterous_pose_flow_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "flow_path": str(args.flow_path),
        "episodes": args.episodes,
        "success_rate": float(np.mean(successes)),
        "median_start_goal_cost": float(np.median(start_costs)),
        "median_best_goal_cost": float(np.median(best_costs)),
        "median_final_goal_cost": float(np.median(costs)),
        "mean_final_goal_cost": float(np.mean(costs)),
        "median_improvement": float(np.median(np.asarray(start_costs) - np.asarray(costs))),
        "improved_fraction": float(np.mean(np.asarray(costs) < np.asarray(start_costs))),
        "median_final_position_cm": float(100.0 * np.median(position_errors)),
        "median_final_rotation_deg": float(np.degrees(np.median(rotation_errors))),
        "selection": args.selection,
        "candidates": args.candidates,
        "execute": args.execute,
        "action_scale": args.action_scale,
        "neutral_candidate": args.neutral_candidate,
        "stop_position_cm": args.stop_position_cm,
        "stop_rotation_deg": args.stop_rotation_deg,
        "position_step": args.position_step,
        "rotation_step_deg": args.rotation_step_deg,
        "contact_retain_weight": args.contact_retain_weight,
        "local_goal_rotation_deg": args.local_goal_rotation_deg,
        "local_goal_position_cm": args.local_goal_position_cm,
        "seed": args.seed,
        "torch_seed": args.torch_seed,
    }
    print(json.dumps(result), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")
    if args.rollout_out is not None:
        save_episodes_npz(args.rollout_out, rollout_episodes, spec)


if __name__ == "__main__":
    main()
