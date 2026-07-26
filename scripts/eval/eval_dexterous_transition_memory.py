"""Evaluate JEPA-conditioned episodic action memory on Shadow Block.

The controller uses no demonstrations, rewards, or policy gradients.  Its
proposals are real action chunks from reward-free exploration, retrieved first
by their observed local SE(3) effect and then by proximity to the live JEPA
state.  The learned world model can optionally rank the retrieved chunks.
"""
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

from jepa_robotics.algos.task_families.dexterous import (
    DexterousTransitionMemory,
    pose_cost,
    quat_conjugate,
    quat_exp,
    quat_log,
    quat_mul,
    quat_normalize,
    relative_pose_features,
    step_pose,
)
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


def build_memory(
    episodes_path: Path,
    world_model,
    normalizer,
    spec,
    device: torch.device,
    *,
    horizon: int,
    max_pairs: int,
    seed: int,
) -> DexterousTransitionMemory:
    states, deltas, chunks = [], [], []
    achieved = spec.obs_dim
    desired = spec.obs_dim + spec.goal_dim
    for episode in load_episodes_npz(episodes_path):
        for index in range(len(episode.actions) - horizon + 1):
            current = episode.states[index].astype(np.float32).copy()
            future_pose = episode.states[
                index + horizon, achieved:achieved + 7
            ].astype(np.float32)
            current[desired:desired + 7] = future_pose
            states.append(current)
            deltas.append(
                relative_pose_features(
                    current[achieved:achieved + 7],
                    future_pose,
                    position_scale=0.05,
                )
            )
            chunks.append(
                episode.actions[index:index + horizon].astype(np.float32)
            )
    if len(states) > max_pairs:
        keep = np.random.default_rng(seed).choice(
            len(states), max_pairs, replace=False
        )
        states = [states[index] for index in keep]
        deltas = [deltas[index] for index in keep]
        chunks = [chunks[index] for index in keep]
    normalized = normalizer.encode(np.asarray(states, dtype=np.float32))
    latent_batches = []
    with torch.no_grad():
        for start in range(0, len(normalized), 8192):
            latent_batches.append(
                world_model.encode(
                    torch.from_numpy(
                        normalized[start:start + 8192].astype(np.float32)
                    ).to(device)
                )
            )
    return DexterousTransitionMemory(
        torch.cat(latent_batches),
        torch.from_numpy(np.asarray(deltas, dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(chunks, dtype=np.float32)).to(device),
    )


class TransitionMemoryController:
    def __init__(
        self,
        world_model,
        normalizer,
        spec,
        memory,
        device,
        action_low,
        action_high,
        *,
        horizon: int,
        candidates: int,
        pose_pool: int,
        execute: int,
        position_step: float,
        rotation_step: float,
        selection: str,
        slew_limit: float,
        stop_position: float,
        stop_rotation: float,
    ) -> None:
        self.world_model = world_model
        self.normalizer = normalizer
        self.spec = spec
        self.memory = memory
        self.device = device
        self.low, self.high = action_low, action_high
        self.horizon = horizon
        self.candidates = candidates
        self.pose_pool = pose_pool
        self.execute = execute
        self.position_step = position_step
        self.rotation_step = rotation_step
        self.selection = selection
        self.slew_limit = slew_limit
        self.stop_position = stop_position
        self.stop_rotation = stop_rotation
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
        self.previous = np.zeros(spec.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.buffer = []
        self.previous.fill(0.0)

    def _rate_limit(self, actions: torch.Tensor) -> torch.Tensor:
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
        subgoal = step_pose(
            achieved,
            desired,
            max_position_step=self.position_step,
            max_rotation_step=self.rotation_step,
        )
        model_state = state.copy()
        model_state[self.desired:self.desired + 7] = subgoal
        normalized = torch.from_numpy(
            self.normalizer.encode(model_state).astype(np.float32)
        ).unsqueeze(0).to(self.device)
        latent = self.world_model.encode(normalized)
        delta = torch.from_numpy(
            relative_pose_features(
                achieved, subgoal, position_scale=0.05
            )
        ).to(self.device)
        actions, _ = self.memory.query(
            latent,
            delta,
            candidates=self.candidates,
            pose_pool=self.pose_pool,
        )
        actions = self._rate_limit(actions.clamp(self.low, self.high))
        if self.selection == "nearest":
            return actions[0].cpu().numpy()
        rollout = self.world_model.predict_rollout(
            latent.expand(len(actions), -1), actions, self.horizon
        )
        object_prediction = self.world_model.predict_object(rollout[:, -1])
        if object_prediction is None:
            predicted = self.normalizer.decode_tensor(
                self.world_model.state_probe(rollout[:, -1])
            )[:, self.achieved:self.achieved + 7]
        else:
            predicted = object_prediction * self.object_std + self.object_mean
        position = torch.linalg.vector_norm(
            predicted[:, :3]
            - torch.as_tensor(subgoal[:3], device=self.device),
            dim=-1,
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
        return actions[
            int(torch.argmin(10.0 * position + rotation))
        ].cpu().numpy()

    def act(self, observation) -> np.ndarray:
        state = flatten_obs(observation)
        error = relative_pose_features(
            state[self.achieved:self.achieved + 7],
            state[self.desired:self.desired + 7],
            position_scale=1.0,
        )
        if (
            np.linalg.norm(error[:3]) <= self.stop_position
            and np.linalg.norm(error[3:]) <= self.stop_rotation
        ):
            self.buffer = []
            action = np.zeros(self.spec.action_dim, dtype=np.float32)
            self.previous = action
            return action
        if not self.buffer:
            plan = self._plan(state)
            self.buffer = [
                plan[index].copy()
                for index in range(max(1, min(self.execute, len(plan))))
            ]
        action = np.clip(
            self.buffer.pop(0),
            self.low.cpu().numpy(),
            self.high.cpu().numpy(),
        ).astype(np.float32)
        self.previous = action.copy()
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=76000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-memory-pairs", type=int, default=100000)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--pose-pool", type=int, default=2048)
    parser.add_argument("--execute", type=int, default=4)
    parser.add_argument("--position-step", type=float, default=0.006)
    parser.add_argument("--rotation-step-deg", type=float, default=15.0)
    parser.add_argument(
        "--selection", choices=["nearest", "model"], default="model"
    )
    parser.add_argument("--slew-limit", type=float, default=0.35)
    parser.add_argument("--stop-position-cm", type=float, default=1.0)
    parser.add_argument("--stop-rotation-deg", type=float, default=5.7)
    parser.add_argument("--local-goal-rotation-deg", type=float, default=0.0)
    parser.add_argument("--local-goal-position-cm", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-episodes", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    task = resolve_task(args.task, None)
    world_model, normalizer, spec, _ = load_jepa_artifact(
        args.model_path, device
    )
    memory = build_memory(
        args.episodes_npz,
        world_model,
        normalizer,
        spec,
        device,
        horizon=args.horizon,
        max_pairs=args.max_memory_pairs,
        seed=args.seed,
    )
    env = make_env(
        task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps
    )
    low = torch.as_tensor(
        env.action_space.low, dtype=torch.float32, device=device
    )
    high = torch.as_tensor(
        env.action_space.high, dtype=torch.float32, device=device
    )
    controller = TransitionMemoryController(
        world_model,
        normalizer,
        spec,
        memory,
        device,
        low,
        high,
        horizon=args.horizon,
        candidates=args.candidates,
        pose_pool=args.pose_pool,
        execute=args.execute,
        position_step=args.position_step,
        rotation_step=np.radians(args.rotation_step_deg),
        selection=args.selection,
        slew_limit=args.slew_limit,
        stop_position=args.stop_position_cm / 100.0,
        stop_rotation=np.radians(args.stop_rotation_deg),
    )
    successes, costs, starts, bests = [], [], [], []
    positions, rotations = [], []
    action_delta_sq, action_jerk_sq = [], []
    goal_rng = np.random.default_rng(args.seed + 991)
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        if args.local_goal_rotation_deg > 0 or args.local_goal_position_cm > 0:
            state = flatten_obs(observation)
            pose = state[spec.obs_dim:spec.obs_dim + 7].copy()
            axis = goal_rng.normal(size=3).astype(np.float32)
            axis /= np.linalg.norm(axis) + 1e-9
            target_q = quat_mul(
                quat_exp(axis * np.radians(args.local_goal_rotation_deg)),
                pose[3:],
            )
            direction = goal_rng.normal(size=3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-9
            target_p = (
                pose[:3]
                + direction * (args.local_goal_position_cm / 100.0)
            )
            env.unwrapped.goal = np.concatenate(
                [target_p, quat_normalize(target_q)]
            ).astype(np.float32)
            observation = env.unwrapped._get_obs()
        controller.reset()
        state = flatten_obs(observation)
        start = pose_cost(
            state[spec.obs_dim:spec.obs_dim + 7],
            state[spec.obs_dim + 7:spec.obs_dim + 14],
        )
        best = start
        terminated = truncated = False
        info = {}
        actions = []
        while not (terminated or truncated):
            action = controller.act(observation)
            actions.append(action.copy())
            observation, _, terminated, truncated, info = env.step(action)
            live = flatten_obs(observation)
            best = min(
                best,
                pose_cost(
                    live[spec.obs_dim:spec.obs_dim + 7],
                    live[spec.obs_dim + 7:spec.obs_dim + 14],
                ),
            )
        state = flatten_obs(observation)
        achieved = state[spec.obs_dim:spec.obs_dim + 7]
        desired = state[spec.obs_dim + 7:spec.obs_dim + 14]
        q_error = quat_mul(
            quat_normalize(desired[3:]),
            quat_conjugate(quat_normalize(achieved[3:])),
        )
        successes.append(float(info.get("is_success", 0.0)))
        starts.append(start)
        bests.append(best)
        costs.append(pose_cost(achieved, desired))
        positions.append(float(np.linalg.norm(achieved[:3] - desired[:3])))
        rotations.append(float(np.linalg.norm(quat_log(q_error))))
        episode_actions = np.asarray(actions, dtype=np.float32)
        deltas = np.diff(
            np.concatenate(
                [np.zeros((1, spec.action_dim), np.float32), episode_actions],
                axis=0,
            ),
            axis=0,
        )
        jerks = np.diff(deltas, axis=0)
        action_delta_sq.extend(np.square(deltas).reshape(-1).tolist())
        action_jerk_sq.extend(np.square(jerks).reshape(-1).tolist())
        if args.log_episodes:
            print(
                json.dumps(
                    {
                        "event": "transition_memory_episode",
                        "episode": episode,
                        "success": successes[-1],
                        "start_goal_cost": round(start, 4),
                        "best_goal_cost": round(best, 4),
                        "final_goal_cost": round(costs[-1], 4),
                    }
                ),
                flush=True,
            )
    env.close()
    result = {
        "event": "dexterous_transition_memory_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "episodes_npz": str(args.episodes_npz),
        "episodes": args.episodes,
        "success_rate": float(np.mean(successes)),
        "median_start_goal_cost": float(np.median(starts)),
        "median_best_goal_cost": float(np.median(bests)),
        "median_final_goal_cost": float(np.median(costs)),
        "median_improvement": float(
            np.median(np.asarray(starts) - np.asarray(costs))
        ),
        "improved_fraction": float(
            np.mean(np.asarray(costs) < np.asarray(starts))
        ),
        "median_final_position_cm": float(100.0 * np.median(positions)),
        "median_final_rotation_deg": float(np.degrees(np.median(rotations))),
        "action_delta_rms": float(np.sqrt(np.mean(action_delta_sq))),
        "action_jerk_rms": float(np.sqrt(np.mean(action_jerk_sq))),
        "memory_pairs": len(memory.pose_deltas),
        "horizon": args.horizon,
        "candidates": args.candidates,
        "pose_pool": args.pose_pool,
        "execute": args.execute,
        "selection": args.selection,
        "slew_limit": args.slew_limit,
        "stop_position_cm": args.stop_position_cm,
        "stop_rotation_deg": args.stop_rotation_deg,
        "local_goal_rotation_deg": args.local_goal_rotation_deg,
        "local_goal_position_cm": args.local_goal_position_cm,
        "seed": args.seed,
    }
    print(json.dumps(result), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
