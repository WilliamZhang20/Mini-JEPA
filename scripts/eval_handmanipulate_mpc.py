"""Goal-conditioned latent MPC for the Shadow Hand HandManipulate suite.

Pure SSL control: a DexterousJEPA world model (trained self-supervised on
exploration and no demos) plus the env-provided target object pose. At each
step, CEM over action sequences: roll candidates through the world model, decode
the predicted object pose (achieved_goal slice) via the state probe, and score
distance to the desired object pose. Execute the first action(s), replan (MPC).

HandManipulate is a difficult dexterous-control benchmark; this is
the principled demo-free SSL controller for it, reported honestly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.algos.priors import make_ddpm, sample_action_chunks
from jepa_robotics.algos.planning.objectives import CommonScoringMixin
from jepa_robotics.algos.task_families.dexterous import (
    pose_cost,
    quat_exp,
    quat_mul,
    quat_normalize,
)
from jepa_robotics.tasks import resolve_task
from jepa_robotics.models import DexterousFlowPrior


class LatentMPC(CommonScoringMixin):
    def __init__(self, wm, norm, spec, dev, horizon, candidates, iters, elite_frac,
                 init_std, exec_k, disagree_weight, action_l2_weight=0.0,
                 action_delta_weight=0.0, slew_limit=0.0,
                 score_mode="pose", play_flow=None, play_flow_config=None,
                 search_depth=1, beam_width=8, branch_factor=32,
                 stop_position=0.0, stop_rotation=0.0):
        self.wm, self.norm, self.spec, self.dev = wm, norm, spec, dev
        self.H, self.N, self.iters = horizon, candidates, iters
        self.elite = max(1, int(candidates * elite_frac))
        self.init_std, self.exec_k, self.disagree_w = init_std, exec_k, disagree_weight
        self.action_l2_weight = action_l2_weight
        self.action_delta_weight = action_delta_weight
        self.slew_limit = slew_limit
        self.score_mode = score_mode
        self.context_len = int(getattr(wm, "context_len", 1))
        self.play_flow = play_flow
        self.play_flow_config = play_flow_config
        self.search_depth = int(search_depth)
        self.beam_width = int(beam_width)
        self.branch_factor = int(branch_factor)
        self.stop_position = float(stop_position)
        self.stop_rotation = float(stop_rotation)
        self.play_ddpm = (
            make_ddpm(int(play_flow_config["flow_steps"]), dev)
            if play_flow is not None
            else None
        )
        self.ag_lo = spec.obs_dim                      # achieved_goal slice in flat state
        self.ag_hi = spec.obs_dim + spec.goal_dim
        self.dg_lo = spec.obs_dim + spec.goal_dim      # desired_goal slice
        self.dg_hi = spec.obs_dim + 2 * spec.goal_dim
        self.ag_mean = torch.as_tensor(
            norm.mean[self.ag_lo:self.ag_hi], dtype=torch.float32, device=dev
        )
        self.ag_std = torch.as_tensor(
            norm.std[self.ag_lo:self.ag_hi], dtype=torch.float32, device=dev
        )
        self._buf = []

    def reset(self):
        self._buf = []
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        self._state_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []

    def _observe(self, raw: np.ndarray) -> None:
        """Add one real transition to the short JEPA context."""
        normalized = self.norm.encode(raw)
        if self._state_history:
            self._action_history.append(self.prev_action.copy())
        self._state_history.append(normalized)
        self._state_history = self._state_history[-self.context_len :]
        if self.context_len > 1:
            self._action_history = self._action_history[
                -(self.context_len - 1) :
            ]
        else:
            self._action_history = []

    def _context_tensors(self):
        """Left-pad the first few environment steps with a stationary context."""
        states = list(self._state_history)
        if not states:
            raise RuntimeError("MPC context is empty")
        states = [states[0]] * (self.context_len - len(states)) + states
        actions = list(self._action_history)
        actions = [
            np.zeros(self.spec.action_dim, dtype=np.float32)
        ] * (self.context_len - 1 - len(actions)) + actions
        state_tensor = torch.from_numpy(np.stack(states)).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            z_history = self.wm.encode(
                state_tensor.reshape(self.context_len, -1)
            ).reshape(1, self.context_len, -1)
        if actions:
            action_history = (
                torch.from_numpy(np.stack(actions)).unsqueeze(0).to(self.dev)
            )
        else:
            action_history = torch.zeros(
                1, 0, self.spec.action_dim, device=self.dev
            )
        return z_history, action_history

    def _rollout(self, z_history, action_history, actions):
        if hasattr(self.wm, "predict_rollout_context"):
            return self.wm.predict_rollout_context(
                z_history.expand(actions.shape[0], -1, -1),
                action_history.expand(actions.shape[0], -1, -1),
                actions,
                self.H,
            )
        return self.wm.predict_rollout(
            z_history[:, -1].expand(actions.shape[0], -1),
            actions,
            self.H,
        )

    def _sample_play_actions(
        self, z_history, action_history, samples_per_context: int
    ) -> torch.Tensor:
        """Sample smooth, state-conditioned chunks from reward-free play."""
        if self.play_flow is None:
            raise RuntimeError("No play prior is loaded")
        condition = torch.cat(
            [z_history.flatten(1), action_history.flatten(1)], dim=-1
        ).repeat_interleave(samples_per_context, dim=0)
        candidates = len(condition)
        sampled = sample_action_chunks(
            self.play_flow,
            self.play_ddpm,
            condition,
            int(self.play_flow_config["chunk_dim"]),
            self.dev,
            objective="flow",
            flow_steps=int(self.play_flow_config.get("sample_steps", 12)),
        ).reshape(candidates, self.H, self.spec.action_dim)
        increment_mean = torch.as_tensor(
            self.play_flow_config["increment_mean"],
            dtype=sampled.dtype,
            device=self.dev,
        )
        increment_std = torch.as_tensor(
            self.play_flow_config["increment_std"],
            dtype=sampled.dtype,
            device=self.dev,
        )
        increments = sampled * increment_std + increment_mean
        if action_history.shape[1] > 0:
            previous = action_history[:, -1]
        else:
            previous = torch.as_tensor(
                self.prev_action, dtype=sampled.dtype, device=self.dev
            ).view(1, -1).expand(len(z_history), -1)
        previous = previous.repeat_interleave(
            samples_per_context, dim=0
        ).unsqueeze(1)
        actions = previous + torch.cumsum(increments, dim=1)
        if self.slew_limit > 0:
            limited = []
            live = previous[:, 0]
            for step in range(self.H):
                live = live + (actions[:, step] - live).clamp(
                    -self.slew_limit, self.slew_limit
                )
                limited.append(live)
            actions = torch.stack(limited, dim=1)
        return actions

    def _physical_endpoint_cost(
        self, rollout: torch.Tensor, raw_goal: torch.Tensor
    ) -> torch.Tensor:
        object_pred = (
            self.wm.predict_object(rollout[:, -1])
            if hasattr(self.wm, "predict_object")
            else None
        )
        if object_pred is not None:
            achieved = object_pred * self.ag_std + self.ag_mean
        else:
            state = self.norm.decode_tensor(
                self.wm.state_probe(rollout[:, -1])
            )
            achieved = state[:, self.ag_lo:self.ag_hi]
        position = torch.linalg.vector_norm(
            achieved[:, :3] - raw_goal[:3], dim=-1
        )
        qa = torch.nn.functional.normalize(achieved[:, 3:], dim=-1)
        qb = torch.nn.functional.normalize(raw_goal[3:], dim=-1)
        rotation = 2.0 * torch.acos(
            (qa * qb).sum(-1).abs().clamp(max=1.0)
        )
        return 10.0 * position + rotation

    def _play_beam_plan(
        self,
        z_history: torch.Tensor,
        action_history: torch.Tensor,
        raw_goal: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        """Compose reward-free play primitives with latent beam search."""
        nodes_z = z_history
        nodes_a = action_history
        first_chunks = None
        best_cost = float("inf")
        best_first = None
        for depth in range(self.search_depth):
            chunks = self._sample_play_actions(
                nodes_z, nodes_a, self.branch_factor
            ).clamp(low, high)
            parents = torch.arange(
                len(nodes_z), device=self.dev
            ).repeat_interleave(self.branch_factor)
            expanded_z = nodes_z[parents]
            expanded_a = nodes_a[parents]
            rollout = self.wm.predict_rollout_context(
                expanded_z, expanded_a, chunks, self.H
            )
            cost = self._physical_endpoint_cost(rollout, raw_goal)
            candidate_first = (
                chunks
                if first_chunks is None
                else first_chunks[parents]
            )
            local = int(torch.argmin(cost))
            if float(cost[local]) < best_cost:
                best_cost = float(cost[local])
                best_first = candidate_first[local]
            keep = torch.argsort(cost)[: self.beam_width]
            if self.context_len <= self.H:
                nodes_z = rollout[keep, -self.context_len :]
            else:
                nodes_z = torch.cat(
                    [expanded_z[keep, self.H :], rollout[keep]], dim=1
                )
            if self.context_len > 1:
                nodes_a = chunks[keep, -(self.context_len - 1) :]
            else:
                nodes_a = chunks.new_zeros(
                    len(keep), 0, self.spec.action_dim
                )
            first_chunks = candidate_first[keep]
        return best_first

    @torch.no_grad()
    def _plan(self, raw, env):
        A = self.spec.action_dim
        z_history, action_history = self._context_tensors()
        # Score in the environment's PHYSICAL goal geometry, not z-scored state
        # space: quaternion components cannot be compared component-wise (a
        # sign-equivalent quaternion is the same orientation but has large
        # per-dim error). Use the simulator's own dense surrogate,
        # 10*position_distance + quaternion_geodesic_angle.
        raw_dg = torch.as_tensor(raw[self.dg_lo:self.dg_hi], dtype=torch.float32, device=self.dev)
        target_z = None
        if self.score_mode == "latent-object":
            target_raw = raw.copy()
            target_raw[self.ag_lo:self.ag_hi] = raw[self.dg_lo:self.dg_hi]
            target_state = torch.from_numpy(
                self.norm.encode(target_raw)
            ).unsqueeze(0).to(self.dev)
            target_z = self.wm.encode_target(target_state)
            slots = int(getattr(self.wm, "latent_slots", 1))
            slot_dim = int(self.wm.latent_dim) // slots
            target_z = target_z[:, :slot_dim]
        lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.dev)
        hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.dev)
        if self.play_flow is not None and self.search_depth > 1:
            return self._play_beam_plan(
                z_history, action_history, raw_dg, lo, hi
            ).cpu().numpy()
        mean = torch.zeros(self.H, A, device=self.dev)
        std = torch.full((self.H, A), self.init_std, device=self.dev)
        best_first = None
        for iteration in range(self.iters):
            if iteration == 0 and self.play_flow is not None:
                acts = self._sample_play_actions(
                    z_history, action_history, self.N
                ).clamp(lo, hi)
            else:
                eps = torch.randn(self.N, self.H, A, device=self.dev)
                acts = (
                    mean.unsqueeze(0) + std.unsqueeze(0) * eps
                ).clamp(lo, hi)
            acts = self._rate_limit_actions(acts)
            roll = self._rollout(z_history, action_history, acts)
            if self.score_mode == "latent-object":
                slot_dim = target_z.shape[-1]
                latent_dist = torch.square(
                    roll[:, :, :slot_dim] - target_z.unsqueeze(1)
                ).mean(dim=-1)
                cost = latent_dist[:, -1] + 0.25 * latent_dist.mean(1)
            else:
                object_pred = (
                    self.wm.predict_object(roll)
                    if hasattr(self.wm, "predict_object")
                    else None
                )
                if object_pred is not None:
                    pred_ag = object_pred * self.ag_std + self.ag_mean
                else:
                    pred = self.norm.decode_tensor(self.wm.state_probe(roll))
                    pred_ag = pred[:, :, self.ag_lo:self.ag_hi]
                pos_dist = torch.linalg.vector_norm(
                    pred_ag[:, :, :3] - raw_dg[:3], dim=-1
                )
                qa = pred_ag[:, :, 3:]
                qa = qa / torch.linalg.vector_norm(
                    qa, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                qb = raw_dg[3:].view(1, 1, 4)
                qb = qb / torch.linalg.vector_norm(
                    qb, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                dot = (qa * qb).sum(-1).abs().clamp(max=1.0)
                rot_dist = 2.0 * torch.acos(dot)
                dist = 10.0 * pos_dist + rot_dist
                cost = dist[:, -1] + 0.25 * dist.mean(1)
            if self.disagree_w > 0 and getattr(self.wm, "ensemble_heads", 1) > 1:
                # per-candidate epistemic uncertainty (batch-mean disagreement()
                # would add the same scalar to every candidate and do nothing)
                if hasattr(self.wm, "rollout_heads_context"):
                    head_rolls = self.wm.rollout_heads_context(
                        z_history.expand(self.N, -1, -1),
                        action_history.expand(self.N, -1, -1),
                        acts,
                        self.H,
                    )
                else:
                    head_rolls = self.wm.rollout_heads(
                        z_history[:, -1].expand(self.N, -1), acts, self.H
                    )
                cost = cost + self.disagree_w * head_rolls.var(dim=0).mean(dim=(1, 2))
            cost = cost + self._action_regularizers(acts)
            order = torch.argsort(cost)
            elite = acts[order[: self.elite]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.02)
            best_first = acts[order[0]]
        return best_first.cpu().numpy()

    def act(self, obs, env):
        raw = flatten_obs(obs)
        self._observe(raw)
        if not self._buf:
            achieved = raw[self.ag_lo:self.ag_hi]
            desired = raw[self.dg_lo:self.dg_hi]
            position = float(np.linalg.norm(achieved[:3] - desired[:3]))
            qa = achieved[3:] / (np.linalg.norm(achieved[3:]) + 1e-9)
            qb = desired[3:] / (np.linalg.norm(desired[3:]) + 1e-9)
            rotation = float(
                2.0 * np.arccos(min(1.0, abs(float(qa @ qb))))
            )
            if (
                self.stop_position > 0
                and self.stop_rotation > 0
                and position <= self.stop_position
                and rotation <= self.stop_rotation
            ):
                plan = np.repeat(
                    self.prev_action[None], self.H, axis=0
                )
            else:
                plan = self._plan(raw, env)
            self._buf = [plan[i].copy() for i in range(max(1, min(self.exec_k, len(plan))))]
        action = np.clip(
            self._buf.pop(0), env.action_space.low, env.action_space.high
        ).astype(np.float32)
        if self.slew_limit > 0:
            action = np.clip(
                self.prev_action
                + np.clip(action - self.prev_action, -self.slew_limit, self.slew_limit),
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
        self.prev_action = action.copy()
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True, help="DexterousJEPA artifact (arch=dexterous)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=40000)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--candidates", type=int, default=512)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--exec-k", type=int, default=2)
    p.add_argument("--disagree-weight", type=float, default=0.0)
    p.add_argument("--action-l2-weight", type=float, default=0.0)
    p.add_argument("--action-delta-weight", type=float, default=0.0)
    p.add_argument("--slew-limit", type=float, default=0.0)
    p.add_argument(
        "--score-mode",
        choices=["pose", "latent-object"],
        default="pose",
        help="Decoded physical pose cost or direct L2 distance in the JEPA object slot.",
    )
    p.add_argument(
        "--play-prior-path",
        type=Path,
        default=None,
        help="Reward-free state-conditioned action-chunk flow used for the first CEM population.",
    )
    p.add_argument("--play-prior-sample-steps", type=int, default=12)
    p.add_argument("--search-depth", type=int, default=1)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--branch-factor", type=int, default=32)
    p.add_argument("--stop-position-cm", type=float, default=0.0)
    p.add_argument("--stop-rotation-deg", type=float, default=0.0)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--local-goal-rotation-deg", type=float, default=0.0)
    p.add_argument("--local-goal-position-cm", type=float, default=0.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log-episodes", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    play_flow = None
    play_flow_config = None
    if args.play_prior_path is not None:
        play_artifact = torch.load(
            args.play_prior_path, map_location=dev, weights_only=False
        )
        play_flow_config = dict(play_artifact["config"])
        if int(play_flow_config["chunk"]) != args.horizon:
            raise ValueError(
                "Play-prior chunk must match the MPC horizon "
                f"({play_flow_config['chunk']} != {args.horizon})"
            )
        play_flow_config["sample_steps"] = args.play_prior_sample_steps
        play_flow = DexterousFlowPrior(
            int(play_flow_config["chunk_dim"]),
            int(play_flow_config["condition_dim"]),
            hidden=int(play_flow_config["hidden"]),
            n_blocks=int(play_flow_config["blocks"]),
            heads=int(play_flow_config["heads"]),
            action_dim=int(play_flow_config["action_dim"]),
        ).to(dev)
        play_flow.load_state_dict(
            play_artifact.get("ema", play_artifact["state_dict"])
        )
        play_flow.eval()
        for parameter in play_flow.parameters():
            parameter.requires_grad_(False)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    mpc = LatentMPC(wm, norm, spec, dev, args.horizon, args.candidates, args.iters,
                    args.elite_frac, args.init_std, args.exec_k, args.disagree_weight,
                    args.action_l2_weight, args.action_delta_weight, args.slew_limit,
                    args.score_mode, play_flow, play_flow_config,
                    args.search_depth, args.beam_width, args.branch_factor,
                    args.stop_position_cm / 100.0,
                    np.radians(args.stop_rotation_deg))

    successes, gaps, start_gaps, best_gaps = [], [], [], []
    position_errors, rotation_errors = [], []
    action_delta_sq, action_jerk_sq = [], []
    goal_rng = np.random.default_rng(args.seed + 991)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.local_goal_rotation_deg > 0 or args.local_goal_position_cm > 0:
            reset_state = flatten_obs(obs)
            current_pose = reset_state[spec.obs_dim:spec.obs_dim + 7].copy()
            axis = goal_rng.normal(size=3).astype(np.float32)
            axis /= np.linalg.norm(axis) + 1e-9
            target_q = quat_mul(
                quat_exp(axis * np.radians(args.local_goal_rotation_deg)),
                current_pose[3:],
            )
            direction = goal_rng.normal(size=3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-9
            target_position = (
                current_pose[:3]
                + direction * (args.local_goal_position_cm / 100.0)
            )
            env.unwrapped.goal = np.concatenate(
                [target_position, quat_normalize(target_q)]
            ).astype(np.float32)
            obs = env.unwrapped._get_obs()
        mpc.reset()
        initial = flatten_obs(obs)
        start_gap = pose_cost(
            initial[spec.obs_dim:spec.obs_dim + 7],
            initial[spec.obs_dim + 7:spec.obs_dim + 14],
        )
        best_gap = start_gap
        term = trunc = False; info = {}
        actions: list[np.ndarray] = []
        while not (term or trunc):
            action = mpc.act(obs, env)
            actions.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            live = flatten_obs(obs)
            best_gap = min(
                best_gap,
                pose_cost(
                    live[spec.obs_dim:spec.obs_dim + 7],
                    live[spec.obs_dim + 7:spec.obs_dim + 14],
                ),
            )
        successes.append(float(info.get("is_success", 0.0)))
        raw = flatten_obs(obs)
        achieved = raw[spec.obs_dim:spec.obs_dim + spec.goal_dim]
        desired = raw[spec.obs_dim + spec.goal_dim:spec.obs_dim + 2 * spec.goal_dim]
        pos = float(np.linalg.norm(achieved[:3] - desired[:3]))
        qa = achieved[3:] / (np.linalg.norm(achieved[3:]) + 1e-9)
        qb = desired[3:] / (np.linalg.norm(desired[3:]) + 1e-9)
        rot = float(2.0 * np.arccos(min(1.0, abs(float(qa @ qb)))))
        gaps.append(10.0 * pos + rot)
        start_gaps.append(start_gap)
        best_gaps.append(best_gap)
        position_errors.append(pos)
        rotation_errors.append(rot)
        ep_actions = np.asarray(actions, dtype=np.float32)
        deltas = np.diff(
            np.concatenate([np.zeros((1, spec.action_dim), np.float32), ep_actions], axis=0),
            axis=0,
        )
        jerks = np.diff(deltas, axis=0)
        action_delta_sq.extend(np.square(deltas).reshape(-1).tolist())
        action_jerk_sq.extend(np.square(jerks).reshape(-1).tolist())
        if args.log_episodes:
            print(json.dumps({
                "event": "handmanipulate_mpc_episode",
                "episode": ep,
                "success": successes[-1],
                "start_goal_cost": round(start_gap, 4),
                "best_goal_cost": round(best_gap, 4),
                "final_goal_cost": round(gaps[-1], 4),
                "action_delta_rms": round(float(np.sqrt(np.mean(np.square(deltas)))), 4),
                "action_jerk_rms": round(float(np.sqrt(np.mean(np.square(jerks)))), 4),
            }), flush=True)
    env.close()
    row = {"event": "handmanipulate_mpc_eval", "task": task.name, "model_path": str(args.model_path),
           "episodes": args.episodes, "success_rate": float(np.mean(successes)),
           "median_start_goal_cost": float(np.median(start_gaps)),
           "median_best_goal_cost": float(np.median(best_gaps)),
           "median_final_goal_cost": float(np.median(gaps)),
           "median_improvement": float(np.median(np.asarray(start_gaps) - np.asarray(gaps))),
           "improved_fraction": float(np.mean(np.asarray(gaps) < np.asarray(start_gaps))),
           "median_final_position_cm": float(100.0 * np.median(position_errors)),
           "median_final_rotation_deg": float(np.degrees(np.median(rotation_errors))),
           "action_delta_rms": float(np.sqrt(np.mean(action_delta_sq))),
           "action_jerk_rms": float(np.sqrt(np.mean(action_jerk_sq))),
           "horizon": args.horizon, "candidates": args.candidates, "iters": args.iters,
           "exec_k": args.exec_k, "disagree_weight": args.disagree_weight,
           "action_l2_weight": args.action_l2_weight,
           "action_delta_weight": args.action_delta_weight, "slew_limit": args.slew_limit,
           "score_mode": args.score_mode,
           "play_prior_path": (
               str(args.play_prior_path) if args.play_prior_path else None
           ),
           "search_depth": args.search_depth,
           "beam_width": args.beam_width,
           "branch_factor": args.branch_factor,
           "stop_position_cm": args.stop_position_cm,
           "stop_rotation_deg": args.stop_rotation_deg,
           "local_goal_rotation_deg": args.local_goal_rotation_deg,
           "local_goal_position_cm": args.local_goal_position_cm,
           "seed": args.seed, "torch_seed": args.torch_seed}
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
