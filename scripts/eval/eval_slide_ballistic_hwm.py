"""Evaluate commit-and-coast FetchSlide control with a ballistic HWM."""
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

from jepa_robotics.algos.world_models.ballistic import (
    BallisticHWM,
    EquivariantBallisticHWM,
    canonical_ballistic_features,
    fetch_slide_ready,
    goal_relative_macro_candidates,
    goal_to_world_frame,
    slide_macro_action,
)
from jepa_robotics.data import scripted_slide_action
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact, rollout_policy
from jepa_robotics.tasks import resolve_task


class BallisticHWMPolicy:
    name = "ballistic_hwm_commit_coast"

    def __init__(self, wm, norm, spec, hwm, device, uncertainty_weight,
                 angle_limit, angle_count, amplitudes, feature_mean=None, feature_std=None):
        self.wm, self.norm, self.spec, self.hwm, self.device = wm, norm, spec, hwm, device
        self.uncertainty_weight = uncertainty_weight
        self.angle_limit, self.angle_count, self.amplitudes = angle_limit, angle_count, amplitudes
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.macro = None
        self.duration = 0
        self.strike_step = 0
        self.lift_step = 0
        self.planned_pre = None

    @torch.no_grad()
    def _plan(self, raw):
        self.planned_pre = raw.copy()
        obj = raw[self.spec.obs_dim : self.spec.obs_dim + self.spec.goal_dim]
        goal = raw[self.spec.obs_dim + self.spec.goal_dim : self.spec.obs_dim + 2 * self.spec.goal_dim]
        macros, durations = goal_relative_macro_candidates(
            obj[:2], goal[:2], angle_limit_deg=self.angle_limit,
            angle_count=self.angle_count, amplitudes=self.amplitudes,
        )
        state = torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.device)
        z = self.wm.encode(state).repeat(len(macros), 1)
        macro_t = torch.from_numpy(macros).to(self.device)
        if isinstance(self.hwm, EquivariantBallisticHWM):
            repeated_state = np.repeat(raw[None], len(macros), axis=0)
            feature_np = canonical_ballistic_features(
                repeated_state, macros, self.spec.obs_dim, self.spec.goal_dim
            )
            feature_np = (feature_np - self.feature_mean) / self.feature_std
            features = torch.from_numpy(feature_np.astype(np.float32)).to(self.device)
            _pred_z, pred_canonical = self.hwm(z, features)
            pred_disp_np = goal_to_world_frame(
                pred_canonical.cpu().numpy(),
                repeated_state[:, None, :],
                self.spec.obs_dim,
                self.spec.goal_dim,
            )
            pred_disp = torch.from_numpy(pred_disp_np).to(self.device)
        else:
            side = state.repeat(len(macros), 1) if self.hwm.side_dim else None
            _pred_z, pred_disp = self.hwm(z, macro_t, side)
        endpoint = torch.as_tensor(obj, device=self.device)[None, None] + pred_disp
        target = torch.as_tensor(goal, device=self.device)[None, None]
        mean_endpoint = endpoint.mean(dim=1)
        distance = torch.linalg.norm(mean_endpoint - target[:, 0], dim=-1)
        uncertainty = endpoint.std(dim=1).norm(dim=-1)
        score = distance + self.uncertainty_weight * uncertainty
        best = int(torch.argmin(score).cpu())
        return macros[best], int(durations[best])

    def act(self, obs, env):
        raw = flatten_obs(obs)
        if self.phase == "approach":
            if not fetch_slide_ready(raw, self.spec.obs_dim, self.spec.goal_dim):
                return scripted_slide_action(obs, self.spec.action_dim, 12.0)
            self.macro, self.duration = self._plan(raw)
            self.phase = "strike"
        if self.phase == "strike":
            action = slide_macro_action(self.macro, raw, self.spec.obs_dim,
                                        self.spec.goal_dim, self.spec.action_dim)
            self.strike_step += 1
            if self.strike_step >= self.duration:
                self.phase = "lift"
            return action
        action = np.zeros(self.spec.action_dim, np.float32)
        if self.spec.action_dim >= 4:
            action[3] = -1.0
        if self.phase == "lift":
            action[2] = 1.0
            self.lift_step += 1
            if self.lift_step >= 4:
                self.phase = "coast"
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--hwm-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=14000)
    p.add_argument("--uncertainty-weight", type=float, default=1.0)
    p.add_argument("--angle-limit", type=float, default=45.0)
    p.add_argument("--angle-count", type=int, default=31)
    p.add_argument("--amplitudes", type=int, default=17)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None,
                   help="Record evaluated episodes with the event-HWM controller.")
    p.add_argument("--video-episodes", type=int, default=6,
                   help="Number of consecutive episodes to concatenate into one video.")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--trials-out", type=Path, default=None,
                   help="Save the planner's selected macro and observed coast endpoint for self-supervised HWM recalibration.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    task = resolve_task(args.task, None)
    wm, norm, spec, _cfg = load_jepa_artifact(args.model_path, device)
    ckpt = torch.load(args.hwm_path, map_location=device, weights_only=False)
    if ckpt.get("architecture") == "equivariant_v2":
        hwm = EquivariantBallisticHWM(
            int(ckpt["latent_dim"]),
            int(ckpt["feature_dim"]),
            int(ckpt["hidden"]),
            int(ckpt["heads"]),
            int(ckpt["blocks"]),
        ).to(device)
        feature_mean = np.asarray(ckpt["feature_mean"], np.float32)
        feature_std = np.asarray(ckpt["feature_std"], np.float32)
    else:
        hwm = BallisticHWM(int(ckpt["latent_dim"]), int(ckpt["macro_dim"]),
                           int(ckpt["hidden"]), int(ckpt["heads"]),
                           side_dim=int(ckpt.get("side_dim", 0))).to(device)
        feature_mean = feature_std = None
    hwm.load_state_dict(ckpt["state_dict"])
    hwm.eval()
    policy = BallisticHWMPolicy(wm, norm, spec, hwm, device, args.uncertainty_weight,
                                args.angle_limit, args.angle_count, args.amplitudes,
                                feature_mean, feature_std)
    env = make_env(
        task.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps or task.max_episode_steps,
        render_mode="rgb_array" if args.video_out is not None else None,
        width=args.width if args.video_out is not None else None,
        height=args.height if args.video_out is not None else None,
    )
    if args.trials_out is None:
        metrics = rollout_policy(
            env,
            policy,
            episodes=args.episodes,
            seed=args.seed,
            video_path=args.video_out,
            video_episodes=min(args.video_episodes, args.episodes),
            fps=args.fps,
        )
    else:
        pre_states, final_states, macros, durations = [], [], [], []
        successes, distances, lengths = [], [], []
        action_norms, action_deltas = [], []
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            policy.reset()
            term = trunc = False
            info = {}
            steps = 0
            prev_action = None
            while not (term or trunc):
                action = policy.act(obs, env)
                action_norms.append(float(np.linalg.norm(action)))
                if prev_action is not None:
                    action_deltas.append(float(np.linalg.norm(action - prev_action)))
                prev_action = action.copy()
                obs, _, term, trunc, info = env.step(action)
                steps += 1
            final = flatten_obs(obs).copy()
            achieved = final[spec.obs_dim : spec.obs_dim + spec.goal_dim]
            desired = final[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
            successes.append(float(info.get("is_success", 0.0)))
            distances.append(float(np.linalg.norm(achieved - desired)))
            lengths.append(float(steps))
            if policy.planned_pre is not None and policy.macro is not None:
                pre_states.append(policy.planned_pre)
                final_states.append(final)
                macros.append(policy.macro)
                durations.append(policy.duration)
        metrics = {
            "policy": policy.name, "episodes": float(args.episodes),
            "success_rate": float(np.mean(successes)),
            "mean_final_distance": float(np.mean(distances)),
            "mean_episode_length": float(np.mean(lengths)),
            "mean_action_norm": float(np.mean(action_norms)),
            "mean_action_delta": float(np.mean(action_deltas)),
        }
        args.trials_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trials_out,
            pre_states=np.asarray(pre_states, np.float32),
            final_states=np.asarray(final_states, np.float32),
            macros=np.asarray(macros, np.float32),
            durations=np.asarray(durations, np.int64),
            successes=np.asarray(successes[: len(pre_states)], np.float32),
            final_distances=np.asarray(distances[: len(pre_states)], np.float32),
            obs_dim=np.asarray(spec.obs_dim), goal_dim=np.asarray(spec.goal_dim),
            action_dim=np.asarray(spec.action_dim), max_duration=np.asarray(10),
        )
    env.close()
    row = {"event": "slide_ballistic_hwm_eval", "hwm_path": str(args.hwm_path), **metrics}
    print(json.dumps(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
