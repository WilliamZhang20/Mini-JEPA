"""Evaluate inverse-prior action chunks selected by JEPA dynamics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, goal_state_from_state, make_env
from jepa_robotics.evaluate import load_jepa_artifact, rollout_policy
from jepa_robotics.subgoals import load_subgoal_artifact, make_latent_subgoal_target_state
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.task_families.fetch import geometry_features as fetch_geometry_features
from jepa_robotics.algos.priors import InversePrior


class InverseJEPAChunkPolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        prior,
        ckpt: dict,
        device,
        candidates: int,
        noise_std: float,
        exec_k: int,
        goal_mode: str,
        subgoal_artifact: dict | None,
        target_horizon: int | None,
        latent_weight: float,
        state_weight: float,
        final_goal_weight: float,
        action_l2_weight: float,
        action_delta_weight: float,
        action_scale: float,
    ) -> None:
        self.name = f"jepa_inverse_chunk_select_{goal_mode}_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.prior = prior
        self.ckpt = ckpt
        self.device = device
        self.candidates = candidates
        self.noise_std = noise_std
        self.exec_k = exec_k
        self.goal_mode = goal_mode
        self.subgoal_artifact = subgoal_artifact
        self.target_horizon = target_horizon
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.final_goal_weight = final_goal_weight
        self.action_l2_weight = action_l2_weight
        self.action_delta_weight = action_delta_weight
        self.action_scale = action_scale
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.cached: list[np.ndarray] = []

    def reset(self) -> None:
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.cached = []

    def _target_state(self, raw_state: np.ndarray) -> np.ndarray:
        if self.goal_mode == "local":
            if self.subgoal_artifact is None:
                raise ValueError("goal_mode=local requires a subgoal artifact")
            return make_latent_subgoal_target_state(raw_state, self.spec, self.subgoal_artifact)[1]
        return goal_state_from_state(raw_state, self.spec)

    @torch.no_grad()
    def _plan(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        raw_state = flatten_obs(obs)
        target_state = self._target_state(raw_state)
        state = torch.from_numpy(self.normalizer.encode(raw_state)).unsqueeze(0).to(self.device)
        target = torch.from_numpy(self.normalizer.encode(target_state)).unsqueeze(0).to(self.device)
        z = self.wm.encode(state)
        z_goal = self.wm.encode_target(target)
        future_horizons = list(self.ckpt.get("future_horizons", [int(self.ckpt["H"])]))
        target_h = int(self.target_horizon or int(self.ckpt["H"]))
        h_token = torch.tensor([[float(target_h) / float(max(future_horizons))]], dtype=z.dtype, device=self.device)
        cond = torch.cat([z, z_goal, h_token], dim=-1)
        if bool(self.ckpt.get("concat_geometry", False)):
            geom = fetch_geometry_features(raw_state, target_state, self.spec)
            mean = np.asarray(self.ckpt["geom_mean"], dtype=np.float32)
            std = np.asarray(self.ckpt["geom_std"], dtype=np.float32)
            geom_t = torch.from_numpy((geom - mean) / np.maximum(std, 1e-6)).unsqueeze(0).to(self.device)
            cond = torch.cat([cond, geom_t], dim=-1)
        base = self.prior(cond).view(1, int(self.ckpt["H"]), int(self.ckpt["action_dim"])) * self.action_scale
        if self.candidates > 1 and self.noise_std > 0.0:
            noise = torch.randn(self.candidates, *base.shape[1:], device=self.device) * self.noise_std
            action_tensor = base.repeat(self.candidates, 1, 1) + noise
            action_tensor[0] = base[0]
        else:
            action_tensor = base.repeat(self.candidates, 1, 1)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.clamp(low, high)
        H = int(self.ckpt["H"])
        traj_z = self.wm.predict_rollout(z.repeat(self.candidates, 1), action_tensor, H)
        scores = self.latent_weight * torch.sum(
            (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        if self.state_weight > 0.0 or self.final_goal_weight > 0.0:
            pred_state = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            pred_achieved = pred_state[..., self.spec.obs_dim : self.spec.obs_dim + self.spec.goal_dim]
            target_achieved = torch.as_tensor(
                target_state[self.spec.obs_dim : self.spec.obs_dim + self.spec.goal_dim],
                dtype=pred_state.dtype,
                device=self.device,
            ).view(1, 1, -1)
            local_dist = torch.linalg.norm(pred_achieved - target_achieved, dim=-1)
            scores = scores + self.state_weight * (local_dist[:, -1] + 0.25 * local_dist.mean(dim=1))
            final_goal = torch.as_tensor(
                raw_state[self.spec.obs_dim + self.spec.goal_dim : self.spec.obs_dim + 2 * self.spec.goal_dim],
                dtype=pred_state.dtype,
                device=self.device,
            ).view(1, 1, -1)
            final_dist = torch.linalg.norm(pred_achieved - final_goal, dim=-1)
            scores = scores + self.final_goal_weight * (final_dist[:, -1] + 0.25 * final_dist.mean(dim=1))
        if self.action_l2_weight > 0.0:
            scores = scores + self.action_l2_weight * action_tensor.square().mean(dim=(1, 2))
        if self.action_delta_weight > 0.0:
            prev = torch.as_tensor(self.prev_action, dtype=torch.float32, device=self.device).view(1, 1, -1)
            first_delta = action_tensor[:, :1] - prev
            seq_delta = action_tensor[:, 1:] - action_tensor[:, :-1]
            scores = scores + self.action_delta_weight * torch.cat([first_delta, seq_delta], dim=1).square().mean(dim=(1, 2))
        best = int(torch.argmin(scores).detach().cpu())
        return action_tensor[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        if not self.cached:
            plan = self._plan(obs, env)
            k = max(1, min(self.exec_k, len(plan)))
            self.cached = [plan[i].copy() for i in range(k)]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
        self.prev_action = action
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--inverse-path", type=Path, required=True)
    p.add_argument("--subgoal-path", type=Path, default=None)
    p.add_argument("--goal-mode", choices=["final", "local"], default="final")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--noise-std", type=float, default=0.15)
    p.add_argument("--exec-k", type=int, default=24)
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.0)
    p.add_argument("--final-goal-weight", type=float, default=0.0)
    p.add_argument("--action-l2-weight", type=float, default=0.0)
    p.add_argument("--action-delta-weight", type=float, default=0.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--video-episodes", type=int, default=1,
                   help="Number of consecutive episodes to concatenate into the video.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    task = resolve_task(args.task, None)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    wm, normalizer, spec, cfg = load_jepa_artifact(args.model_path, device)
    ckpt = torch.load(args.inverse_path, map_location=device, weights_only=False)
    prior = InversePrior(int(ckpt["cond_dim"]), int(ckpt["chunk_dim"]), int(ckpt["hidden"]), int(ckpt["n_blocks"])).to(device)
    prior.load_state_dict(ckpt["state_dict"])
    prior.eval()
    subgoal_artifact = load_subgoal_artifact(args.subgoal_path) if args.subgoal_path is not None else None
    if args.goal_mode == "local" and subgoal_artifact is None:
        raise ValueError("--goal-mode local requires --subgoal-path")
    env = make_env(
        task.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps or task.max_episode_steps,
        render_mode="rgb_array" if args.video_out is not None else None,
        width=args.width if args.video_out is not None else None,
        height=args.height if args.video_out is not None else None,
    )
    policy = InverseJEPAChunkPolicy(
        wm=wm, normalizer=normalizer, spec=spec, prior=prior, ckpt=ckpt, device=device,
        candidates=args.candidates, noise_std=args.noise_std, exec_k=args.exec_k,
        goal_mode=args.goal_mode, subgoal_artifact=subgoal_artifact,
        target_horizon=args.target_horizon, latent_weight=args.latent_weight,
        state_weight=args.state_weight, final_goal_weight=args.final_goal_weight,
        action_l2_weight=args.action_l2_weight, action_delta_weight=args.action_delta_weight,
        action_scale=args.action_scale,
    )
    metrics = rollout_policy(
        env,
        policy,
        episodes=args.episodes,
        seed=args.seed,
        video_path=args.video_out,
        video_episodes=min(args.video_episodes, args.episodes),
        fps=args.fps,
    )
    env.close()
    row = {"event": "inverse_jepa_eval", "task": task.name, "model_path": str(args.model_path), "inverse_path": str(args.inverse_path), "model_config": cfg, **metrics}
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
