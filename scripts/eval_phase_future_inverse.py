"""Evaluate phase-conditioned future inverse priors on flat demo tasks."""
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

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact, rollout_policy
from jepa_robotics.algos.phase import PhaseFutureIndex, phase_features
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.priors import InversePrior


def load_progress_head(path: Path | None, device):
    if path is None:
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = MLP([int(ckpt["latent_dim"]), int(ckpt["hidden"]), int(ckpt["hidden"]), 1], layer_norm=True).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


class PhaseInversePolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        prior,
        progress_head,
        ckpt,
        future_index,
        device,
        candidates: int,
        noise_std: float,
        exec_k: int,
        phase_window: int,
        phase_mode: str,
        monotone_phase: bool,
        max_steps: int,
        target_horizon: int | None,
        latent_weight: float,
        state_weight: float,
        progress_weight: float,
        progress_adv_weight: float,
        action_delta_weight: float,
        action_scale: float,
    ) -> None:
        self.name = f"phase_inverse_jepa_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.prior = prior
        self.progress_head = progress_head
        self.ckpt = ckpt
        self.future_index = future_index
        self.device = device
        self.candidates = candidates
        self.noise_std = noise_std
        self.exec_k = exec_k
        self.phase_window = phase_window
        self.phase_mode = phase_mode
        self.monotone_phase = monotone_phase
        self.max_steps = max_steps
        self.target_horizon = target_horizon
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.progress_weight = progress_weight
        self.progress_adv_weight = progress_adv_weight
        self.action_delta_weight = action_delta_weight
        self.action_scale = action_scale
        self.cached: list[np.ndarray] = []
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.phase = 0
        self.step = 0

    def reset(self) -> None:
        self.cached = []
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.phase = 0
        self.step = 0

    @torch.no_grad()
    def _plan(self, obs, env):
        raw = flatten_obs(obs)
        if self.phase_mode == "schedule":
            progress = float(self.step) / float(max(1, self.max_steps))
            self.phase = int(np.clip(np.floor(progress * int(self.ckpt["n_phases"])), 0, int(self.ckpt["n_phases"]) - 1))
        else:
            self.phase = self.future_index.estimate_phase(raw, self.normalizer, self.phase, self.monotone_phase)
        target_state, cur_phase, target_phase, cur_progress, target_progress = self.future_index.query(
            raw,
            self.normalizer,
            self.phase,
            self.phase_window,
        )
        self.phase = max(self.phase, cur_phase) if self.monotone_phase else cur_phase
        s_np = self.normalizer.encode(raw).astype(np.float32)
        tgt_np = self.normalizer.encode(target_state).astype(np.float32)
        s = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
        tgt = torch.from_numpy(tgt_np).unsqueeze(0).to(self.device)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(tgt)
        horizons = list(self.ckpt.get("future_horizons", [int(self.ckpt["H"])]))
        target_h = int(self.target_horizon or max(horizons))
        h_token = torch.tensor([[float(target_h) / float(max(horizons))]], dtype=z.dtype, device=self.device)
        phase_feat = phase_features(
            int(cur_phase),
            int(target_phase),
            cur_progress,
            target_progress,
            int(self.ckpt["n_phases"]),
            self.device,
        ).view(1, -1)
        parts = [z, z_goal, h_token, phase_feat]
        if bool(self.ckpt.get("concat_raw", False)):
            parts.extend([s, tgt])
        cond = torch.cat(parts, dim=-1)
        base = self.prior(cond).view(1, int(self.ckpt["H"]), int(self.ckpt["action_dim"])) * self.action_scale
        if self.candidates > 1 and self.noise_std > 0:
            actions = base.repeat(self.candidates, 1, 1) + torch.randn(self.candidates, *base.shape[1:], device=self.device) * self.noise_std
            actions[0] = base[0]
        else:
            actions = base.repeat(self.candidates, 1, 1)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        actions = actions.clamp(low, high)
        if actions.shape[0] == 1 and self.state_weight <= 0 and self.progress_head is None and self.action_delta_weight <= 0:
            return actions[0].detach().cpu().numpy().astype(np.float32)
        traj_z = self.wm.predict_rollout(z.repeat(actions.shape[0], 1), actions, int(self.ckpt["H"]))
        scores = self.latent_weight * torch.sum(
            (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        if self.state_weight > 0:
            pred = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            target_t = torch.as_tensor(target_state, dtype=pred.dtype, device=self.device).view(1, 1, -1)
            dist = torch.linalg.norm(pred - target_t, dim=-1)
            scores = scores + self.state_weight * (dist[:, -1] + 0.25 * dist.mean(dim=1))
        if self.progress_head is not None and (self.progress_weight > 0 or self.progress_adv_weight > 0):
            pred_progress = torch.sigmoid(self.progress_head(traj_z[:, -1])).squeeze(-1)
            target_p = torch.full_like(pred_progress, float(target_progress))
            scores = scores + self.progress_weight * (pred_progress - target_p).square()
            scores = scores - self.progress_adv_weight * pred_progress
        if self.action_delta_weight > 0:
            prev = torch.as_tensor(self.prev_action, dtype=torch.float32, device=self.device).view(1, 1, -1)
            delta = torch.cat([actions[:, :1] - prev, actions[:, 1:] - actions[:, :-1]], dim=1)
            scores = scores + self.action_delta_weight * delta.square().mean(dim=(1, 2))
        best = int(torch.argmin(scores).detach().cpu())
        return actions[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs, env):
        if not self.cached:
            plan = self._plan(obs, env)
            k = max(1, min(self.exec_k, len(plan)))
            self.cached = [plan[i].copy() for i in range(k)]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
        self.prev_action = action
        self.step += 1
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--inverse-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--phase-window", type=int, default=0)
    p.add_argument("--phase-mode", choices=["nearest", "schedule"], default="nearest")
    p.add_argument("--no-monotone-phase", action="store_true")
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.0)
    p.add_argument("--progress-head", type=Path, default=None)
    p.add_argument("--progress-weight", type=float, default=0.0,
                   help="Score candidates by squared predicted-progress error to the demo future progress.")
    p.add_argument("--progress-adv-weight", type=float, default=0.0,
                   help="Reward candidates whose JEPA rollout predicts higher demo progress.")
    p.add_argument("--action-delta-weight", type=float, default=0.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    ckpt = torch.load(args.inverse_path, map_location=dev, weights_only=False)
    progress_head = load_progress_head(args.progress_head, dev)
    prior = InversePrior(int(ckpt["cond_dim"]), int(ckpt["chunk_dim"]), int(ckpt["hidden"]), int(ckpt["n_blocks"])).to(dev)
    prior.load_state_dict(ckpt["state_dict"])
    prior.eval()
    future_index = PhaseFutureIndex(ckpt, norm)
    env = make_env(
        task.env_id,
        seed=args.seed,
        max_episode_steps=task.max_episode_steps,
        render_mode="rgb_array" if args.video_out is not None else None,
        width=args.width if args.video_out is not None else None,
        height=args.height if args.video_out is not None else None,
    )
    policy = PhaseInversePolicy(
        wm=wm,
        normalizer=norm,
        spec=spec,
        prior=prior,
        progress_head=progress_head,
        ckpt=ckpt,
        future_index=future_index,
        device=dev,
        candidates=args.candidates,
        noise_std=args.noise_std,
        exec_k=args.exec_k,
        phase_window=args.phase_window,
        phase_mode=args.phase_mode,
        monotone_phase=not args.no_monotone_phase,
        max_steps=task.max_episode_steps,
        target_horizon=args.target_horizon,
        latent_weight=args.latent_weight,
        state_weight=args.state_weight,
        progress_weight=args.progress_weight,
        progress_adv_weight=args.progress_adv_weight,
        action_delta_weight=args.action_delta_weight,
        action_scale=args.action_scale,
    )
    metrics = rollout_policy(
        env,
        policy,
        episodes=args.episodes,
        seed=args.seed,
        video_path=args.video_out,
        fps=args.fps,
    )
    env.close()
    row = {
        "event": "phase_inverse_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "inverse_path": str(args.inverse_path),
        "model_config": cfg,
        **metrics,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
