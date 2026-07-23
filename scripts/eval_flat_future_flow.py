"""Evaluate flat-task flow chunks with demo-future retrieval and JEPA selection."""
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
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.futures import NearestFutureIndex
from jepa_robotics.algos.priors import append_emphasis, sample_chunk
from jepa_robotics.algos.priors import EpsNet, make_ddpm


class FlatFlowPolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        flow,
        ckpt,
        future_index,
        device,
        candidates: int,
        exec_k: int,
        flow_steps: int,
        target_horizon: int | None,
        latent_weight: float,
        state_weight: float,
        action_l2_weight: float,
        action_delta_weight: float,
        action_scale: float,
    ) -> None:
        self.name = f"flat_flow_jepa_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.flow = flow
        self.ckpt = ckpt
        self.future_index = future_index
        self.device = device
        self.candidates = candidates
        self.exec_k = exec_k
        self.flow_steps = flow_steps
        self.target_horizon = target_horizon
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.action_l2_weight = action_l2_weight
        self.action_delta_weight = action_delta_weight
        self.action_scale = action_scale
        self.ddpm = make_ddpm(int(ckpt["diffusion_steps"]), device)
        self.cached: list[np.ndarray] = []
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.cached = []
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)

    @torch.no_grad()
    def _plan(self, obs, env):
        raw = flatten_obs(obs)
        target_state = self.future_index.query(raw, self.normalizer)
        s_np = self.normalizer.encode(raw).astype(np.float32)
        tgt_np = self.normalizer.encode(target_state).astype(np.float32)
        s = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
        tgt = torch.from_numpy(tgt_np).unsqueeze(0).to(self.device)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(tgt)
        horizons = list(self.ckpt.get("future_horizons", [int(self.ckpt["H"])]))
        target_h = int(self.target_horizon or max(horizons))
        h_token = torch.tensor([[float(target_h) / float(max(horizons))]], dtype=z.dtype, device=self.device)
        parts = [z, z_goal, h_token]
        if bool(self.ckpt.get("concat_raw", False)):
            parts.extend([s, tgt])
        append_emphasis(parts, s, self.ckpt)
        cond = torch.cat(parts, dim=-1).repeat(self.candidates, 1)
        chunks = sample_chunk(
            self.flow,
            self.ddpm,
            cond,
            int(self.ckpt["chunk_dim"]),
            self.device,
            objective="flow",
            flow_steps=self.flow_steps,
        )
        H = int(self.ckpt["H"])
        A = int(self.ckpt["action_dim"])
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        actions = (chunks.view(self.candidates, H, A) * self.action_scale).clamp(low, high)
        if actions.shape[0] == 1 and self.state_weight <= 0 and self.action_l2_weight <= 0 and self.action_delta_weight <= 0:
            return actions[0].detach().cpu().numpy().astype(np.float32)
        traj_z = self.wm.predict_rollout(z.repeat(actions.shape[0], 1), actions, H)
        scores = self.latent_weight * torch.sum(
            (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        if self.state_weight > 0:
            pred = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            target_t = torch.as_tensor(target_state, dtype=pred.dtype, device=self.device).view(1, 1, -1)
            dist = torch.linalg.norm(pred - target_t, dim=-1)
            scores = scores + self.state_weight * (dist[:, -1] + 0.25 * dist.mean(dim=1))
        if self.action_l2_weight > 0:
            scores = scores + self.action_l2_weight * actions.square().mean(dim=(1, 2))
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
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--flow-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.0)
    p.add_argument("--action-l2-weight", type=float, default=0.0)
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
    ckpt = torch.load(args.flow_path, map_location=dev, weights_only=False)
    flow = EpsNet(
        int(ckpt["chunk_dim"]),
        int(ckpt["cond_dim"]),
        int(ckpt["hidden"]),
        n_blocks=int(ckpt["n_blocks"]),
    ).to(dev)
    flow.load_state_dict(ckpt["ema"])
    flow.eval()
    future_index = NearestFutureIndex(np.asarray(ckpt["bank_states"]), np.asarray(ckpt["bank_futures"]), norm)
    env = make_env(
        task.env_id,
        seed=args.seed,
        max_episode_steps=task.max_episode_steps,
        render_mode="rgb_array" if args.video_out is not None else None,
        width=args.width if args.video_out is not None else None,
        height=args.height if args.video_out is not None else None,
    )
    policy = FlatFlowPolicy(
        wm=wm,
        normalizer=norm,
        spec=spec,
        flow=flow,
        ckpt=ckpt,
        future_index=future_index,
        device=dev,
        candidates=args.candidates,
        exec_k=args.exec_k,
        flow_steps=args.flow_steps,
        target_horizon=args.target_horizon,
        latent_weight=args.latent_weight,
        state_weight=args.state_weight,
        action_l2_weight=args.action_l2_weight,
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
        "event": "flat_flow_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "flow_path": str(args.flow_path),
        "model_config": cfg,
        **metrics,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
