"""Goal-conditioned latent MPC for the Shadow Hand HandManipulate suite.

Pure SSL control: a DexterousJEPA world model (trained self-supervised on
exploration, no demos, no RL) plus the env-provided target object pose. At each
step, CEM over action sequences: roll candidates through the world model, decode
the predicted object pose (achieved_goal slice) via the state probe, and score
distance to the desired object pose. Execute the first action(s), replan (MPC).

HandManipulate is an RL-hard benchmark (Dactyl used massive-scale RL); this is
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
from jepa_robotics.tasks import resolve_task


class LatentMPC:
    def __init__(self, wm, norm, spec, dev, horizon, candidates, iters, elite_frac,
                 init_std, exec_k, disagree_weight):
        self.wm, self.norm, self.spec, self.dev = wm, norm, spec, dev
        self.H, self.N, self.iters = horizon, candidates, iters
        self.elite = max(1, int(candidates * elite_frac))
        self.init_std, self.exec_k, self.disagree_w = init_std, exec_k, disagree_weight
        self.ag_lo = spec.obs_dim                      # achieved_goal slice in flat state
        self.ag_hi = spec.obs_dim + spec.goal_dim
        self.dg_lo = spec.obs_dim + spec.goal_dim      # desired_goal slice
        self.dg_hi = spec.obs_dim + 2 * spec.goal_dim
        self._buf = []

    def reset(self):
        self._buf = []

    @torch.no_grad()
    def _plan(self, raw, env):
        A = self.spec.action_dim
        sn = torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev)
        z = self.wm.encode(sn)
        # Score in the environment's physical goal coordinates.  In particular,
        # quaternion components cannot be compared in z-scored state space: a
        # sign-equivalent quaternion may have a large component-wise error while
        # representing the same orientation.  The HandManipulate reward is
        # -(10 * position_distance + quaternion_angle), so use that exact dense
        # surrogate for planning as well.
        raw_dg = torch.as_tensor(raw[self.dg_lo:self.dg_hi], dtype=torch.float32, device=self.dev)
        lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.dev)
        hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.dev)
        mean = torch.zeros(self.H, A, device=self.dev)
        std = torch.full((self.H, A), self.init_std, device=self.dev)
        best_first = None
        for _ in range(self.iters):
            eps = torch.randn(self.N, self.H, A, device=self.dev)
            acts = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(lo, hi)
            roll = self.wm.predict_rollout(z.expand(self.N, -1), acts, self.H)  # [N,H,latent]
            pred = self.norm.decode_tensor(self.wm.state_probe(roll))           # raw state
            pred_ag = pred[:, :, self.ag_lo:self.ag_hi]                         # [N,H,7]
            pos_dist = torch.linalg.vector_norm(pred_ag[:, :, :3] - raw_dg[:3], dim=-1)
            qa = pred_ag[:, :, 3:]
            qb = raw_dg[3:].view(1, 1, 4)
            # The simulator computes 2*acos(q_a * conjugate(q_b)).  Normalizing
            # the predicted quaternion keeps a briefly imperfect state probe
            # from manufacturing an invalid angle.
            qa = qa / torch.linalg.vector_norm(qa, dim=-1, keepdim=True).clamp_min(1e-6)
            qb = qb / torch.linalg.vector_norm(qb, dim=-1, keepdim=True).clamp_min(1e-6)
            dot = (qa * qb).sum(-1).clamp(-1.0, 1.0)
            rot_dist = 2.0 * torch.acos(dot)
            dist = 10.0 * pos_dist + rot_dist
            cost = dist[:, -1] + 0.25 * dist.mean(1)                           # terminal + path
            if self.disagree_w > 0:
                # ``disagreement`` is a batch-mean diagnostic in the shared
                # model interface.  CEM instead needs one uncertainty value
                # per candidate; otherwise the same scalar is added to every
                # candidate and has no planning effect.
                if getattr(self.wm, "ensemble_heads", 1) > 1:
                    head_rolls = self.wm.rollout_heads(z.expand(self.N, -1), acts, self.H)
                    uncertainty = head_rolls.var(dim=0).mean(dim=(1, 2))
                    cost = cost + self.disagree_w * uncertainty
            order = torch.argsort(cost)
            elite = acts[order[: self.elite]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.02)
            best_first = acts[order[0]]
        return best_first.cpu().numpy()

    def act(self, obs, env):
        if not self._buf:
            plan = self._plan(flatten_obs(obs), env)
            self._buf = [plan[i].copy() for i in range(max(1, min(self.exec_k, len(plan))))]
        return np.clip(self._buf.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)


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
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--log-path", type=Path, default=None,
                   help="Optional JSONL path; writes one completion row per episode and a final summary.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    mpc = LatentMPC(wm, norm, spec, dev, args.horizon, args.candidates, args.iters,
                    args.elite_frac, args.init_std, args.exec_k, args.disagree_weight)

    successes = []
    log_file = None
    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_path.open("w")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        mpc.reset()
        term = trunc = False; info = {}
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(mpc.act(obs, env))
        success = float(info.get("is_success", 0.0))
        successes.append(success)
        episode_row = {"event": "handmanipulate_mpc_episode", "task": task.name,
                       "episode": ep, "success": success,
                       "running_success_rate": float(np.mean(successes))}
        print(json.dumps(episode_row), flush=True)
        if log_file is not None:
            log_file.write(json.dumps(episode_row) + "\n")
            log_file.flush()
    env.close()
    row = {"event": "handmanipulate_mpc_eval", "task": task.name, "model_path": str(args.model_path),
           "episodes": args.episodes, "success_rate": float(np.mean(successes)),
           "horizon": args.horizon, "candidates": args.candidates, "iters": args.iters,
           "exec_k": args.exec_k, "disagree_weight": args.disagree_weight, "torch_seed": args.torch_seed}
    print(json.dumps(row, default=str), flush=True)
    if log_file is not None:
        log_file.write(json.dumps(row, default=str) + "\n")
        log_file.close()


if __name__ == "__main__":
    main()
