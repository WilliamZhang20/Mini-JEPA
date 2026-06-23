"""Action-chunked goal-conditioned walker for AntMaze (the kitchen recipe applied
to locomotion).

The single-step reactive MLP walker falls ~half the time: per-step BC produces
torque commands that don't maintain a coherent gait, so the ant destabilises.
An action-CHUNK policy predicts the next H actions at once; executing the chunk
keeps the gait phase consistent (much less falling), and re-predicting every few
steps keeps it goal-directed (receding horizon) — the same fix that solved the
FrankaKitchen contact sequence.

Trained by behaviour cloning on the offline demos with hindsight goal relabeling
(desired_goal -> a future achieved position) so it's a general nearby-goal
reacher. Acts on the JEPA latent (which helps the small policy) by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task


class ChunkPolicy(nn.Module):
    """Maps a latent (or raw obs) to a chunk of H actions, tanh-bounded."""

    def __init__(self, in_dim, action_dim, chunk, hidden):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.net = MLP([in_dim, hidden, hidden, action_dim * chunk], layer_norm=True)

    def forward(self, z):
        return torch.tanh(self.net(z)).reshape(-1, self.chunk, self.action_dim)


def build_chunks(episodes, spec, normalizer, chunk, her_frac, rng):
    """(state_t, action chunk a_{t:t+H}) with HER goal relabeling. Pads the last
    chunk by repeating the final action."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    S_list, C_list = [], []
    for ep in episodes:
        S, A = ep.states, ep.actions
        T = len(A)
        for t in range(T):
            s = S[t].copy()
            if rng.random() < her_frac:
                tf = int(rng.integers(t + 1, len(S)))
                s[ds:de] = S[tf, gs:ge]
            chunk_a = A[t: t + chunk]
            if len(chunk_a) < chunk:  # pad by repeating last action
                pad = np.repeat(chunk_a[-1:], chunk - len(chunk_a), axis=0)
                chunk_a = np.concatenate([chunk_a, pad], axis=0)
            S_list.append(s); C_list.append(chunk_a)
    S = normalizer.encode(np.asarray(S_list, np.float32))
    return S, np.asarray(C_list, np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--her-frac", type=float, default=0.85)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()

    rng = np.random.default_rng(0)
    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    S, C = build_chunks(eps, spec, norm, args.chunk, args.her_frac, rng)
    print(json.dumps({"event": "chunk_data", "n": len(S), "chunk": args.chunk,
                      "in_dim": (S.shape[1] if args.raw else int(cfg["latent_dim"]))}), flush=True)

    St = torch.from_numpy(S).to(dev)
    Ct = torch.from_numpy(C).to(dev)
    if args.raw:
        Z = St; in_dim = S.shape[1]
    else:
        with torch.no_grad():
            Z = torch.cat([wm.encode(St[i:i + 16384]) for i in range(0, len(St), 16384)], 0)
        in_dim = Z.shape[1]

    policy = ChunkPolicy(in_dim, spec.action_dim, args.chunk, args.hidden).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    N = len(Z)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        pred = policy(Z[idx])
        loss = nn.functional.mse_loss(pred, Ct[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0:
            print(json.dumps({"event": "chunk_bc", "step": step, "loss": round(float(loss), 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(),
                "config": {"latent_dim": in_dim, "action_dim": spec.action_dim,
                           "hidden_dim": args.hidden, "chunk": args.chunk, "raw": bool(args.raw)}},
               args.out)
    print(json.dumps({"event": "chunk_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
