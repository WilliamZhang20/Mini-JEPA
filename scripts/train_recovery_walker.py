"""Flip-recovery flow prior for the AntMaze walker, trained on SELF-TRIALS.

Eval diagnostics showed the canonical directed flow walker spends ~38% of
steps flipped on its back (torso body-z . world-z < 0) and every failure is a
1000-step timeout — the demos contain no flipped states, so once flipped the
walker's conditioning is off-manifold and it flails until the clock runs out.
Every speed lever (best-of-N argmax, quantile selection, progress conditioning)
made success WORSE because a faster gait flips more.

The missing skill is recovery, and demos cannot teach it. Self-trials can:
rollouts of the live controller (eval_hjepa_hwm.py --rollout-out) are full of
flips and occasional natural recoveries. This script mines chunks that start
tipped/flipped (uprightness < --enter) and raise uprightness by >= --min-gain,
and fits a rectified-flow prior over those action chunks conditioned on
(JEPA latent | raw normalized obs). The walker (LowLevelFlow) switches to this
prior whenever live uprightness drops below the enter threshold and back to the
gait flow once recovered — trials teaching which action chunks cause which
futures, applied to uprightness.
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

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.train_flow_walker import FlowNet


def uprightness(states: np.ndarray) -> np.ndarray:
    """Torso body-z projected on world-z from the quat at obs dims 1:5 (w,x,y,z)."""
    qx, qy = states[:, 2], states[:, 3]
    return 1.0 - 2.0 * (qx ** 2 + qy ** 2)


def build_recovery_chunks(episodes, chunk, *, enter=0.3, min_gain=0.15):
    S_list, C_list = [], []
    n_flipped = n_recov = 0
    for ep in episodes:
        S, A = ep.states, ep.actions
        u = uprightness(S)
        T = len(A)
        for t in range(T - chunk):
            if u[t] >= enter:
                continue
            n_flipped += 1
            gain = float(u[t + chunk] - u[t])
            if gain < min_gain:
                continue
            n_recov += 1
            S_list.append(S[t])
            C_list.append(A[t: t + chunk])
    stats = {"flipped_starts": n_flipped, "recovery_chunks": n_recov}
    if not C_list:
        return None, None, stats
    return np.asarray(S_list, np.float32), np.asarray(C_list, np.float32), stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, nargs="+", required=True,
                   help="Self-trial rollout npz files (eval_hjepa_hwm.py --rollout-out).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--enter", type=float, default=0.3,
                   help="Uprightness below which a state counts as tipped/flipped.")
    p.add_argument("--min-gain", type=float, default=0.15,
                   help="Keep a chunk only if uprightness rises at least this much across it.")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)

    eps = []
    for path in args.episodes_npz:
        eps.extend(load_episodes_npz(path))
    S, C, stats = build_recovery_chunks(eps, args.chunk, enter=args.enter, min_gain=args.min_gain)
    print(json.dumps({"event": "recovery_data", **stats}), flush=True)
    if S is None or len(S) < 256:
        raise SystemExit(f"Too few recovery chunks ({stats['recovery_chunks']}); collect targeted recovery trials first.")

    St = torch.from_numpy(norm.encode(S)).to(dev)
    with torch.no_grad():
        Z = torch.cat([wm.encode(St[i:i + 16384]) for i in range(0, len(St), 16384)], 0)
    cond = torch.cat([Z, St], dim=1)
    Ct = torch.from_numpy(C.reshape(len(C), -1)).to(dev)
    chunk_dim, cond_dim, N = Ct.shape[1], cond.shape[1], len(cond)
    print(json.dumps({"event": "recovery_train_data", "n": N, "chunk_dim": int(chunk_dim),
                      "cond_dim": int(cond_dim)}), flush=True)

    net = FlowNet(chunk_dim, cond_dim, args.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        x1 = Ct[idx]; c = cond[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=dev)
        xt = (1 - t) * x0 + t * x1
        loss = nn.functional.mse_loss(net(xt, t, c), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10000 == 0:
            print(json.dumps({"event": "recovery_train", "step": step, "loss": round(float(loss), 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"flow": net.state_dict(),
                "config": {"chunk": args.chunk, "hidden": args.hidden,
                           "chunk_dim": int(chunk_dim), "cond_dim": int(cond_dim),
                           "latent_dim": int(cfg["latent_dim"]),
                           "enter": args.enter, "min_gain": args.min_gain}},
               args.out)
    print(json.dumps({"event": "recovery_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
