"""Flip-risk chunk scorer for the AntMaze flow walker, trained on SELF-TRIALS.

Companion to train_recovery_walker.py (see its docstring for the diagnosis):
half of all controller steps end up flipped, and flips — not gait speed — cap
Medium/Large success. Avoidance is cheaper than recovery, and unlike speed
selection it has no winner's curse: we only need to VETO the flip tail, not
chase the fast tail.

Trains S(cond, chunk) -> min torso uprightness over the chunk on self-trial
rollouts (eval_hjepa_hwm.py --rollout-out), which are full of both safe chunks
and transitions into flips. cond mirrors the walker checkpoint's layout
(JEPA latent [+ raw obs] [+ goal-delta emphasis]) with the desired goal
relabeled to a nearby future achieved xy, mimicking live subgoal conditioning.
At eval (LowLevelFlow risk filter) the walker samples N gait chunks, discards
those with predicted min-uprightness below a threshold, and picks randomly
among the safe ones.
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

from jepa_robotics.algos.maze_low_level import ProgressScorer
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.train_recovery_walker import uprightness


def build_risk_chunks(episodes, chunk, rng, *, gs=27, ge=29, ds=29, de=31, max_relabel_h=40):
    S_list, C_list, U_list = [], [], []
    for ep in episodes:
        S, A = ep.states, ep.actions
        u = uprightness(S)
        T = len(A)
        for t in range(T - chunk):
            dh = int(rng.integers(1, max_relabel_h + 1))
            tf = min(t + dh, len(S) - 1)
            s = S[t].copy()
            s[ds:de] = S[tf, gs:ge]
            S_list.append(s)
            C_list.append(A[t: t + chunk])
            U_list.append(u[t + 1: t + chunk + 1].min())
    return (np.asarray(S_list, np.float32), np.asarray(C_list, np.float32),
            np.asarray(U_list, np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, nargs="+", required=True)
    p.add_argument("--walker", type=Path, required=True,
                   help="Flow-walker checkpoint whose conditioning layout the scorer must match.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, norm, _, _ = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)

    wcfg = torch.load(args.walker, map_location="cpu", weights_only=False)["config"]
    chunk = int(wcfg["chunk"])
    emphasis = int(wcfg.get("emphasis_repeat", 0) or 0)
    a_lo, a_hi = (int(x) for x in wcfg.get("agent_dims", [27, 29]))
    g_lo, g_hi = (int(x) for x in wcfg.get("goal_dims", [29, 31]))

    rng = np.random.default_rng(0)
    eps = []
    for path in args.episodes_npz:
        eps.extend(load_episodes_npz(path))
    S, C, U = build_risk_chunks(eps, chunk, rng, gs=a_lo, ge=a_hi, ds=g_lo, de=g_hi)
    St = torch.from_numpy(norm.encode(S)).to(dev)
    with torch.no_grad():
        Z = torch.cat([wm.encode(St[i:i + 16384]) for i in range(0, len(St), 16384)], 0)
    cond = torch.cat([Z, St], dim=1) if bool(wcfg["concat_raw"]) else Z
    if emphasis > 0:
        delta = (St[:, g_lo:g_hi] - St[:, a_lo:a_hi]).repeat(1, emphasis)
        cond = torch.cat([cond, delta], dim=1)
    assert cond.shape[1] == int(wcfg["cond_dim"]), (cond.shape[1], wcfg["cond_dim"])
    Ct = torch.from_numpy(C.reshape(len(C), -1)).to(dev)
    Ut = torch.from_numpy(U).to(dev).unsqueeze(1)
    N = len(cond)
    n_hold = max(1, int(N * args.holdout_frac))
    perm = torch.randperm(N, device=dev)
    hold, tr = perm[:n_hold], perm[n_hold:]
    print(json.dumps({"event": "risk_data", "n": N,
                      "frac_flip_outcome": round(float((U < 0.0).mean()), 3),
                      "frac_safe_outcome": round(float((U > 0.7).mean()), 3)}), flush=True)

    scorer = ProgressScorer(int(cond.shape[1]), int(Ct.shape[1]), args.hidden)
    net = scorer.net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        idx = tr[torch.randint(0, len(tr), (args.batch_size,), device=dev)]
        loss = nn.functional.mse_loss(net(torch.cat([cond[idx], Ct[idx]], dim=-1)), Ut[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10000 == 0:
            with torch.no_grad():
                hp = torch.cat([net(torch.cat([cond[hold[i:i + 16384]], Ct[hold[i:i + 16384]]], dim=-1))
                                for i in range(0, n_hold, 16384)], 0)
                mae = float((hp - Ut[hold]).abs().mean())
            print(json.dumps({"event": "risk_train", "step": step,
                              "loss": round(float(loss.detach()), 4), "holdout_mae": round(mae, 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scorer": net.state_dict(),
                "config": {"cond_dim": int(cond.shape[1]), "chunk_dim": int(Ct.shape[1]),
                           "hidden": args.hidden, "chunk": chunk, "target": "min_uprightness"}},
               args.out)
    print(json.dumps({"event": "risk_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
