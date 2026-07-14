"""Progress scorer for best-of-N flow-walker chunk selection (AntMaze).

The demos are bimodal: ~60% near-stationary, but the moving mode reaches
~1.1-1.5 xy units per 8-step chunk (p90-p99). The flow walker samples one
mode at random, so its realized speed averages over slow and fast — the
documented "imitation-speed ceiling" is an EXTRACTION ceiling, not a data
ceiling. Fix = the repo's Fetch signature move at the gait level: sample N
on-manifold chunks from the (unchanged) flow walker and execute the one a
chunk-outcome predictor S(cond, chunk) -> progress says closes the most xy
distance to the live subgoal.

Trains S on the same directed near-relabel tuples as the walker but WITHOUT
the min-progress filter (the scorer must rank slow/backward chunks low, so it
needs to see them). Conditioning is built exactly like the walker checkpoint's
(JEPA latent [+ raw obs] [+ goal-delta emphasis]) so eval reuses one cond
tensor for both nets; cond_dim is asserted against the walker at load time.
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
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from scripts.train_flow_walker import build_directed_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--walker", type=Path, required=True,
                   help="Flow-walker checkpoint whose conditioning layout (concat_raw, emphasis) the scorer must match.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-relabel-h", type=int, default=40,
                   help="Match the walker's directed relabel horizon.")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, _ = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()

    wcfg = torch.load(args.walker, map_location="cpu", weights_only=False)["config"]
    chunk = int(wcfg["chunk"])
    emphasis = int(wcfg.get("emphasis_repeat", 0) or 0)
    a_lo, a_hi = (int(x) for x in wcfg.get("agent_dims", [27, 29]))
    g_lo, g_hi = (int(x) for x in wcfg.get("goal_dims", [29, 31]))

    rng = np.random.default_rng(0)
    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    # No progress filter: the scorer must learn the full range, including
    # stationary/backward chunks, to rank flow samples.
    S, C, prog = build_directed_chunks(eps, spec, norm, chunk, rng,
                                       max_relabel_h=args.max_relabel_h, min_progress=-1e9)
    St = torch.from_numpy(S).to(dev)
    with torch.no_grad():
        Z = torch.cat([wm.encode(St[i:i + 16384]) for i in range(0, len(St), 16384)], 0)
    cond = torch.cat([Z, St], dim=1) if bool(wcfg["concat_raw"]) else Z
    if emphasis > 0:
        delta = (St[:, g_lo:g_hi] - St[:, a_lo:a_hi]).repeat(1, emphasis)
        cond = torch.cat([cond, delta], dim=1)
    assert cond.shape[1] == int(wcfg["cond_dim"]), (cond.shape[1], wcfg["cond_dim"])
    Ct = torch.from_numpy(C.reshape(len(C), -1)).to(dev)
    Pt = torch.from_numpy(prog).to(dev).unsqueeze(1)
    N = len(cond)
    n_hold = max(1, int(N * args.holdout_frac))
    perm = torch.randperm(N, device=dev)
    hold, tr = perm[:n_hold], perm[n_hold:]
    print(json.dumps({"event": "scorer_data", "n": N, "cond_dim": int(cond.shape[1]),
                      "chunk_dim": int(Ct.shape[1]),
                      "progress_p50": round(float(np.percentile(prog, 50)), 3),
                      "progress_p90": round(float(np.percentile(prog, 90)), 3)}), flush=True)

    scorer = ProgressScorer(int(cond.shape[1]), int(Ct.shape[1]), args.hidden)
    net = scorer.net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        idx = tr[torch.randint(0, len(tr), (args.batch_size,), device=dev)]
        pred = net(torch.cat([cond[idx], Ct[idx]], dim=-1))
        loss = nn.functional.mse_loss(pred, Pt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10000 == 0:
            with torch.no_grad():
                hp = torch.cat([net(torch.cat([cond[hold[i:i + 16384]], Ct[hold[i:i + 16384]]], dim=-1))
                                for i in range(0, n_hold, 16384)], 0)
                mae = float((hp - Pt[hold]).abs().mean())
            print(json.dumps({"event": "scorer_train", "step": step,
                              "loss": round(float(loss), 4), "holdout_mae": round(mae, 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scorer": net.state_dict(),
                "config": {"cond_dim": int(cond.shape[1]), "chunk_dim": int(Ct.shape[1]),
                           "hidden": args.hidden, "chunk": chunk,
                           "max_relabel_h": args.max_relabel_h}},
               args.out)
    print(json.dumps({"event": "scorer_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
