"""Train the learned high level: ``z_t -> (z_{t+h}, state_{t+h}, progress)``.

Replaces the hand-supplied structure the phase-inverse controllers rely on (a
wall-clock phase schedule, a stored demo bank, a nearest-neighbour lookup and a
monotonicity rule) with one network trained on the same demonstrations. See
``jepa_robotics.algos.latent_subgoal``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.latent_subgoal import (
    LatentSubgoalNet,
    build_subgoal_windows,
    subgoal_loss,
)
from jepa_robotics.algos.priors import parse_horizons
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact


@torch.no_grad()
def encode_all(wm, arr: np.ndarray, device, target: bool, batch: int = 65536) -> torch.Tensor:
    out = []
    for i in range(0, len(arr), batch):
        x = torch.from_numpy(arr[i : i + batch]).to(device)
        out.append(wm.encode_target(x) if target else wm.encode(x))
    return torch.cat(out, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizons", type=parse_horizons, default="4,8,12,16")
    p.add_argument("--max-episodes", type=int, default=4000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--state-weight", type=float, default=1.0)
    p.add_argument("--progress-weight", type=float, default=0.1)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    wm, normalizer, spec, cfg = load_jepa_artifact(args.model_path, device)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    states, horizons, futures, progress = build_subgoal_windows(
        episodes, normalizer, list(args.horizons)
    )
    print(json.dumps({"event": "windows", "rows": int(len(states)),
                      "horizons": list(args.horizons)}), flush=True)

    z_in = encode_all(wm, states, device, target=False)
    z_out = encode_all(wm, futures, device, target=True)
    s_out = torch.from_numpy(futures).to(device)
    h_col = torch.from_numpy(horizons).to(device)
    p_col = torch.from_numpy(progress).to(device)

    perm = torch.from_numpy(rng.permutation(len(z_in))).to(device)
    n_hold = max(1, int(len(z_in) * args.holdout_frac))
    hold, train = perm[:n_hold], perm[n_hold:]

    net = LatentSubgoalNet(
        latent_dim=int(cfg["latent_dim"]),
        state_dim=int(spec.state_dim),
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        max_horizon=max(args.horizons),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.train_steps, eta_min=args.lr * 0.05)

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": net.state_dict(),
            "latent_dim": int(cfg["latent_dim"]),
            "state_dim": int(spec.state_dim),
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "max_horizon": max(args.horizons),
            "horizons": list(args.horizons),
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        }, args.out)

    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        idx = train[torch.randint(0, train.numel(), (args.batch_size,), device=device)]
        loss, metrics = subgoal_loss(
            net, z_in[idx], h_col[idx], z_out[idx], s_out[idx], p_col[idx],
            state_weight=args.state_weight, progress_weight=args.progress_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            with torch.no_grad():
                hidx = hold[torch.randint(0, hold.numel(), (min(8192, hold.numel()),), device=device)]
                _, hm = subgoal_loss(
                    net, z_in[hidx], h_col[hidx], z_out[hidx], s_out[hidx], p_col[hidx],
                    state_weight=args.state_weight, progress_weight=args.progress_weight,
                )
            print(json.dumps({"event": "train", "step": step, **metrics,
                              "holdout_latent": hm["latent"], "holdout_state": hm["state"],
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)
            save()
    save()
    print(json.dumps({"event": "saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
