"""Train a latent reward head for model-based control on Adroit.

P1 (use the JEPA *predictor* for control, not just the encoder): to score imagined
rollouts we need a reward signal in latent space. We fit a small MLP
``latent -> reward`` on the offline demo transitions (encoded through the frozen
JEPA encoder), using the dense D4RL reward as the target. The MPC controller then
rolls BC-proposed action candidates through ``predict_rollout`` and scores each by
the cumulative predicted reward of this head — so the dynamics predictor finally
drives action selection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Minari dataset id, e.g. D4RL/pen/expert-v2")
    p.add_argument("--model-path", type=Path, required=True, help="frozen JEPA WM")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import minari

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, device)
    wm.eval()
    latent_dim = int(cfg["latent_dim"])

    ds = minari.load_dataset(args.dataset)
    obs_list, rew_list = [], []
    for i, ep in enumerate(ds.iterate_episodes()):
        obs_raw = ep.observations
        if isinstance(obs_raw, dict):  # kitchen-style dict obs -> keep the flat observation
            obs_raw = obs_raw["observation"]
        o = np.asarray(obs_raw, dtype=np.float32)[:-1]  # align with rewards
        r = np.asarray(ep.rewards, dtype=np.float32)
        n = min(len(o), len(r))
        obs_list.append(o[:n]); rew_list.append(r[:n])
        if i + 1 >= args.max_episodes:
            break
    obs = np.concatenate(obs_list, 0)
    rew = np.concatenate(rew_list, 0)
    # normalize reward target (store stats for the controller)
    r_mean, r_std = float(rew.mean()), float(rew.std() + 1e-6)
    print(f'{{"event": "reward_data", "pairs": {len(obs)}, "r_mean": {r_mean:.3f}, "r_std": {r_std:.3f}}}', flush=True)

    # encode all states through the frozen JEPA encoder
    with torch.no_grad():
        zs = []
        for i in range(0, len(obs), 8192):
            x = torch.from_numpy(norm.encode(obs[i:i + 8192])).to(device)
            zs.append(wm.encode(x).cpu())
        Z = torch.cat(zs, 0)
    R = torch.from_numpy((rew - r_mean) / r_std).unsqueeze(-1)

    head = MLP([latent_dim, args.hidden, args.hidden, 1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(Z)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            z = Z[idx].to(device); y = R[idx].to(device)
            pred = head(z)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f'{{"event": "reward_train", "epoch": {ep}, "mse": {tot / n:.4f}}}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "latent_dim": latent_dim, "hidden": args.hidden,
                "r_mean": r_mean, "r_std": r_std}, args.out)
    print(f'{{"event": "reward_head_saved", "path": "{args.out}"}}', flush=True)


if __name__ == "__main__":
    main()
