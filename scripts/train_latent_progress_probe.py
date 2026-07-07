"""Train a self-supervised latent progress probe from offline demonstrations.

For flat non-goal tasks, normalized time within an expert trajectory is a useful
weak target for high-level progress. The probe is not a reward label and is not
an action policy; it lets JEPA rollout scoring ask whether a candidate action
chunk moves the imagined latent toward later demo phases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=173)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    wm, norm, _spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    states, progress = [], []
    for ep in load_episodes_npz(args.episodes_npz)[: args.max_episodes]:
        s = ep.states[:-1].astype(np.float32)
        n = len(s)
        if n == 0:
            continue
        states.append(s)
        progress.append(np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)[:, None])
    S = np.concatenate(states, axis=0)
    Y = np.concatenate(progress, axis=0)
    with torch.no_grad():
        zs = []
        for i in range(0, len(S), 16384):
            x = torch.from_numpy(norm.encode(S[i : i + 16384])).to(dev)
            zs.append(wm.encode(x).detach())
        Z = torch.cat(zs, dim=0)
    Yt = torch.from_numpy(Y).to(dev)
    head = MLP([int(cfg["latent_dim"]), args.hidden, args.hidden, 1], layer_norm=True).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    print(json.dumps({"event": "progress_probe_data", "pairs": int(len(Z)), "latent_dim": int(cfg["latent_dim"])}), flush=True)
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, len(Z), (args.batch_size,), device=dev)
        pred = torch.sigmoid(head(Z[idx]))
        loss = nn.functional.mse_loss(pred, Yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "progress_probe_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        },
        args.out,
    )
    print(json.dumps({"event": "progress_probe_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
