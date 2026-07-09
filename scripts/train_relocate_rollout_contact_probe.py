"""Train a Relocate contact probe on JEPA-predicted rollout latents.

The plain contact probe fits true encoder latents, but candidate selection scores
``wm.predict_rollout`` latents. This trainer closes that distribution gap:

    z_rollout[i] = JEPA(z_t, a_{t:t+i})
    target[i] = [||palm-ball||, ||ball-target||] from true state s_{t+i}

The probe remains self-supervised from state geometry. It uses no action-label
policy at runtime; actions are only transition evidence for calibrating the
latent-space scorer.
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
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-starts", type=int, default=100000)
    p.add_argument("--encode-batch", type=int, default=2048)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=307)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    wm, norm, _spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    starts, chunks, future_states = [], [], []
    H = args.chunk
    for ep in load_episodes_npz(args.episodes_npz)[: args.max_episodes]:
        s = ep.states.astype(np.float32)
        a = ep.actions.astype(np.float32)
        if len(a) < H or s.shape[-1] < 39:
            continue
        for t in range(len(a) - H + 1):
            starts.append(s[t])
            chunks.append(a[t : t + H])
            future_states.append(s[t + 1 : t + H + 1])
    if not starts:
        raise RuntimeError("no valid rollout-probe starts")
    if args.max_starts > 0 and len(starts) > args.max_starts:
        idx = rng.choice(len(starts), size=args.max_starts, replace=False)
        starts = [starts[i] for i in idx]
        chunks = [chunks[i] for i in idx]
        future_states = [future_states[i] for i in idx]

    S0 = np.asarray(starts, dtype=np.float32)
    A = np.asarray(chunks, dtype=np.float32)
    SF = np.asarray(future_states, dtype=np.float32)
    palm_ball = np.linalg.norm(SF[..., 30:33], axis=-1)
    ball_target = np.linalg.norm(SF[..., 36:39], axis=-1)
    Y = np.stack([palm_ball, ball_target], axis=-1).reshape(-1, 2).astype(np.float32)

    zs = []
    with torch.no_grad():
        for i in range(0, len(S0), args.encode_batch):
            s0 = torch.from_numpy(norm.encode(S0[i : i + args.encode_batch])).to(dev)
            act = torch.from_numpy(A[i : i + args.encode_batch]).to(dev)
            z0 = wm.encode(s0)
            rollout = wm.predict_rollout(z0, act, H)
            zs.append(rollout.reshape(-1, int(cfg["latent_dim"])).detach())
    Z = torch.cat(zs, dim=0)
    Yt = torch.from_numpy(Y).to(dev)
    if len(Z) != len(Yt):
        raise RuntimeError(f"latent/label mismatch: {len(Z)} vs {len(Yt)}")

    head = MLP([int(cfg["latent_dim"]), args.hidden, args.hidden, 2], layer_norm=True).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "event": "relocate_rollout_contact_probe_data",
                "starts": int(len(S0)),
                "pairs": int(len(Z)),
                "chunk": int(H),
                "latent_dim": int(cfg["latent_dim"]),
                "mean_palm_ball": float(Y[:, 0].mean()),
                "mean_ball_target": float(Y[:, 1].mean()),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, len(Z), (args.batch_size,), device=dev)
        pred = torch.clamp(head(Z[idx]), min=0.0)
        loss = nn.functional.smooth_l1_loss(pred, Yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "relocate_rollout_contact_probe_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
            "targets": "palm_ball_distance,ball_target_distance",
            "trained_on": "jepa_predicted_rollout_latents",
            "chunk": int(H),
        },
        args.out,
    )
    print(json.dumps({"event": "relocate_rollout_contact_probe_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
