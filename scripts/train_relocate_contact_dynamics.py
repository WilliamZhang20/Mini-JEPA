"""Train an action-conditioned contact dynamics head for Relocate.

The JEPA latent rollout is not contact-faithful enough for Relocate candidate
ranking. This model predicts the contact-relevant state trace directly:

    f(z_t, raw_t, a_{t:t+H-1}) -> [||palm-ball||, ||ball-target||]_{t+1:t+H}

It is still self-supervised from transition trials. The output is a fine-state
world-model head used only for planning/scoring, not an action-label policy.
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


def make_targets(states: np.ndarray, t: int, H: int) -> np.ndarray:
    future = states[t + 1 : t + H + 1]
    palm_ball = np.linalg.norm(future[:, 30:33], axis=-1)
    ball_target = np.linalg.norm(future[:, 36:39], axis=-1)
    return np.stack([palm_ball, ball_target], axis=-1).reshape(-1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, nargs="+", required=True,
                   help="One or more npz episode files; pairs are pooled so the head can train on combined trial distributions.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=2000)
    p.add_argument("--max-pairs", type=int, default=200000)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=353)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    H = args.chunk
    states0, actions, targets = [], [], []
    episodes = []
    for npz_path in args.episodes_npz:
        episodes.extend(load_episodes_npz(npz_path))
    if len(episodes) > args.max_episodes:
        keep = rng.choice(len(episodes), size=args.max_episodes, replace=False)
        episodes = [episodes[int(i)] for i in keep]
    for ep in episodes:
        s = ep.states.astype(np.float32)
        a = ep.actions.astype(np.float32)
        if len(a) < H or s.shape[-1] < 39:
            continue
        for t in range(len(a) - H + 1):
            states0.append(s[t])
            actions.append(a[t : t + H].reshape(-1))
            targets.append(make_targets(s, t, H))
    if not states0:
        raise RuntimeError("no valid contact-dynamics pairs")
    S0 = np.asarray(states0, dtype=np.float32)
    A = np.asarray(actions, dtype=np.float32)
    Y = np.asarray(targets, dtype=np.float32)
    if args.max_pairs > 0 and len(S0) > args.max_pairs:
        idx = rng.choice(len(S0), size=args.max_pairs, replace=False)
        S0 = S0[idx]
        A = A[idx]
        Y = Y[idx]

    with torch.no_grad():
        zs = []
        raws = []
        for i in range(0, len(S0), 16384):
            raw = torch.from_numpy(norm.encode(S0[i : i + 16384])).to(dev)
            raws.append(raw)
            zs.append(wm.encode(raw).detach())
        Z = torch.cat(zs, dim=0)
        Raw = torch.cat(raws, dim=0)
    At = torch.from_numpy(A).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    Cond = torch.cat([Z, Raw, At], dim=-1)

    sizes = [int(Cond.shape[1])] + [args.hidden] * max(1, args.n_blocks) + [int(Yt.shape[1])]
    head = MLP(sizes, layer_norm=True).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "event": "relocate_contact_dynamics_data",
                "pairs": int(len(Cond)),
                "cond_dim": int(Cond.shape[1]),
                "target_dim": int(Yt.shape[1]),
                "chunk": int(H),
                "mean_palm_ball": float(Y[:, 0::2].mean()),
                "mean_ball_target": float(Y[:, 1::2].mean()),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, len(Cond), (args.batch_size,), device=dev)
        pred = torch.clamp(head(Cond[idx]), min=0.0)
        loss = nn.functional.smooth_l1_loss(pred, Yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "relocate_contact_dynamics_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "cond_dim": int(Cond.shape[1]),
            "target_dim": int(Yt.shape[1]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "latent_dim": int(cfg["latent_dim"]),
            "state_dim": int(spec.state_dim),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "model_path": str(args.model_path),
            "episodes_npz": [str(x) for x in args.episodes_npz],
        },
        args.out,
    )
    print(json.dumps({"event": "relocate_contact_dynamics_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
