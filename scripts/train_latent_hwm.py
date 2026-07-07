"""Train an HWM-style high-level model in the frozen JEPA latent space.

Training is self-supervised from trajectories:

    z_t = encoder(s_t)
    z_{t+S} = target_encoder(s_{t+S})
    m_t = macro_encoder(a_{t:t+S-1})
    high_world_model(z_t, m_t) -> z_{t+S}

No reward labels or action-label policy cloning are used by the high level.
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

from jepa_robotics.algos.hwm import LatentMacroPredictor, MacroActionEncoder, sample_macro_dataset
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import covariance_regularizer, normalized_mse, variance_regularizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument("--macro-dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-var", type=float, default=0.1)
    p.add_argument("--lambda-cov", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=991)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    states, futures, chunks, starts, finals = sample_macro_dataset(episodes, args.stride, args.overlap)
    if len(states) == 0:
        raise ValueError("No macro training windows were found.")
    with torch.no_grad():
        z = wm.encode(torch.from_numpy(norm.encode(states)).to(dev))
        z_future = wm.encode_target(torch.from_numpy(norm.encode(futures)).to(dev))
    chunk_t = torch.from_numpy(chunks).to(dev)

    macro = MacroActionEncoder(spec.action_dim, args.macro_dim).to(dev)
    pred = LatentMacroPredictor(int(cfg["latent_dim"]), args.macro_dim, args.hidden, args.n_blocks).to(dev)
    opt = torch.optim.AdamW(list(macro.parameters()) + list(pred.parameters()), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "event": "latent_hwm_data",
                "pairs": int(len(states)),
                "stride": int(args.stride),
                "macro_dim": int(args.macro_dim),
                "latent_dim": int(cfg["latent_dim"]),
                "start_goal_pairs": int(len(starts)),
            }
        ),
        flush=True,
    )
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, z.shape[0], (args.batch_size,), device=dev)
        m = macro(chunk_t[idx])
        z_pred = pred(z[idx], m)
        l_pred = normalized_mse(z_pred, z_future[idx])
        l_var = variance_regularizer(z_pred)
        l_cov = covariance_regularizer(z_pred)
        loss = l_pred + args.lambda_var * l_var + args.lambda_cov * l_cov
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(macro.parameters()) + list(pred.parameters()), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(
                json.dumps(
                    {
                        "event": "latent_hwm_train",
                        "step": int(step),
                        "pred": float(l_pred.detach().cpu()),
                        "var": float(l_var.detach().cpu()),
                        "cov": float(l_cov.detach().cpu()),
                    }
                ),
                flush=True,
            )

    with torch.no_grad():
        macros = macro(chunk_t)
        macro_mean = macros.mean(0).detach().cpu().numpy()
        macro_std = macros.std(0).clamp_min(0.05).detach().cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "macro_state": macro.state_dict(),
            "predictor_state": pred.state_dict(),
            "config": {
                "latent_dim": int(cfg["latent_dim"]),
                "action_dim": int(spec.action_dim),
                "macro_dim": int(args.macro_dim),
                "hidden": int(args.hidden),
                "n_blocks": int(args.n_blocks),
                "stride": int(args.stride),
                "macro_mean": macro_mean.tolist(),
                "macro_std": macro_std.tolist(),
                "model_path": str(args.model_path),
                "episodes_npz": str(args.episodes_npz),
            },
            "start_states": starts.astype(np.float32),
            "final_states": finals.astype(np.float32),
        },
        args.out,
    )
    print(json.dumps({"event": "latent_hwm_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
