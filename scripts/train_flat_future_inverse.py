"""Train a future-conditioned inverse chunk prior for flat offline-demo tasks.

For non-goal tasks such as Adroit, demos supply desirable futures but there is no
explicit desired_goal field. Training uses self-supervised transition tuples:

    z_t = encoder(s_t)
    z_future = target_encoder(s_{t+h})
    inverse(z_t, z_future, h) -> a_{t:t+H-1}

The saved demo future bank lets evaluation retrieve a nearby demonstrated future
state without copying its action label.
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
from jepa_robotics.algos.priors import parse_horizons
from jepa_robotics.algos.priors import InversePrior


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--trial-episodes-npz", type=Path, nargs="*", default=[],
                   help="Optional trial rollout npz files. Their hindsight (state, reached-future, action-chunk) pairs join training, but they are excluded from the demo future bank so retrieval still targets demonstrated futures.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-bank", type=int, default=50000)
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--concat-raw", action="store_true",
                   help="Append normalized current/future states to the latent condition.")
    p.add_argument("--input-noise-std", type=float, default=0.0,
                   help="Gaussian noise std (normalized state units) applied to the CURRENT state input at every train step, re-encoded through the frozen encoder. Future targets and action chunks stay clean, so this teaches the inverse to funnel nearby off-manifold states back onto the demo without diluting the expert action manifold.")
    p.add_argument("--require-possession", choices=["none", "held", "free"], default="none",
                   help="Segment-pure specialist training: 'held' keeps only pairs whose current AND future frames satisfy the possession predicate (raw-state dims below), 'free' keeps only pairs whose current frame does not. Specialists split a bimodal regime without blurring the expert manifold.")
    p.add_argument("--possession-dims", default="30,33",
                   help="Raw-state slice (lo,hi) whose vector norm defines the possession predicate.")
    p.add_argument("--possession-threshold", type=float, default=0.06)
    p.add_argument("--emphasis-dims", default=None,
                   help="Raw-state slice (lo,hi) of the CURRENT state to duplicate --emphasis-repeat extra times in the conditioning. Upweights live contact geometry (e.g. the palm-ball vector 30,33) so the closure chunk servos to the live ball, not the demo ball. Targets and futures are unchanged, so the expert action manifold is not blurred.")
    p.add_argument("--emphasis-repeat", type=int, default=0)
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    trial_episodes = []
    for trial_path in args.trial_episodes_npz:
        trial_episodes.extend(load_episodes_npz(trial_path))
    H = args.chunk
    future_horizons = args.future_horizons or [H]
    max_future = max(max(future_horizons), H)
    pred_lo, pred_hi = (int(x) for x in args.possession_dims.split(","))
    cur_states, fut_states, fut_targets, h_tokens, chunks = [], [], [], [], []
    bank_states, bank_futures = [], []
    with torch.no_grad():
        for in_bank, ep in [(True, ep) for ep in episodes] + [(False, ep) for ep in trial_episodes]:
            S = ep.states.astype(np.float32)
            A = ep.actions.astype(np.float32)
            if len(A) < max_future:
                continue
            held = np.linalg.norm(S[:, pred_lo:pred_hi], axis=-1) < args.possession_threshold
            Sn = torch.from_numpy(norm.encode(S)).to(dev)
            z_target = wm.encode_target(Sn)
            for t in range(len(A) - max_future + 1):
                for future_h in future_horizons:
                    if args.require_possession == "held" and not (held[t] and held[t + future_h]):
                        continue
                    if args.require_possession == "free" and held[t]:
                        continue
                    cur_states.append(Sn[t])
                    fut_states.append(Sn[t + future_h])
                    fut_targets.append(z_target[t + future_h])
                    h_tokens.append(torch.tensor([float(future_h) / float(max(future_horizons))], dtype=Sn.dtype, device=dev))
                    chunks.append(torch.from_numpy(A[t : t + H].reshape(-1)).to(dev))
                if in_bank:
                    bank_states.append(S[t])
                    bank_futures.append(S[t + max(future_horizons)])

    SCur = torch.stack(cur_states)
    SFut = torch.stack(fut_states)
    ZFut = torch.stack(fut_targets)
    HTok = torch.stack(h_tokens)
    Chunk = torch.stack(chunks)

    emph_lo = emph_hi = None
    if args.emphasis_dims is not None and args.emphasis_repeat > 0:
        emph_lo, emph_hi = (int(x) for x in args.emphasis_dims.split(","))

    def make_cond(idx: torch.Tensor) -> torch.Tensor:
        s = SCur[idx]
        if args.input_noise_std > 0:
            s = s + torch.randn_like(s) * args.input_noise_std
        z = wm.encode(s)
        parts = [z, ZFut[idx], HTok[idx]]
        if args.concat_raw:
            parts.extend([s, SFut[idx]])
        if emph_lo is not None:
            parts.append(s[:, emph_lo:emph_hi].repeat(1, args.emphasis_repeat))
        return torch.cat(parts, dim=-1)

    with torch.no_grad():
        cond_dim = int(make_cond(torch.arange(1, device=dev)).shape[1])
    bank_states_np = np.asarray(bank_states, dtype=np.float32)
    bank_futures_np = np.asarray(bank_futures, dtype=np.float32)
    if len(bank_states_np) > args.max_bank:
        idx = rng.choice(len(bank_states_np), size=args.max_bank, replace=False)
        bank_states_np = bank_states_np[idx]
        bank_futures_np = bank_futures_np[idx]

    prior = InversePrior(cond_dim, Chunk.shape[1], args.hidden, args.n_blocks).to(dev)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)
    print(json.dumps({"event": "flat_inverse_data", "pairs": int(SCur.shape[0]), "bank": int(len(bank_states_np)), "chunk": H, "cond_dim": cond_dim, "chunk_dim": int(Chunk.shape[1]), "concat_raw": bool(args.concat_raw), "input_noise_std": float(args.input_noise_std)}), flush=True)
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, SCur.shape[0], (args.batch_size,), device=dev)
        with torch.no_grad():
            cond = make_cond(idx)
        pred = prior(cond)
        loss = nn.functional.mse_loss(pred, Chunk[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "flat_inverse_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "cond_dim": cond_dim,
            "chunk_dim": int(Chunk.shape[1]),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "future_horizons": future_horizons,
            "concat_raw": bool(args.concat_raw),
            "bank_states": bank_states_np,
            "bank_futures": bank_futures_np,
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
            "trial_episodes_npz": [str(x) for x in args.trial_episodes_npz],
            "input_noise_std": float(args.input_noise_std),
            "require_possession": args.require_possession,
            "possession_threshold": float(args.possession_threshold),
            "emphasis_dims": args.emphasis_dims if emph_lo is not None else None,
            "emphasis_repeat": int(args.emphasis_repeat) if emph_lo is not None else 0,
        },
        args.out,
    )
    print(json.dumps({"event": "flat_inverse_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
