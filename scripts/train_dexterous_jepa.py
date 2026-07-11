"""Train the DexterousJEPA transformer world model for Shadow Hand tasks.

Self-supervised JEPA objective on offline demos:

    z_t      = encode(s_t)
    z_future = target_encoder(s_{t+h})            # EMA, stop-gradient
    pred     = predict(z_t, a_{t:t+h}, h)
    loss = normalized_mse(pred, sg[z_future])     # JEPA prediction
         + VICReg(z_t)                            # anti-collapse
         + state-probe MSE                        # decodable geometry
         + contact-consistency MSE                # DexWM-style fingertip/relative detail

Saves in the load_jepa_artifact format with ``config["arch"]="dexterous"`` so it
runs under every existing planner/eval. GPU-train in the GPU session:

    python scripts/train_dexterous_jepa.py --task adroit_relocate \
      --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
      --out runs/adroit_relocate/checkpoints/relocate_dexterous_jepa.pt \
      --horizons 1,2,4,8,16 --latent-dim 192 --d-model 256 --enc-depth 4 \
      --dyn-depth 4 --heads 8 --ensemble-heads 3 --contact-dims 30,39 \
      --steps 60000 --batch-size 256 --device cuda
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

from jepa_robotics.data import fit_normalizer, load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.models import DexterousJEPA, normalized_mse, variance_regularizer, covariance_regularizer
from jepa_robotics.tasks import resolve_task


def parse_ints(s):
    return [int(x) for x in s.split(",")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizons", type=parse_ints, default=[1, 2, 4, 8, 16])
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--latent-dim", type=int, default=192)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=4)
    p.add_argument("--dyn-depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ensemble-heads", type=int, default=3)
    p.add_argument("--contact-dims", default=None, help="lo,hi raw-state slice for the contact-consistency head (e.g. 30,39 for Relocate palm-ball+ball-target)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=0.5)
    p.add_argument("--lambda-state", type=float, default=1.0)
    p.add_argument("--lambda-contact", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    norm = fit_normalizer(episodes)
    max_h = max(args.horizons)
    contact = tuple(parse_ints(args.contact_dims)) if args.contact_dims else None

    # Precompute per-episode normalized states; sample (ep, t, h) transitions on the fly.
    ep_states = [norm.encode(ep.states.astype(np.float32)) for ep in episodes if len(ep.actions) > max_h]
    ep_actions = [ep.actions.astype(np.float32) for ep in episodes if len(ep.actions) > max_h]
    lengths = np.array([len(a) for a in ep_actions])
    ep_ids = np.arange(len(ep_actions))
    rng = np.random.default_rng(0)

    wm = DexterousJEPA(
        state_dim=spec.state_dim, action_dim=spec.action_dim, latent_dim=args.latent_dim,
        d_model=args.d_model, enc_depth=args.enc_depth, dyn_depth=args.dyn_depth, heads=args.heads,
        max_horizon=max_h, ensemble_heads=args.ensemble_heads, contact_dims=contact, dropout=args.dropout,
    ).to(dev)
    opt = torch.optim.AdamW(wm.parameters(), lr=args.lr, weight_decay=1e-4)
    nparams = sum(p.numel() for p in wm.parameters()) / 1e6
    print(json.dumps({"event": "dex_jepa_data", "episodes": len(ep_actions), "state_dim": spec.state_dim,
                      "action_dim": spec.action_dim, "params_M": round(nparams, 2), "contact_dims": contact}), flush=True)

    def sample_batch(h):
        cur, fut, chunks = [], [], []
        for _ in range(args.batch_size):
            e = int(rng.choice(ep_ids))
            T = lengths[e]
            t = int(rng.integers(0, T - h))
            cur.append(ep_states[e][t]); fut.append(ep_states[e][t + h])
            a = ep_actions[e][t:t + h]
            if len(a) < max_h:
                a = np.concatenate([a, np.zeros((max_h - len(a), spec.action_dim), np.float32)], 0)
            chunks.append(a)
        return (torch.from_numpy(np.stack(cur)).to(dev),
                torch.from_numpy(np.stack(fut)).to(dev),
                torch.from_numpy(np.stack(chunks)).to(dev))

    for step in range(1, args.steps + 1):
        h = int(rng.choice(args.horizons))
        s_t, s_fut, chunks = sample_batch(h)
        z = wm.encode(s_t)
        with torch.no_grad():
            target = wm.encode_target(s_fut)
        pred = wm.predict(z, chunks, h)
        loss = normalized_mse(pred, target)
        loss = loss + args.lambda_var * variance_regularizer(z) + args.lambda_cov * covariance_regularizer(z)
        loss = loss + args.lambda_state * torch.nn.functional.mse_loss(wm.state_probe(z), s_t)
        if contact is not None:
            lo, hi = contact
            loss = loss + args.lambda_contact * torch.nn.functional.mse_loss(wm.contact_consistency(z), s_t[:, lo:hi])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        wm.update_target(args.ema)
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "dex_jepa_train", "step": step, "h": h, "loss": round(float(loss.detach()), 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": wm.state_dict(),
        "normalizer": {"mean": norm.mean, "std": norm.std},
        "spec": spec.__dict__,
        "config": {
            "task": args.task, "env_id": task.env_id, "arch": "dexterous",
            "horizons": args.horizons, "latent_dim": args.latent_dim, "hidden_dim": args.d_model,
            "d_model": args.d_model, "enc_depth": args.enc_depth, "dyn_depth": args.dyn_depth,
            "heads": args.heads, "max_horizon": max_h, "ensemble_heads": args.ensemble_heads,
            "contact_dims": list(contact) if contact else None,
        },
    }, args.out)
    print(json.dumps({"event": "dex_jepa_saved", "path": str(args.out), "params_M": round(nparams, 2)}), flush=True)


if __name__ == "__main__":
    main()
