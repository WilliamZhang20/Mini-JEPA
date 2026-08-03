"""Train a dedicated future-conditioned actor on the frozen JEPA latent.

The world model's inverse head is an *auxiliary* of the JEPA objective — its job
is to keep the encoder control-aware, and it is trained at a single horizon with
a small loss weight. Using it as the controller therefore under-sells the
learned-structure setup against the phase-inverse baseline, which has a
dedicated 512-wide chunk prior.

This trains that dedicated prior with **no hand-supplied structure**: the
condition is ``(z_t, z_goal, horizon)`` only. There is no phase index, no phase
count, no monotonicity rule, no stored demo bank, and no task-specific emphasis
or switch threshold. At training time ``z_goal`` is the encoded true future
``h`` steps ahead; at run time it comes from the learned subgoal net, so the
actor never sees a hand-built target.
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.priors import (
    FlowChunkActor,
    InversePrior,
    PredictorGuidedRefiner,
    parse_horizons,
)
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact


@torch.no_grad()
def encode_all(wm, arr: np.ndarray, device, target: bool, batch: int = 65536) -> torch.Tensor:
    out = []
    for i in range(0, len(arr), batch):
        x = torch.from_numpy(arr[i : i + batch]).to(device)
        out.append(wm.encode_target(x) if target else wm.encode(x))
    return torch.cat(out, dim=0)


def build_rows(episodes, normalizer, horizons: list[int], chunk: int):
    """(state index, future index, horizon, action chunk) rows across all demos."""
    states, futures, h_col, chunks = [], [], [], []
    cursor = 0
    flat_states = []
    for ep in episodes:
        raw = np.asarray(ep.states, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        n = min(len(raw) - 1, len(act))
        if n < max(max(horizons), chunk) + 1:
            continue
        flat_states.append(normalizer.encode(raw[: n + 1]))
        for h in horizons:
            last = n - max(h, chunk)
            if last <= 0:
                continue
            t = np.arange(last, dtype=np.int64)
            states.append(cursor + t)
            futures.append(cursor + t + h)
            h_col.append(np.full(len(t), float(h), dtype=np.float32))
            chunks.append(np.stack([act[i : i + chunk].reshape(-1) for i in t]))
        cursor += n + 1
    return (
        np.concatenate(flat_states),
        np.concatenate(states),
        np.concatenate(futures),
        np.concatenate(h_col),
        np.concatenate(chunks),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--actor-type", choices=["inverse", "flow", "refiner"], default="inverse",
                   help="'inverse' is a deterministic MLP; 'flow' is a rectified-flow "
                        "prior whose samples give diverse on-manifold rank candidates; "
                        "'refiner' corrects a chunk from the predictor's own rollout error")
    p.add_argument("--flow-steps", type=int, default=16,
                   help="flow integration steps at sampling time (stored in the checkpoint)")
    p.add_argument("--refine-noise-max", type=float, default=0.6,
                   help="refiner training: max std of the perturbation applied to the "
                        "demo chunk before asking the head to correct it back")
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--horizons", type=parse_horizons, default="2,4,8")
    p.add_argument("--max-episodes", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=3000)
    p.add_argument("--seed", type=int, default=17)
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
    flat, s_idx, f_idx, h_col, chunks = build_rows(
        episodes, normalizer, list(args.horizons), args.chunk
    )
    print(json.dumps({"event": "rows", "rows": int(len(s_idx)), "states": int(len(flat)),
                      "chunk": args.chunk, "horizons": list(args.horizons)}), flush=True)

    z_online = encode_all(wm, flat, device, target=False)
    z_target = encode_all(wm, flat, device, target=True)
    s_idx_t = torch.from_numpy(s_idx).to(device)
    f_idx_t = torch.from_numpy(f_idx).to(device)
    h_t = torch.from_numpy(h_col).to(device).unsqueeze(-1) / float(max(args.horizons))
    chunk_t = torch.from_numpy(chunks).to(device)

    cond_dim = 2 * int(cfg["latent_dim"]) + 1
    chunk_dim = int(chunk_t.shape[1])
    if args.actor_type == "flow":
        actor = FlowChunkActor(
            cond_dim, chunk_dim, args.hidden, args.n_blocks, flow_steps=args.flow_steps
        ).to(device)
    elif args.actor_type == "refiner":
        actor = PredictorGuidedRefiner(
            int(cfg["latent_dim"]), chunk_dim, args.hidden, args.n_blocks
        ).to(device)
    else:
        actor = InversePrior(cond_dim, chunk_dim, args.hidden, args.n_blocks).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.train_steps, eta_min=args.lr * 0.05
    )

    perm = torch.from_numpy(rng.permutation(len(s_idx))).to(device)
    n_hold = max(1, int(len(s_idx) * args.holdout_frac))
    hold, train = perm[:n_hold], perm[n_hold:]

    def cond_of(idx):
        return torch.cat([z_online[s_idx_t[idx]], z_target[f_idx_t[idx]], h_t[idx]], dim=-1)

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": actor.state_dict(),
            "actor_type": args.actor_type,
            "flow_steps": int(args.flow_steps),
            "cond_dim": cond_dim,
            "chunk_dim": chunk_dim,
            "action_dim": int(spec.action_dim),
            "chunk": int(args.chunk),
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "horizons": list(args.horizons),
            "max_horizon": max(args.horizons),
            "latent_dim": int(cfg["latent_dim"]),
            "model_path": str(args.model_path),
            "episodes_npz": str(args.episodes_npz),
        }, args.out)

    act_dim = int(spec.action_dim)

    def batch_loss(idx):
        if args.actor_type == "flow":
            return actor.loss(chunk_t[idx], cond_of(idx))
        if args.actor_type == "refiner":
            a_star = chunk_t[idx]
            sigma = 0.05 + torch.rand(a_star.shape[0], 1, device=device) * args.refine_noise_max
            a0 = (a_star + sigma * torch.randn_like(a_star)).clamp(-1.0, 1.0)
            z_b = z_online[s_idx_t[idx]]
            zg_b = z_target[f_idx_t[idx]]
            with torch.no_grad():
                z_end = wm.rollout_heads(
                    z_b, a0.view(-1, args.chunk, act_dim), args.chunk
                ).mean(dim=0)[:, -1]
            delta = actor(z_b, zg_b, zg_b - z_end, h_t[idx], a0)
            return nn.functional.mse_loss(delta, a_star - a0)
        return nn.functional.mse_loss(actor(cond_of(idx)), chunk_t[idx])

    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        idx = train[torch.randint(0, train.numel(), (args.batch_size,), device=device)]
        loss = batch_loss(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            with torch.no_grad():
                hidx = hold[torch.randint(0, hold.numel(), (min(8192, hold.numel()),), device=device)]
                hl = batch_loss(hidx)
            print(json.dumps({"event": "train", "step": step, "loss": float(loss.detach()),
                              "holdout": float(hl), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
            save()
    save()
    print(json.dumps({"event": "saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
