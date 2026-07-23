"""Train a self-supervised inverse action-chunk prior for Fetch.

The prior learns ``a_{t:t+H-1} = inverse(z_t, z_{t+h}, h, geometry)`` from
trial transitions. It is not executed as a policy during training; at eval time
the JEPA world model can still score noisy candidates around its prediction.
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

from jepa_robotics.algos.task_families.fetch import geometry_features as fetch_geometry_features
from jepa_robotics.algos.priors import InversePrior, parse_horizons
from jepa_robotics.data import collect_episodes, load_episodes_npz, load_spec_npz
from jepa_robotics.envs import goal_state_from_state, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, default=None,
                   help="Optional offline trial trajectories; skips live collection.")
    p.add_argument("--collect-steps", type=int, default=120_000)
    p.add_argument("--scripted-fraction", type=float, default=0.95)
    p.add_argument("--controller-gain", type=float, default=12.0)
    p.add_argument("--action-noise", type=float, default=0.05)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--concat-geometry", action="store_true")
    p.add_argument("--condition-on-goal-state", action="store_true",
                   help="Use the episode desired-goal state as z_future instead of the observed t+h state.")
    p.add_argument("--train-steps", type=int, default=35_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    task = resolve_task(args.task, None)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wm, normalizer, spec, cfg = load_jepa_artifact(args.model_path, device)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    if args.episodes_npz is not None:
        episodes = load_episodes_npz(args.episodes_npz)
        env_spec = load_spec_npz(args.episodes_npz) or spec
    else:
        env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
        episodes, env_spec = collect_episodes(
            env,
            num_steps=args.collect_steps,
            seed=args.seed,
            scripted_fraction=args.scripted_fraction,
            controller_gain=args.controller_gain,
            action_noise=args.action_noise,
            controller=task.controller,
            log_every=max(1, args.collect_steps // 5),
        )
        env.close()
    if env_spec != spec:
        raise ValueError(f"Model spec {spec} does not match collected env spec {env_spec}.")

    H = args.chunk
    future_horizons = args.future_horizons or [H]
    max_future = max(max(future_horizons), H)
    conds, chunks, geom_feats = [], [], []
    with torch.no_grad():
        for ep in episodes:
            S = ep.states.astype(np.float32)
            A = ep.actions.astype(np.float32)
            if len(A) < max_future:
                continue
            Sn = torch.from_numpy(normalizer.encode(S)).to(device)
            z_online = wm.encode(Sn)
            if args.condition_on_goal_state:
                G = np.stack([goal_state_from_state(s, spec) for s in S]).astype(np.float32)
                Gn = torch.from_numpy(normalizer.encode(G)).to(device)
                z_goal_states = wm.encode_target(Gn)
            else:
                z_target = wm.encode_target(Sn)
            for t in range(len(A) - max_future + 1):
                for future_h in future_horizons:
                    h_token = torch.tensor([float(future_h) / float(max(future_horizons))], dtype=z_online.dtype, device=device)
                    if args.condition_on_goal_state:
                        target_z = z_goal_states[t]
                        target_state = G[t]
                    else:
                        target_z = z_target[t + future_h]
                        target_state = S[t + future_h]
                    conds.append(torch.cat([z_online[t], target_z, h_token], dim=-1))
                    chunks.append(torch.from_numpy(A[t : t + H].reshape(-1)).to(device))
                    if args.concat_geometry:
                        geom_feats.append(fetch_geometry_features(S[t], target_state, spec))
    Cond = torch.stack(conds, dim=0)
    geom_mean = geom_std = None
    if args.concat_geometry:
        Geom = torch.from_numpy(np.stack(geom_feats).astype(np.float32)).to(device)
        geom_mean = Geom.mean(dim=0, keepdim=True)
        geom_std = Geom.std(dim=0, keepdim=True).clamp_min(1e-6)
        Cond = torch.cat([Cond, (Geom - geom_mean) / geom_std], dim=-1)
    Chunk = torch.stack(chunks, dim=0)
    net = InversePrior(Cond.shape[1], Chunk.shape[1], args.hidden, args.n_blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    print(json.dumps({"event": "inverse_prior_data", "pairs": int(Cond.shape[0]), "chunk": H, "cond_dim": int(Cond.shape[1]), "chunk_dim": int(Chunk.shape[1])}), flush=True)
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, Cond.shape[0], (args.batch_size,), device=device)
        pred = net(Cond[idx])
        loss = nn.functional.mse_loss(pred, Chunk[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "inverse_prior_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "cond_dim": int(Cond.shape[1]),
            "chunk_dim": int(Chunk.shape[1]),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "future_horizons": future_horizons,
            "concat_geometry": bool(args.concat_geometry),
            "geom_mean": None if geom_mean is None else geom_mean.squeeze(0).detach().cpu().numpy(),
            "geom_std": None if geom_std is None else geom_std.squeeze(0).detach().cpu().numpy(),
            "model_path": str(args.model_path),
            "task": task.name,
            "episodes_npz": None if args.episodes_npz is None else str(args.episodes_npz),
            "condition_on_goal_state": bool(args.condition_on_goal_state),
        },
        args.out,
    )
    print(json.dumps({"event": "inverse_prior_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
