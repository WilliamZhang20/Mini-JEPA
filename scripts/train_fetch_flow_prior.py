"""Train a goal/future-conditioned flow action prior for Fetch.

This follows the loop:

    z_t = encoder(o_t)
    z_future = target_encoder(o_{t+H})
    flow(a_{t:t+H-1} | z_t, z_future)

The action chunk is modeled generatively with conditional flow matching. At
evaluation, ``z_future`` can be replaced by an encoded goal/demo-future latent
and the JEPA world model scores sampled chunks before execution.
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

from jepa_robotics.data import Episode, collect_episodes
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import SB3Policy, load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.priors import EpsNet, make_ddpm


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(v) for v in value.split(",") if v.strip()})
    if not horizons or min(horizons) < 1:
        raise argparse.ArgumentTypeError("future horizons must be positive, e.g. 4,8,12,16")
    return horizons


def fetch_geometry_features(state: np.ndarray, target_state: np.ndarray, spec) -> np.ndarray:
    """Small exact-geometry side channel for Fetch object manipulation.

    The flow is still generative over action chunks. These features tell it the
    current and desired local geometry precisely, while JEPA still supplies the
    latent representation/dynamics used for scoring.
    """
    obs = state[: spec.obs_dim]
    target_obs = target_state[: spec.obs_dim]
    grip = obs[:3]
    obj = state[spec.obs_dim : spec.obs_dim + spec.goal_dim]
    goal = state[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
    target_grip = target_obs[:3]
    target_obj = target_state[spec.obs_dim : spec.obs_dim + spec.goal_dim]
    return np.concatenate(
        [
            grip - obj,
            obj - goal,
            grip - goal,
            target_obj - obj,
            target_grip - grip,
            np.array(
                [
                    np.linalg.norm(obj - goal),
                    np.linalg.norm(target_obj - obj),
                    np.linalg.norm(target_obj - goal),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32)


def collect_policy_episodes(env, policy, *, num_steps: int, seed: int, log_every: int = 0):
    spec = obs_spec_from_env(env)
    episodes: list[Episode] = []
    total_steps = 0
    episode_idx = 0
    while total_steps < num_steps:
        obs, _ = env.reset(seed=seed + episode_idx)
        if hasattr(policy, "reset"):
            policy.reset()
        states = [flatten_obs(obs)]
        actions = []
        terminated = truncated = False
        while not (terminated or truncated) and total_steps < num_steps:
            action = policy.act(obs, env)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            next_obs, _, terminated, truncated, _info = env.step(action)
            actions.append(action)
            states.append(flatten_obs(next_obs))
            obs = next_obs
            total_steps += 1
            if log_every > 0 and total_steps % log_every == 0:
                print(
                    f'{{"event": "collect", "steps": {total_steps}, "target_steps": {num_steps}}}',
                    flush=True,
                )
        if actions:
            episodes.append(
                Episode(
                    states=np.stack(states).astype(np.float32),
                    actions=np.stack(actions).astype(np.float32),
                )
            )
        episode_idx += 1
    return episodes, spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_push")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collect-steps", type=int, default=100_000)
    p.add_argument("--scripted-fraction", type=float, default=0.9)
    p.add_argument("--trial-policy-path", type=Path, default=None,
                   help="Optional SB3 checkpoint used only to collect trial trajectories.")
    p.add_argument("--controller-gain", type=float, default=12.0)
    p.add_argument("--action-noise", type=float, default=0.15)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument(
        "--future-horizons",
        type=parse_horizons,
        default=None,
        help="Comma-separated future horizons to condition on. Each sample is "
             "flow(a_{t:t+chunk-1} | z_t, z_{t+h}, h). Defaults to --chunk.",
    )
    p.add_argument("--concat-geometry", action="store_true",
                   help="Append standardized Fetch geometry features to the latent condition.")
    p.add_argument("--train-steps", type=int, default=40_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=61)
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

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    if args.trial_policy_path is not None:
        trial_policy = SB3Policy(args.trial_policy_path, name="trial_policy", env=env)
        episodes, env_spec = collect_policy_episodes(
            env,
            trial_policy,
            num_steps=args.collect_steps,
            seed=args.seed,
            log_every=max(1, args.collect_steps // 5),
        )
    else:
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
    conds, chunks = [], []
    geom_feats = []
    with torch.no_grad():
        for ep in episodes:
            S = ep.states.astype(np.float32)
            A = ep.actions.astype(np.float32)
            if len(A) < max_future:
                continue
            Sn = torch.from_numpy(normalizer.encode(S)).to(device)
            z_online = wm.encode(Sn)
            z_target = wm.encode_target(Sn)
            for t in range(len(A) - max_future + 1):
                for future_h in future_horizons:
                    h_token = torch.tensor(
                        [float(future_h) / float(max(future_horizons))],
                        dtype=z_online.dtype,
                        device=device,
                    )
                    conds.append(torch.cat([z_online[t], z_target[t + future_h], h_token], dim=-1))
                    chunks.append(torch.from_numpy(A[t : t + H].reshape(-1)).to(device))
                    if args.concat_geometry:
                        geom_feats.append(fetch_geometry_features(S[t], S[t + future_h], spec))
    Cond = torch.stack(conds, dim=0)
    geom_mean = geom_std = None
    if args.concat_geometry:
        Geom = torch.from_numpy(np.stack(geom_feats).astype(np.float32)).to(device)
        geom_mean = Geom.mean(dim=0, keepdim=True)
        geom_std = Geom.std(dim=0, keepdim=True).clamp_min(1e-6)
        Cond = torch.cat([Cond, (Geom - geom_mean) / geom_std], dim=-1)
    Chunk = torch.stack(chunks, dim=0)
    n = Cond.shape[0]
    chunk_dim = Chunk.shape[1]
    cond_dim = Cond.shape[1]
    action_dim = spec.action_dim
    print(
        json.dumps(
            {
                "event": "flow_prior_data",
                "pairs": int(n),
                "chunk": H,
                "chunk_dim": int(chunk_dim),
                "cond_dim": int(cond_dim),
                "action_dim": int(action_dim),
                "future_horizons": future_horizons,
                "concat_geometry": bool(args.concat_geometry),
                "trial_policy_path": None if args.trial_policy_path is None else str(args.trial_policy_path),
            }
        ),
        flush=True,
    )

    net = EpsNet(chunk_dim, cond_dim, args.hidden, n_blocks=args.n_blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema_decay = 0.999
    Tsteps = args.flow_steps
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        a1 = Chunk[idx]
        cond = Cond[idx]
        tau = torch.rand(args.batch_size, device=device)
        a0 = torch.randn_like(a1)
        a_tau = (1.0 - tau)[:, None] * a0 + tau[:, None] * a1
        target_v = a1 - a0
        pred_v = net(a_tau, tau * Tsteps, cond)
        loss = nn.functional.mse_loss(pred_v, target_v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                ema[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
        if step == 1 or step % 2000 == 0:
            print(json.dumps({"event": "flow_prior_train", "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": ema,
            "state_dict": net.state_dict(),
            "chunk_dim": int(chunk_dim),
            "cond_dim": int(cond_dim),
            "action_dim": int(action_dim),
            "H": int(H),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "diffusion_steps": int(args.flow_steps),
            "objective": "flow",
            "conditioning": "z_t_z_future",
            "future_horizons": future_horizons,
            "concat_geometry": bool(args.concat_geometry),
            "geom_mean": None if geom_mean is None else geom_mean.squeeze(0).detach().cpu().numpy(),
            "geom_std": None if geom_std is None else geom_std.squeeze(0).detach().cpu().numpy(),
            "model_path": str(args.model_path),
            "task": task.name,
            "trial_policy_path": None if args.trial_policy_path is None else str(args.trial_policy_path),
        },
        args.out,
    )
    print(json.dumps({"event": "flow_prior_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
