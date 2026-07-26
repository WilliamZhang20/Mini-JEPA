"""Train a self-supervised topology-aware route potential for H-JEPA.

Within each demonstration, a future achieved position is a reachable goal and
its temporal separation is a supervision signal for route distance. The model
learns

    D(z_high_t, xy_t, xy_future) ~= log(1 + macro hops)

over the full trajectory, including pairs whose shortest feasible route first
moves farther away in Euclidean distance. At evaluation, feasible macro-flow
candidates are ranked by D at their predicted endpoint instead of straight-line
distance to the final goal.
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

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.algos.hwm import (
    CoordinateTopologyScorer,
    HighEncoder,
    TopologyScorer,
)
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--hwm", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--pair-step", type=int, default=None)
    parser.add_argument("--max-future-hops", type=int, default=20)
    parser.add_argument("--min-displacement", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--coordinate-only",
        action="store_true",
        help="Learn topology from xy pairs only so the scorer can rank direct waypoint samples.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=260000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, wm_cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for parameter in wm.parameters():
        parameter.requires_grad_(False)

    hwm_art = torch.load(args.hwm, map_location=dev, weights_only=False)
    hwm_cfg = hwm_art["config"]
    psi = HighEncoder(
        int(hwm_cfg["low_dim"]),
        int(hwm_cfg["abstract_dim"]),
        int(hwm_cfg["hidden"]),
    ).to(dev)
    psi.load_state_dict(hwm_art["psi"])
    psi.eval()
    for parameter in psi.parameters():
        parameter.requires_grad_(False)

    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    env.close()
    pos_slice = slice(spec.obs_dim, spec.obs_dim + spec.goal_dim)
    stride = int(args.stride or hwm_cfg["stride"])
    pair_step = int(args.pair_step or max(1, stride // 2))

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    split = max(1, int(len(episodes) * args.holdout_frac))
    episode_sets = {"holdout": episodes[:split], "train": episodes[split:]}
    arrays: dict[str, tuple[torch.Tensor, ...]] = {}
    target_scale = float(np.log1p(args.max_future_hops))

    for split_name, split_episodes in episode_sets.items():
        raw_states: list[np.ndarray] = []
        current_pos: list[np.ndarray] = []
        future_pos: list[np.ndarray] = []
        targets: list[float] = []
        for episode in split_episodes:
            states = np.asarray(episode.states, dtype=np.float32)
            if len(states) <= stride:
                continue
            for t in range(0, len(states) - stride, pair_step):
                cur = states[t, pos_slice]
                for hops in range(1, args.max_future_hops + 1):
                    future_t = t + hops * stride
                    if future_t >= len(states):
                        break
                    goal = states[future_t, pos_slice]
                    if np.linalg.norm(goal - cur) < args.min_displacement:
                        continue
                    raw_states.append(states[t])
                    current_pos.append(cur)
                    future_pos.append(goal)
                    targets.append(float(np.log1p(hops) / target_scale))

        states_np = norm.encode(np.asarray(raw_states, dtype=np.float32))
        with torch.no_grad():
            latent_parts = []
            for start in range(0, len(states_np), 16384):
                low = wm.encode(torch.from_numpy(states_np[start : start + 16384]).to(dev))
                latent_parts.append(psi(low).cpu())
            latent = torch.cat(latent_parts)
        arrays[split_name] = (
            latent,
            torch.from_numpy(np.asarray(current_pos, dtype=np.float32)),
            torch.from_numpy(np.asarray(future_pos, dtype=np.float32)),
            torch.from_numpy(np.asarray(targets, dtype=np.float32)),
        )
        print(
            json.dumps(
                {
                    "event": "topology_data",
                    "split": split_name,
                    "pairs": len(targets),
                    "episodes": len(split_episodes),
                    "max_future_hops": args.max_future_hops,
                }
            ),
            flush=True,
        )

    scorer = (
        CoordinateTopologyScorer(spec.goal_dim, args.hidden)
        if args.coordinate_only
        else TopologyScorer(
            int(hwm_cfg["abstract_dim"]),
            spec.goal_dim,
            args.hidden,
        )
    ).to(dev)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=1e-4)
    train = tuple(value.to(dev) for value in arrays["train"])
    holdout = arrays["holdout"]
    num_train = len(train[0])
    best_holdout = float("inf")
    best_state = None

    for step in range(1, args.steps + 1):
        indices = torch.randint(0, num_train, (args.batch_size,), device=dev)
        prediction = scorer(train[0][indices], train[1][indices], train[2][indices])
        loss = nn.functional.smooth_l1_loss(prediction, train[3][indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 2000 == 0:
            with torch.no_grad():
                count = min(32768, len(holdout[0]))
                hold_idx = torch.linspace(0, len(holdout[0]) - 1, count).long()
                hold_pred = scorer(
                    holdout[0][hold_idx].to(dev),
                    holdout[1][hold_idx].to(dev),
                    holdout[2][hold_idx].to(dev),
                )
                hold_loss = float(
                    nn.functional.smooth_l1_loss(
                        hold_pred, holdout[3][hold_idx].to(dev)
                    )
                )
            if hold_loss < best_holdout:
                best_holdout = hold_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in scorer.state_dict().items()
                }
            print(
                json.dumps(
                    {
                        "event": "topology_train",
                        "step": step,
                        "loss": round(float(loss), 6),
                        "holdout_loss": round(hold_loss, 6),
                    }
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or scorer.state_dict(),
            "config": {
                "abstract_dim": int(hwm_cfg["abstract_dim"]),
                "goal_dim": spec.goal_dim,
                "hidden": args.hidden,
                "stride": stride,
                "max_future_hops": args.max_future_hops,
                "target_scale": target_scale,
                "hwm": str(args.hwm),
                "model_path": str(args.model_path),
                "episodes_npz": str(args.episodes_npz),
                "holdout_loss": best_holdout,
                "architecture": "topology_distance_v1",
                "coordinate_only": bool(args.coordinate_only),
            },
        },
        args.out,
    )
    print(
        json.dumps(
            {
                "event": "topology_saved",
                "path": str(args.out),
                "best_holdout_loss": best_holdout,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
