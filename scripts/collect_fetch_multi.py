"""Collect a single union dataset across FetchReach + FetchPush + FetchPickAndPlace.

Roadmap B step 2: per-episode task sampling with the canonical state adapter, so
every recorded transition shares one 35-D superset state. The resulting .npz
(states/actions + canonical ObsSpec) trains BOTH the unified world model
(``train.py --episodes-npz``) and the unified policy (``train_policy.py
--episodes-npz``).

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/collect_fetch_multi.py \
        --collect-steps 600000 --out runs/fetch_multi/data/union.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import collect_fetch_multi_episodes, save_episodes_npz


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--collect-steps", type=int, default=600_000)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--scripted-fraction", type=float, default=0.8)
    p.add_argument("--controller-gain", type=float, default=12.0)
    p.add_argument("--action-noise", type=float, default=0.2)
    p.add_argument("--collect-log-every", type=int, default=50_000)
    p.add_argument("--out", type=Path, default=Path("runs/fetch_multi/data/union.npz"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    episodes, spec = collect_fetch_multi_episodes(
        num_steps=args.collect_steps,
        seed=args.seed,
        scripted_fraction=args.scripted_fraction,
        controller_gain=args.controller_gain,
        action_noise=args.action_noise,
        log_every=args.collect_log_every,
    )
    save_episodes_npz(args.out, episodes, spec)
    total = int(sum(len(e.actions) for e in episodes))
    print(json.dumps({
        "event": "collected_fetch_multi",
        "path": str(args.out),
        "episodes": len(episodes),
        "transitions": total,
        "state_dim": spec.state_dim,
    }), flush=True)


if __name__ == "__main__":
    main()
