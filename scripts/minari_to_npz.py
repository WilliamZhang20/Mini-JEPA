"""Convert a Minari (D4RL) offline dataset to the JEPA ``Episode`` npz format.

The Adroit envs have no scriptable expert, and from-scratch RL doesn't reliably
solve them, so the real data source is offline expert demonstrations. This adapts
D4RL/<task>/expert-v2 (obs/action dims match our gymnasium-robotics v1 envs) into
the same ragged (states, actions) npz that train.py / train_policy.py read via
``--episodes-npz`` — feeding both the world model and a BC controller.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Minari dataset id, e.g. D4RL/door/expert-v2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1500)
    p.add_argument("--success-only", action="store_true",
                   help="Keep only episodes whose infos['success'] ever fires (purely goal-reaching data).")
    args = p.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")
    import minari

    ds = minari.load_dataset(args.dataset)
    states_all, actions_all = [], []
    n_succ = 0
    for ep in ds.iterate_episodes():
        o = ep.observations
        if isinstance(o, dict):
            # goal env (AntMaze): flatten = concat[observation, achieved_goal, desired_goal]
            obs = np.concatenate([np.asarray(o["observation"], dtype=np.float32),
                                  np.asarray(o["achieved_goal"], dtype=np.float32),
                                  np.asarray(o["desired_goal"], dtype=np.float32)], axis=-1)
        else:
            obs = np.asarray(o, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        if obs.shape[0] != act.shape[0] + 1:
            # some datasets store obs as a dict or with mismatched length; skip oddities
            continue
        succ = 0.0
        inf = ep.infos
        if isinstance(inf, dict) and "success" in inf:
            succ = float(np.max(np.asarray(inf["success"])))
        if args.success_only and succ < 0.5:
            continue
        n_succ += int(succ >= 0.5)
        states_all.append(obs)
        actions_all.append(act)
        if len(states_all) >= args.max_episodes:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             states=np.array(states_all, dtype=object),
             actions=np.array(actions_all, dtype=object))
    print(f'{{"event": "converted", "dataset": "{args.dataset}", "out": "{args.out}", '
          f'"episodes": {len(states_all)}, "success_frac": {n_succ / max(1, len(states_all)):.3f}, '
          f'"obs_dim": {states_all[0].shape[1]}, "action_dim": {actions_all[0].shape[1]}}}', flush=True)


if __name__ == "__main__":
    main()
