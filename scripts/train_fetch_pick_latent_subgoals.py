"""Build self-supervised latent subgoals for Fetch push/pick experiments.

This is an experimental alternative to BC/RL-tuned controllers. Demos identify
desirable futures; their action labels are not a supervised policy target and
are not saved in this artifact. The already-trained JEPA dynamics model remains
the source of "which actions cause which futures".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import collect_episodes
from jepa_robotics.envs import make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.subgoals import (
    build_fetch_pick_subgoal_artifact,
    build_fetch_push_subgoal_artifact,
    save_subgoal_artifact,
)
from jepa_robotics.tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fetch_pick_place")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collect-steps", type=int, default=80_000)
    parser.add_argument("--scripted-fraction", type=float, default=1.0)
    parser.add_argument("--controller-gain", type=float, default=12.0)
    parser.add_argument("--action-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    task = resolve_task(args.task, None)
    if task.name not in {"fetch_pick_place", "fetch_push"}:
        raise ValueError("This builder is scoped to FetchPickAndPlace and FetchPush.")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, normalizer, spec, _config = load_jepa_artifact(args.model_path, device)
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

    build = build_fetch_push_subgoal_artifact if task.name == "fetch_push" else build_fetch_pick_subgoal_artifact
    artifact = build(episodes, spec, model=model, normalizer=normalizer, device=device)
    save_subgoal_artifact(args.out, artifact)
    counts = {phase: int(artifact["templates"][phase]["count"]) for phase in artifact["phases"]}
    print(
        json.dumps(
            {
                "event": "latent_subgoals_saved",
                "path": str(args.out),
                "episodes": len(episodes),
                "states": int(sum(len(ep.states) for ep in episodes)),
                "phase_counts": counts,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
