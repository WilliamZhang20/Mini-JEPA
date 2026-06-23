"""Evaluate ONE unified JEPA world model + ONE policy across reach/push/pick.

Roadmap B evaluation: the same `--model-path` and `--policy-path` are run on each
Fetch sub-task through the canonical state adapter, and per-task success rates are
reported (so they can be compared against the three per-task specialists). For
each sub-task we score Random, the Scripted reference, the learned policy alone,
and the policy-seeded world-model MPC (with `object_present` gating in the manip
cost so the reach task ignores the absent object).

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/eval_fetch_multi.py \
        --model-path runs/fetch_multi/checkpoints/fetch_multi_model.pt \
        --policy-path runs/fetch_multi/checkpoints/fetch_multi_policy.pt \
        --episodes 30 --video-dir runs/fetch_multi/videos
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

from jepa_robotics.envs import CANONICAL_OBJECT_PRESENT_IDX, FETCH_MULTI_SUBTASKS, make_env
from jepa_robotics.evaluate import (
    JEPAMPCPolicy,
    LearnedPolicyOnly,
    RandomPolicy,
    ScriptedGoalPolicy,
    load_jepa_artifact,
    load_policy_artifact,
    rollout_policy,
)

# Per-sub-task manip reach weight. Push must NOT pull the gripper to the object
# centre (it contacts the far side); pick benefits from a small reach term; reach
# has no object so the term is gated off regardless.
SUBTASK_REACH_WEIGHT = {"reach": 0.0, "push": 0.0, "pick": 0.1}
SUBTASK_ACTION_STD = {"reach": 0.5, "push": 0.3, "pick": 0.5}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--mpc-candidates", type=int, default=128)
    p.add_argument("--mpc-horizon", type=int, default=12)
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--policy-proposal-fraction", type=float, default=0.5)
    p.add_argument("--manip-path-weight", type=float, default=0.3)
    p.add_argument("--scripted-gain", type=float, default=12.0)
    p.add_argument("--out", type=Path, default=Path("runs/fetch_multi/eval_results/fetch_multi_eval.jsonl"))
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    model, normalizer, spec, config = load_jepa_artifact(args.model_path, device)
    policy_net, _ = load_policy_artifact(args.policy_path, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {}
    for st in FETCH_MULTI_SUBTASKS:
        name, env_id, controller = st["name"], st["env_id"], st["controller"]
        max_steps = st["max_episode_steps"]

        def build_mpc():
            return JEPAMPCPolicy(
                model=model, normalizer=normalizer, spec=spec, device=device,
                candidates=args.mpc_candidates, horizon=args.mpc_horizon, seed=args.seed + 10_000,
                method="cem", score_mode="manip", cem_iters=args.cem_iters,
                action_std=SUBTASK_ACTION_STD[name],
                manip_reach_weight=SUBTASK_REACH_WEIGHT[name],
                manip_path_weight=args.manip_path_weight,
                scripted_gain=args.scripted_gain, scripted_controller=controller,
                policy_net=policy_net, policy_proposal_fraction=args.policy_proposal_fraction,
                object_present_idx=CANONICAL_OBJECT_PRESENT_IDX,
            )

        policies = [
            RandomPolicy(),
            ScriptedGoalPolicy(action_dim=spec.action_dim, controller=controller, gain=args.scripted_gain),
            LearnedPolicyOnly(model=model, policy_net=policy_net, normalizer=normalizer,
                              spec=spec, device=device, name="jepa_policy"),
            build_mpc(),
        ]
        for policy in policies:
            wants_video = (
                args.video_dir is not None and getattr(policy, "name", "").startswith("jepa_mpc")
            )
            eval_env = make_env(
                env_id, seed=args.seed, max_episode_steps=max_steps,
                canonical_task=name, render_mode="rgb_array" if wants_video else None,
            )
            video_path = (args.video_dir / f"fetch_multi_{name}_{policy.name}.mp4") if wants_video else None
            metrics = rollout_policy(eval_env, policy, episodes=args.episodes, seed=args.seed,
                                     video_path=video_path)
            eval_env.close()
            row = {"event": "cross_eval", "subtask": name, "env_id": env_id, **metrics}
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            if policy.name.startswith("jepa_mpc"):
                summary[name] = metrics["success_rate"]

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps({"event": "fetch_multi_summary", "jepa_mpc_success": summary,
                      "mean": float(np.mean(list(summary.values()))) if summary else 0.0}), flush=True)


if __name__ == "__main__":
    main()
