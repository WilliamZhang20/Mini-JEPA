"""Collect successful rollouts from a flat future-conditioned flow policy.

This is SSL self-imitation data: the current SSL controller generates trials,
and successful trials are saved as transition evidence for the next
future-conditioned action prior. The collector does not use a BC/RL policy.
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

from jepa_robotics.algos.priors import EpsNet
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from scripts.eval_flat_future_flow import FlatFlowPolicy
from scripts.eval_flat_future_inverse import NearestFutureIndex


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--flow-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=70000)
    p.add_argument("--keep-success-only", action="store_true", default=True)
    p.add_argument("--keep-all", action="store_true",
                   help="Save both successful and failed rollouts for contact-dynamics calibration.")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.1)
    p.add_argument("--action-l2-weight", type=float, default=0.0)
    p.add_argument("--action-delta-weight", type=float, default=0.001)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    wm, norm, spec, _cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    ckpt = torch.load(args.flow_path, map_location=dev, weights_only=False)
    flow = EpsNet(
        int(ckpt["chunk_dim"]),
        int(ckpt["cond_dim"]),
        int(ckpt["hidden"]),
        n_blocks=int(ckpt["n_blocks"]),
    ).to(dev)
    flow.load_state_dict(ckpt["ema"])
    flow.eval()
    future_index = NearestFutureIndex(np.asarray(ckpt["bank_states"]), np.asarray(ckpt["bank_futures"]), norm)
    policy = FlatFlowPolicy(
        wm=wm,
        normalizer=norm,
        spec=spec,
        flow=flow,
        ckpt=ckpt,
        future_index=future_index,
        device=dev,
        candidates=args.candidates,
        exec_k=args.exec_k,
        flow_steps=args.flow_steps,
        target_horizon=args.target_horizon,
        latent_weight=args.latent_weight,
        state_weight=args.state_weight,
        action_l2_weight=args.action_l2_weight,
        action_delta_weight=args.action_delta_weight,
        action_scale=args.action_scale,
    )

    states_out, actions_out, successes = [], [], []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()
        states = [flatten_obs(obs).copy()]
        actions = []
        term = trunc = False
        info = {}
        while not (term or trunc):
            action = policy.act(obs, env)
            obs, _, term, trunc, info = env.step(action)
            actions.append(action.copy())
            states.append(flatten_obs(obs).copy())
        env.close()
        success = float(info.get("is_success", info.get("success", 0.0)))
        successes.append(success)
        keep = bool(args.keep_all or (not args.keep_success_only) or success > 0)
        if keep:
            states_out.append(np.asarray(states, dtype=np.float32))
            actions_out.append(np.asarray(actions, dtype=np.float32))
        print(json.dumps({"event": "collect_flat_flow_ep", "episode": ep, "success": success, "kept": int(keep)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        states=np.array(states_out, dtype=object),
        actions=np.array(actions_out, dtype=object),
        rewards=np.array([np.ones(len(a), dtype=np.float32) for a in actions_out], dtype=object),
    )
    print(
        json.dumps(
            {
                "event": "collect_flat_flow_saved",
                "out": str(args.out),
                "episodes": int(args.episodes),
                "kept": int(len(states_out)),
                "success_rate": float(np.mean(successes) if successes else 0.0),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
