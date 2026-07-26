"""Train a full-horizon flow directly over demonstrated HWM waypoints.

The original HWM samples a macro latent, predicts its endpoint through ``g``,
and decodes the endpoint to xy. Around walls, small latent-model errors can
decode to impossible straight-through-wall subgoals. This model removes that
bottleneck:

    waypoint_flow(xy_next | z_high_t, xy_future)

Both the output waypoint and the conditioning goal come from the same
demonstrated trajectory. Full-trajectory future conditioning teaches the first
detour waypoint for distant goals.
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

from jepa_robotics.algos.hwm import HighEncoder
from jepa_robotics.algos.priors import EpsNet
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
    parser.add_argument("--max-future-hops", type=int, default=20)
    parser.add_argument(
        "--condition-desired-goal",
        action="store_true",
        help="Condition every transition on its episode's final desired goal.",
    )
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=261000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, _ = load_jepa_artifact(args.model_path, dev)
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
    desired_slice = slice(
        spec.obs_dim + spec.goal_dim,
        spec.obs_dim + 2 * spec.goal_dim,
    )
    stride = int(hwm_cfg["stride"])
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]

    current_states: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    next_positions: list[np.ndarray] = []
    for episode in episodes:
        states = np.asarray(episode.states, dtype=np.float32)
        if len(states) <= stride:
            continue
        for t in range(0, len(states) - stride, max(1, stride // 2)):
            next_pos = states[t + stride, pos_slice]
            if args.condition_desired_goal:
                current_states.append(states[t])
                target_positions.append(states[t, desired_slice])
                next_positions.append(next_pos)
                continue
            for hops in range(1, args.max_future_hops + 1):
                future_t = t + hops * stride
                if future_t >= len(states):
                    break
                current_states.append(states[t])
                target_positions.append(states[future_t, pos_slice])
                next_positions.append(next_pos)

    current_np = norm.encode(np.asarray(current_states, dtype=np.float32))
    with torch.no_grad():
        high_parts = []
        for start in range(0, len(current_np), 16384):
            low = wm.encode(torch.from_numpy(current_np[start : start + 16384]).to(dev))
            high_parts.append(psi(low).cpu())
        current_high = torch.cat(high_parts)
    targets = torch.from_numpy(np.asarray(target_positions, dtype=np.float32))
    waypoints = torch.from_numpy(np.asarray(next_positions, dtype=np.float32))
    condition = torch.cat([current_high, targets], dim=-1).to(dev)
    waypoints = waypoints.to(dev)
    print(
        json.dumps(
            {
                "event": "waypoint_flow_data",
                "pairs": len(waypoints),
                "cond_dim": condition.shape[1],
                "waypoint_dim": waypoints.shape[1],
                "stride": stride,
                "max_future_hops": args.max_future_hops,
                "condition_desired_goal": bool(args.condition_desired_goal),
            }
        ),
        flush=True,
    )

    net = EpsNet(
        spec.goal_dim,
        int(condition.shape[1]),
        args.hidden,
        n_blocks=args.n_blocks,
    ).to(dev)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {key: value.detach().clone() for key, value in net.state_dict().items()}
    count = len(waypoints)
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, count, (args.batch_size,), device=dev)
        x1 = waypoints[indices]
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=dev)
        xt = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        velocity = net(xt, tau * args.flow_steps, condition[indices])
        loss = nn.functional.mse_loss(velocity, x1 - x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for key, value in net.state_dict().items():
                ema[key].mul_(0.999).add_(value.detach(), alpha=0.001)
        if step == 1 or step % 5000 == 0:
            print(
                json.dumps(
                    {
                        "event": "waypoint_flow_train",
                        "step": step,
                        "loss": round(float(loss.detach()), 6),
                    }
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": ema,
            "state_dict": net.state_dict(),
            "config": {
                "waypoint_dim": spec.goal_dim,
                "cond_dim": int(condition.shape[1]),
                "hidden": args.hidden,
                "n_blocks": args.n_blocks,
                "flow_steps": args.flow_steps,
                "goal_dim": spec.goal_dim,
                "abstract_dim": int(hwm_cfg["abstract_dim"]),
                "stride": stride,
                "max_future_hops": args.max_future_hops,
                "condition_desired_goal": bool(args.condition_desired_goal),
                "architecture": "direct_waypoint_flow_v1",
                "hwm": str(args.hwm),
                "model_path": str(args.model_path),
                "episodes_npz": str(args.episodes_npz),
            },
        },
        args.out,
    )
    print(
        json.dumps({"event": "waypoint_flow_saved", "path": str(args.out)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
