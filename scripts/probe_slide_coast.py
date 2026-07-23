"""Measure the slide world model's open-loop coast-prediction accuracy.

Gate for the open-loop strike-MPC idea: roll the JEPA latent forward open-loop using
the *actual* actions taken in real strike episodes, decode the predicted puck position
via the state probe, and compare to ground truth at each horizon. If the model cannot
track the puck through the post-contact coast, open-loop MPC is hopeless and the world
model must be retrained with longer horizons first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact, ScriptedGoalPolicy
from jepa_robotics.tasks import resolve_task


def collect_episodes(env_id, controller, max_steps, n_eps, seed, action_dim, gain):
    """Run the scripted strike expert and record (states, actions) per episode."""
    expert = ScriptedGoalPolicy(action_dim=action_dim, controller=controller, gain=gain)
    episodes = []
    for ep in range(n_eps):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        states, actions = [flatten_obs(obs)], []
        terminated = truncated = False
        while not (terminated or truncated):
            a = expert.act(obs, env)
            actions.append(np.asarray(a, dtype=np.float32))
            obs, _, terminated, truncated, _ = env.step(a)
            states.append(flatten_obs(obs))
        env.close()
        episodes.append((np.asarray(states), np.asarray(actions)))
    return episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--model-path", type=Path,
                   default=Path("runs/fetch_slide/checkpoints/slide_jepa_beef_scratch_20260613_model.pt"))
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--max-horizon", type=int, default=40)
    p.add_argument("--gain", type=float, default=12.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    model, normalizer, spec, config = load_jepa_artifact(args.model_path, device)
    model.eval()
    model_max_h = int(config["max_horizon"])
    H = min(args.max_horizon, model_max_h)
    print(f'{{"model_max_horizon": {model_max_h}, "probe_horizon": {H}, '
          f'"obs_dim": {spec.obs_dim}, "goal_dim": {spec.goal_dim}}}', flush=True)

    episodes = collect_episodes(task.env_id, task.controller, task.max_episode_steps,
                                args.episodes, args.seed, spec.action_dim, args.gain)

    # puck (achieved_goal) lives at [obs_dim : obs_dim+goal_dim] of the flattened state
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim

    # accumulate per-horizon errors
    puck_err = {h: [] for h in range(1, H + 1)}
    full_err = {h: [] for h in range(1, H + 1)}
    # naive baseline: assume puck never moves from t0 (tests whether model beats "static")
    static_err = {h: [] for h in range(1, H + 1)}

    for states, actions in episodes:
        T = len(actions)
        # start rollout from each timestep where a full horizon-H window of real actions exists
        for t0 in range(0, max(1, T - H)):
            raw0 = states[t0]
            z0 = model.encode(
                torch.from_numpy(normalizer.encode(raw0)).unsqueeze(0).to(device)
            )
            act_win = torch.from_numpy(actions[t0:t0 + H]).unsqueeze(0).to(device)
            with torch.no_grad():
                traj_z = model.predict_rollout(z0, act_win, H)  # [1, H, latent]
                pred_states = normalizer.decode_tensor(model.state_probe(traj_z))[0].cpu().numpy()
            puck0 = states[t0][gs:ge]
            for h in range(1, H + 1):
                gt = states[t0 + h]
                pred = pred_states[h - 1]
                puck_err[h].append(np.linalg.norm(pred[gs:ge] - gt[gs:ge]))
                full_err[h].append(np.linalg.norm(pred - gt))
                static_err[h].append(np.linalg.norm(puck0 - gt[gs:ge]))

    print("horizon  puck_mae   full_mae   static_baseline  model/static")
    for h in [1, 2, 4, 8, 12, 16, 20, 24, 30, 40]:
        if h > H:
            continue
        pm = float(np.mean(puck_err[h]))
        fm = float(np.mean(full_err[h]))
        sm = float(np.mean(static_err[h]))
        ratio = pm / sm if sm > 1e-9 else float("nan")
        print(f"{h:>5}   {pm:8.4f}   {fm:8.4f}   {sm:14.4f}   {ratio:8.3f}", flush=True)


if __name__ == "__main__":
    main()
