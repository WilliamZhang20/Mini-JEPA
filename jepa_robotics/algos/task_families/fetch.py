"""Reusable Fetch-family conditioning and trajectory utilities."""
from __future__ import annotations

import numpy as np

from jepa_robotics.data import Episode
from jepa_robotics.envs import flatten_obs, obs_spec_from_env


def geometry_features(state: np.ndarray, target_state: np.ndarray, spec) -> np.ndarray:
    obs = state[: spec.obs_dim]
    target_obs = target_state[: spec.obs_dim]
    grip = obs[:3]
    obj = state[spec.obs_dim : spec.obs_dim + spec.goal_dim]
    goal = state[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
    target_grip = target_obs[:3]
    target_obj = target_state[spec.obs_dim : spec.obs_dim + spec.goal_dim]
    return np.concatenate([
        grip - obj, obj - goal, grip - goal, target_obj - obj,
        target_grip - grip,
        np.array([np.linalg.norm(obj - goal), np.linalg.norm(target_obj - obj),
                  np.linalg.norm(target_obj - goal)], dtype=np.float32),
    ]).astype(np.float32)


def collect_policy_episodes(env, policy, *, num_steps: int, seed: int, log_every: int = 0):
    spec = obs_spec_from_env(env)
    episodes: list[Episode] = []
    total_steps = episode_idx = 0
    while total_steps < num_steps:
        obs, _ = env.reset(seed=seed + episode_idx)
        if hasattr(policy, "reset"):
            policy.reset()
        states, actions = [flatten_obs(obs)], []
        terminated = truncated = False
        while not (terminated or truncated) and total_steps < num_steps:
            action = np.clip(policy.act(obs, env), env.action_space.low, env.action_space.high).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            states.append(flatten_obs(obs))
            total_steps += 1
            if log_every > 0 and total_steps % log_every == 0:
                print(f'{{"event": "collect", "steps": {total_steps}, "target_steps": {num_steps}}}', flush=True)
        if actions:
            episodes.append(Episode(np.stack(states).astype(np.float32), np.stack(actions).astype(np.float32)))
        episode_idx += 1
    return episodes, spec
