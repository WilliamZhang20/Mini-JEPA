from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsSpec:
    obs_dim: int
    goal_dim: int
    state_dim: int
    action_dim: int
    is_goal_env: bool = True


def register_robotics_envs() -> None:
    try:
        import gymnasium as gym
        import gymnasium_robotics
    except ImportError as exc:
        raise ImportError(
            "Gymnasium Robotics is required. Install with "
            "`pip install 'gymnasium-robotics[mujoco]'` inside your conda env."
        ) from exc

    gym.register_envs(gymnasium_robotics)


def make_env(
    env_id: str,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
):
    import gymnasium as gym

    register_robotics_envs()
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def flatten_obs(obs) -> np.ndarray:
    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [
            np.asarray(obs["observation"], dtype=np.float32).reshape(-1),
            np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1),
            np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def obs_spec_from_env(env) -> ObsSpec:
    obs_space = env.observation_space
    if not hasattr(obs_space, "spaces"):
        obs_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        return ObsSpec(
            obs_dim=obs_dim,
            goal_dim=0,
            state_dim=obs_dim,
            action_dim=action_dim,
            is_goal_env=False,
        )
    obs_dim = int(np.prod(obs_space["observation"].shape))
    goal_dim = int(np.prod(obs_space["achieved_goal"].shape))
    action_dim = int(np.prod(env.action_space.shape))
    return ObsSpec(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        state_dim=obs_dim + 2 * goal_dim,
        action_dim=action_dim,
        is_goal_env=True,
    )


def goal_reach_distance(state: np.ndarray, spec: ObsSpec) -> float:
    if not spec.is_goal_env or spec.goal_dim == 0:
        return 0.0
    achieved = state[spec.obs_dim : spec.obs_dim + spec.goal_dim]
    desired = state[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
    return float(np.linalg.norm(achieved - desired))


def goal_state_from_state(state: np.ndarray, spec: ObsSpec) -> np.ndarray:
    if not spec.is_goal_env or spec.goal_dim == 0:
        return np.array(state, copy=True).astype(np.float32)
    goal_state = np.array(state, copy=True)
    achieved = np.array(goal_state[spec.obs_dim : spec.obs_dim + spec.goal_dim], copy=True)
    desired = goal_state[
        spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim
    ]
    goal_state[spec.obs_dim : spec.obs_dim + spec.goal_dim] = desired
    if spec.obs_dim >= spec.goal_dim:
        obs = goal_state[: spec.obs_dim]
        best_start = 0
        best_error = float("inf")
        for start in range(spec.obs_dim - spec.goal_dim + 1):
            error = float(np.linalg.norm(obs[start : start + spec.goal_dim] - achieved))
            if error < best_error:
                best_error = error
                best_start = start
        goal_state[best_start : best_start + spec.goal_dim] = desired
    return goal_state.astype(np.float32)
