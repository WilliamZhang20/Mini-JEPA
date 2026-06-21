from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsSpec:
    """Dimensions of an environment's observation, goal, flattened state, and action spaces."""

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
    width: int | None = None,
    height: int | None = None,
):
    import gymnasium as gym

    register_robotics_envs()
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    # MuJoCo robotics envs render at 480x480 by default; width/height raise the
    # offscreen framebuffer resolution for crisper recordings.
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    # PointMaze/AntMaze default to a *continuing* task (the goal resamples after
    # every reach and the episode never terminates), which is wrong for our
    # goal-reaching HER setup. Force one fixed goal per episode that terminates
    # on success, matching the Fetch goal envs.
    if "PointMaze" in env_id or "AntMaze" in env_id:
        kwargs.setdefault("continuing_task", False)
        kwargs.setdefault("reset_target", False)
    # FrankaKitchen's default goal is all 7 possible subtasks; the D4RL demos
    # target a fixed 4-task set, so pin the eval goal to match the data.
    if "Kitchen" in env_id:
        kwargs.setdefault("tasks_to_complete",
                          ["microwave", "kettle", "bottom burner", "light switch"])
    env = gym.make(env_id, **kwargs)
    # Maze and Adroit envs report ``info["success"]``; the rest of the pipeline
    # (HER EvalCallback, eval scripts) reads ``info["is_success"]``. Alias it.
    # Harmless elsewhere (only added when ``success`` is present and
    # ``is_success`` is not), so it is safe to apply broadly.
    if "Maze" in env_id or "Adroit" in env_id:
        env = SuccessAliasWrapper(env)
    # FrankaKitchen has a Dict obs whose achieved/desired goals are nested Dicts
    # (per-task target maps, not coordinates). Treat it as a flat non-goal env:
    # expose only the 59-D ``observation`` and surface task-completion as success.
    if "Kitchen" in env_id:
        env = KitchenFlattenWrapper(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env


def _make_kitchen_wrapper():
    import gymnasium as gym
    from gymnasium import spaces

    class _KitchenFlatten(gym.ObservationWrapper):
        """Expose FrankaKitchen as a flat 59-D observation env and report
        ``info['is_success']`` = (all required subtasks completed this episode)."""

        def __init__(self, env):
            super().__init__(env)
            self.observation_space = env.observation_space["observation"]
            # Tasks required this episode are fixed at reset (the env pops them
            # from tasks_to_complete as they complete, so snapshot the goal set).
            self._goal_tasks = list(getattr(env.unwrapped, "tasks_to_complete", []) or [])
            self._n_tasks = len(self._goal_tasks) or 4
            self._done_tasks: set = set()

        def observation(self, obs):
            return np.asarray(obs["observation"], dtype=np.float32)

        def reset(self, **kwargs):
            self._done_tasks = set()
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._done_tasks |= set(info.get("step_task_completions", []))
            n_req = len(info.get("tasks_to_complete", [])) + len(self._done_tasks)
            info["tasks_done"] = len(self._done_tasks)
            info["is_success"] = float(len(self._done_tasks) >= max(1, self._n_tasks))
            return self.observation(obs), reward, terminated, truncated, info

    return _KitchenFlatten


KitchenFlattenWrapper = _make_kitchen_wrapper()


def _make_success_alias_wrapper():
    import gymnasium as gym

    class _SuccessAlias(gym.Wrapper):
        """Expose ``info['is_success']`` (mirroring ``info['success']``) for maze envs."""

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if "is_success" not in info and "success" in info:
                info["is_success"] = float(info["success"])
            return obs, reward, terminated, truncated, info

    return _SuccessAlias


SuccessAliasWrapper = _make_success_alias_wrapper()


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
