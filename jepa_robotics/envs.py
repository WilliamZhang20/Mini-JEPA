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
    canonical_task: str | None = None,
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
        # Standard D4RL kitchen target set = the 4 tasks the complete-v2 demos
        # actually finish (microwave/kettle/light switch/slide cabinet); NOT bottom
        # burner (no demo completes it), which made full-4 near-impossible.
        kwargs.setdefault("tasks_to_complete",
                          ["microwave", "kettle", "light switch", "slide cabinet"])
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
    # Roadmap B: unify reach/push/pick into one observation space so a single
    # JEPA model + policy can serve all three Fetch skills.
    if canonical_task is not None:
        env = CanonicalFetchWrapper(env, canonical_task)
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


# ---------------------------------------------------------------------------
# Roadmap B: one world model + one policy for reach + push + pick-and-place.
#
# The three Fetch tasks share the 4-D action space and goal-conditioned
# structure but differ in observation width (reach: obs=10 -> state=16;
# push/pick: obs=25 -> state=31). The canonical adapter maps every Fetch env
# into ONE fixed-width "superset" observation so a single encoder can serve all
# skills. The layout is the 25-D push/pick observation (object fields zero-filled
# for reach, whose gripper IS its achieved_goal), plus an ``object_present`` flag
# and a 3-way task one-hot. Keeping the gripper/object fields at the *same*
# indices across tasks (rather than naive zero-padding) is what lets the shared
# encoder learn consistent semantics.
# ---------------------------------------------------------------------------

# Canonical observation = [25-D superset Fetch obs] + [object_present] + [task one-hot(3)]
CANONICAL_FETCH_OBS_DIM = 25 + 1 + 3   # = 29
CANONICAL_FETCH_GOAL_DIM = 3
CANONICAL_FETCH_STATE_DIM = CANONICAL_FETCH_OBS_DIM + 2 * CANONICAL_FETCH_GOAL_DIM  # = 35
# One-hot order; index into the trailing flags (object_present is flag 0).
CANONICAL_FETCH_TASK_ORDER = ("reach", "push", "pick")
# Index of ``object_present`` within the flattened canonical state (== obs index,
# since obs starts at 0). Used by the manip MPC score to gate grasp/reach terms.
CANONICAL_OBJECT_PRESENT_IDX = 25

# Per-task wiring for the unified controller: short name -> (env_id, scripted
# controller, has_object). ``max_episode_steps`` matches the per-task specialists.
FETCH_MULTI_SUBTASKS = (
    {"name": "reach", "env_id": "FetchReach-v4", "controller": "reach",
     "has_object": False, "max_episode_steps": 50},
    {"name": "push", "env_id": "FetchPush-v4", "controller": "push",
     "has_object": True, "max_episode_steps": 100},
    {"name": "pick", "env_id": "FetchPickAndPlace-v4", "controller": "pick_place",
     "has_object": True, "max_episode_steps": 100},
)


def canonical_fetch_spec() -> "ObsSpec":
    """The single ObsSpec shared by reach/push/pick under the canonical adapter."""
    return ObsSpec(
        obs_dim=CANONICAL_FETCH_OBS_DIM,
        goal_dim=CANONICAL_FETCH_GOAL_DIM,
        state_dim=CANONICAL_FETCH_STATE_DIM,
        action_dim=4,
        is_goal_env=True,
    )


def _make_canonical_fetch_wrapper():
    import gymnasium as gym
    from gymnasium import spaces

    class _CanonicalFetch(gym.ObservationWrapper):
        """Remap a Fetch env's observation into the unified canonical layout.

        ``canonical_task`` is one of ``reach`` / ``push`` / ``pick`` and fixes the
        task one-hot and ``object_present`` flag. The achieved/desired goals are
        already 3-D in every Fetch env and pass through unchanged.
        """

        def __init__(self, env, canonical_task: str):
            super().__init__(env)
            if canonical_task not in CANONICAL_FETCH_TASK_ORDER:
                raise ValueError(f"Unknown canonical_task {canonical_task!r}")
            self.canonical_task = canonical_task
            self._onehot = CANONICAL_FETCH_TASK_ORDER.index(canonical_task)
            self._object_present = 0.0 if canonical_task == "reach" else 1.0
            spaces_dict = dict(env.observation_space.spaces)
            spaces_dict["observation"] = spaces.Box(
                -np.inf, np.inf, shape=(CANONICAL_FETCH_OBS_DIM,), dtype=np.float32
            )
            self.observation_space = spaces.Dict(spaces_dict)

        def observation(self, obs):
            o = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
            canon = np.zeros(25, dtype=np.float32)
            if self.canonical_task == "reach":
                # FetchReach obs (10-D): grip_pos[0:3], gripper_state[3:5],
                # grip_velp[5:8], gripper_vel[8:10]. The object fields stay zero;
                # object_pos is set to the gripper so achieved_goal == object slot.
                canon[0:3] = o[0:3]      # grip_pos
                canon[3:6] = o[0:3]      # object_pos := gripper (no real object)
                canon[9:11] = o[3:5]     # gripper_state (fingers)
                canon[20:23] = o[5:8]    # grip_velp
                canon[23:25] = o[8:10]   # gripper_vel
            else:
                # FetchPush/PickAndPlace obs is already the 25-D superset layout.
                canon[:25] = o[:25]
            flags = np.zeros(4, dtype=np.float32)
            flags[0] = self._object_present
            flags[1 + self._onehot] = 1.0
            new_obs = dict(obs)
            new_obs["observation"] = np.concatenate([canon, flags]).astype(np.float32)
            return new_obs

    return _CanonicalFetch


CanonicalFetchWrapper = _make_canonical_fetch_wrapper()


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
