from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .envs import ObsSpec, flatten_obs, goal_reach_distance, obs_spec_from_env


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def decode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std


def scripted_reach_action(obs: dict[str, np.ndarray], action_dim: int, gain: float) -> np.ndarray:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)
    action = np.zeros(action_dim, dtype=np.float32)
    action[: min(3, action_dim)] = gain * (desired[: min(3, action_dim)] - achieved[: min(3, action_dim)])
    return action


def unit_vector(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def scripted_pick_place_action(obs: dict[str, np.ndarray], action_dim: int, gain: float) -> np.ndarray:
    """Phase-based grasp-and-place controller.

    Phases are detected purely from geometry: align above the object (fingers
    open), descend, close the fingers, then carry the grasped object to the
    desired goal. The previous version mis-detected the grasp (the fingers
    settle near 0.048 when closed *around* an object, never below the old 0.046
    threshold), so it never lifted; this version keys off ``fingers_open`` and
    3D proximity instead and reliably solves FetchPickAndPlace.
    """
    observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    gripper = observation[:3]
    obj = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    goal = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)
    finger = float(observation[9] + observation[10])  # ~0.10 open, ~0.048 closed on object
    action = np.zeros(action_dim, dtype=np.float32)

    xy = float(np.linalg.norm(gripper[:2] - obj[:2]))
    d3 = float(np.linalg.norm(gripper - obj))
    fingers_open = finger > 0.07
    at_object = d3 < 0.055
    grasped = at_object and not fingers_open

    if not at_object and xy > 0.03:
        target = obj + np.array([0.0, 0.0, 0.06], dtype=np.float32)
        grip = 1.0
    elif not at_object:
        target = obj.copy()
        grip = 1.0
    elif not grasped:
        target = obj.copy()
        grip = -1.0
    else:
        target = goal.copy()
        grip = -1.0

    action[: min(3, action_dim)] = gain * (target[: min(3, action_dim)] - gripper[: min(3, action_dim)])
    if action_dim >= 4:
        action[3] = grip
    return action


def scripted_push_action(obs: dict[str, np.ndarray], action_dim: int, gain: float) -> np.ndarray:
    """Approach-from-above push controller.

    The old controller drove straight to a stand-off point behind the object,
    which meant it plowed *through* the object (often from the goal side) and
    knocked it the wrong way. This version lifts clear of the object, moves over
    the stand-off point on the far side from the goal, descends, then pushes
    toward the goal, easing off and retracting once the object is close so it
    does not overshoot.
    """
    observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    gripper = observation[:3]
    obj = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    goal = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)
    action = np.zeros(action_dim, dtype=np.float32)

    to_goal = goal[:2] - obj[:2]
    dist = float(np.linalg.norm(to_goal))
    d = unit_vector(to_goal) if dist > 1e-6 else np.zeros(2, dtype=np.float32)
    behind = obj[:2] - 0.075 * d
    behind_xy_err = float(np.linalg.norm(gripper[:2] - behind))
    obj_z = float(obj[2])

    if dist < 0.03:
        target = np.array([gripper[0], gripper[1], obj_z + 0.12], dtype=np.float32)
    elif behind_xy_err > 0.03 and gripper[2] < obj_z + 0.07:
        target = np.array([gripper[0], gripper[1], obj_z + 0.1], dtype=np.float32)
    elif behind_xy_err > 0.03:
        target = np.array([behind[0], behind[1], obj_z + 0.1], dtype=np.float32)
    elif gripper[2] > obj_z + 0.02:
        target = np.array([behind[0], behind[1], obj_z], dtype=np.float32)
    else:
        step = min(0.08, 0.6 * dist)
        target_xy = obj[:2] + step * d
        target = np.array([target_xy[0], target_xy[1], obj_z], dtype=np.float32)

    action[: min(3, action_dim)] = gain * (target[: min(3, action_dim)] - gripper[: min(3, action_dim)])
    if action_dim >= 4:
        action[3] = -1.0  # keep the gripper closed for pushing
    return action


def scripted_action(
    obs,
    action_dim: int,
    gain: float,
    controller: str,
    env,
    rng: np.random.Generator,
) -> np.ndarray:
    if not isinstance(obs, dict):
        return env.action_space.sample().astype(np.float32)
    if controller == "pick_place":
        action = scripted_pick_place_action(obs, action_dim, gain)
    elif controller == "push":
        action = scripted_push_action(obs, action_dim, gain)
    else:
        action = scripted_reach_action(obs, action_dim, gain)
    return action.astype(np.float32)


def collect_episodes(
    env,
    *,
    num_steps: int,
    seed: int,
    scripted_fraction: float,
    controller_gain: float,
    action_noise: float,
    controller: str = "reach",
    log_every: int = 0,
) -> tuple[list[Episode], ObsSpec]:
    spec = obs_spec_from_env(env)
    episodes: list[Episode] = []
    total_steps = 0
    episode_idx = 0
    rng = np.random.default_rng(seed)

    while total_steps < num_steps:
        obs, _ = env.reset(seed=seed + episode_idx)
        states = [flatten_obs(obs)]
        actions = []
        terminated = truncated = False

        while not (terminated or truncated) and total_steps < num_steps:
            if rng.random() < scripted_fraction:
                action = scripted_action(obs, spec.action_dim, controller_gain, controller, env, rng)
                action += rng.normal(0.0, action_noise, size=spec.action_dim).astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                action = env.action_space.sample().astype(np.float32)

            next_obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action.astype(np.float32))
            states.append(flatten_obs(next_obs))
            obs = next_obs
            total_steps += 1
            if log_every > 0 and total_steps % log_every == 0:
                print(
                    f'{{"event": "collect", "steps": {total_steps}, "target_steps": {num_steps}}}',
                    flush=True,
                )

        if actions:
            episodes.append(
                Episode(
                    states=np.stack(states).astype(np.float32),
                    actions=np.stack(actions).astype(np.float32),
                )
            )
        episode_idx += 1

    return episodes, spec


def fit_normalizer(episodes: Iterable[Episode], eps: float = 1e-6) -> Normalizer:
    states = np.concatenate([episode.states for episode in episodes], axis=0)
    return Normalizer(
        mean=states.mean(axis=0).astype(np.float32),
        std=(states.std(axis=0) + eps).astype(np.float32),
    )


class JEPATrajectoryDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        normalizer: Normalizer,
        spec: ObsSpec,
        horizons: list[int],
    ) -> None:
        self.episodes = episodes
        self.normalizer = normalizer
        self.spec = spec
        self.horizons = sorted(horizons)
        self.max_horizon = max(self.horizons)
        self.index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(episodes):
            valid = len(episode.actions) - self.max_horizon + 1
            self.index.extend((episode_idx, t) for t in range(max(0, valid)))
        if not self.index:
            raise ValueError("No trajectory windows are long enough for the requested horizons.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, t = self.index[idx]
        episode = self.episodes[episode_idx]
        state = episode.states[t]
        action_seq = episode.actions[t : t + self.max_horizon]
        future_states = np.stack([episode.states[t + h] for h in self.horizons]).astype(np.float32)
        return {
            "state": torch.from_numpy(self.normalizer.encode(state)),
            "raw_state": torch.from_numpy(state),
            "actions": torch.from_numpy(action_seq.astype(np.float32)),
            "future_states": torch.from_numpy(self.normalizer.encode(future_states)),
            "raw_future_states": torch.from_numpy(future_states),
            "distance": torch.tensor(goal_reach_distance(state, self.spec), dtype=torch.float32),
        }
