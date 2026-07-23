"""Event-conditioned macro world models for ballistic manipulation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def fetch_slide_ready(state: np.ndarray, obs_dim: int, goal_dim: int) -> bool:
    """Return whether the gripper is aligned behind the puck for one-way impact."""
    grip = state[:3]
    obj = state[obs_dim : obs_dim + goal_dim]
    goal = state[obs_dim + goal_dim : obs_dim + 2 * goal_dim]
    direction = goal[:2] - obj[:2]
    direction = direction / (float(np.linalg.norm(direction)) + 1e-6)
    rel = grip[:2] - obj[:2]
    along = float(np.dot(rel, direction))
    lateral = float(np.linalg.norm(rel - along * direction))
    low = grip[2] <= float(obj[2]) + 0.03
    return along < 0.0 and lateral < 0.025 and low


def slide_macro_action(
    macro: np.ndarray,
    state: np.ndarray,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
) -> np.ndarray:
    """Decode ``[dir_x, dir_y, amplitude, duration/max_duration]`` to impact action."""
    direction = np.asarray(macro[:2], dtype=np.float32)
    direction /= float(np.linalg.norm(direction)) + 1e-6
    action = np.zeros(action_dim, dtype=np.float32)
    action[:2] = float(macro[2]) * direction
    obj_z = float(state[obs_dim + 2])
    action[2] = float(np.clip(12.0 * (obj_z - state[2]), -1.0, 1.0))
    if action_dim >= 4:
        action[3] = -1.0
    return action


class BallisticHWM(nn.Module):
    """Predict the absorbing post-coast latent and endpoint from one strike macro.

    Unlike a fixed-step recurrent world model, this model advances between two
    events: a pre-impact aligned state and the stopped-puck state. Multiple heads
    expose epistemic disagreement for conservative macro selection.
    """

    def __init__(
        self,
        latent_dim: int,
        macro_dim: int = 4,
        hidden: int = 512,
        n_heads: int = 5,
        side_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.macro_dim = int(macro_dim)
        self.hidden = int(hidden)
        self.n_heads = int(n_heads)
        self.side_dim = int(side_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(latent_dim + macro_dim + side_dim),
            nn.Linear(latent_dim + macro_dim + side_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, latent_dim + 3) for _ in range(n_heads)]
        )

    def forward(
        self,
        z: torch.Tensor,
        macro: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.side_dim:
            if side is None:
                raise ValueError("BallisticHWM was trained with a state side channel")
            features = torch.cat([z, macro, side], dim=-1)
        else:
            features = torch.cat([z, macro], dim=-1)
        h = self.trunk(features)
        pred = torch.stack([head(h) for head in self.heads], dim=1)
        # Predict a latent residual but a physical puck displacement. The event
        # endpoint head is the high-level planning variable; the latent target
        # keeps this an HWM in the JEPA representation space.
        z_next = z[:, None, :] + pred[..., : self.latent_dim]
        displacement = pred[..., self.latent_dim :]
        return z_next, displacement


def _goal_frame(states: np.ndarray, obs_dim: int, goal_dim: int) -> tuple[np.ndarray, np.ndarray]:
    obj = states[..., obs_dim : obs_dim + goal_dim]
    goal = states[..., obs_dim + goal_dim : obs_dim + 2 * goal_dim]
    direction = goal[..., :2] - obj[..., :2]
    direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6)
    perpendicular = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
    return direction.astype(np.float32), perpendicular.astype(np.float32)


def world_to_goal_frame(
    vectors: np.ndarray, states: np.ndarray, obs_dim: int, goal_dim: int
) -> np.ndarray:
    """Rotate world-frame xyz vectors into goal-aligned coordinates."""
    direction, perpendicular = _goal_frame(states, obs_dim, goal_dim)
    vectors = np.asarray(vectors, np.float32)
    return np.stack(
        [
            np.sum(vectors[..., :2] * direction, axis=-1),
            np.sum(vectors[..., :2] * perpendicular, axis=-1),
            vectors[..., 2],
        ],
        axis=-1,
    ).astype(np.float32)


def goal_to_world_frame(
    vectors: np.ndarray, states: np.ndarray, obs_dim: int, goal_dim: int
) -> np.ndarray:
    """Rotate goal-aligned xyz vectors back into world coordinates."""
    direction, perpendicular = _goal_frame(states, obs_dim, goal_dim)
    vectors = np.asarray(vectors, np.float32)
    xy = vectors[..., :1] * direction + vectors[..., 1:2] * perpendicular
    return np.concatenate([xy, vectors[..., 2:3]], axis=-1).astype(np.float32)


def canonical_ballistic_features(
    states: np.ndarray,
    macros: np.ndarray,
    obs_dim: int,
    goal_dim: int,
) -> np.ndarray:
    """Goal-frame contact and velocity features for rotation-equivariant strikes."""
    states = np.asarray(states, np.float32)
    macros = np.asarray(macros, np.float32)
    direction, perpendicular = _goal_frame(states, obs_dim, goal_dim)
    obs = states[..., :obs_dim]
    grip = obs[..., :3]
    obj = states[..., obs_dim : obs_dim + goal_dim]
    goal = states[..., obs_dim + goal_dim : obs_dim + 2 * goal_dim]

    def project_xy(vector: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.sum(vector[..., :2] * direction, axis=-1),
                np.sum(vector[..., :2] * perpendicular, axis=-1),
            ],
            axis=-1,
        )

    grip_rel = project_xy(grip - obj)
    macro_dir = project_xy(np.concatenate(
        [macros[..., :2], np.zeros((*macros.shape[:-1], 1), np.float32)], axis=-1
    ))
    grip_vel = project_xy(obs[..., 20:23]) if obs_dim >= 23 else np.zeros_like(grip_rel)
    # Fetch stores object velocity relative to the gripper.
    object_vel_world = (
        obs[..., 14:17] + obs[..., 20:23]
        if obs_dim >= 23 else np.zeros((*states.shape[:-1], 3), np.float32)
    )
    object_vel = project_xy(object_vel_world)
    scalar = np.stack(
        [
            macros[..., 2],
            macros[..., 3],
            np.linalg.norm(goal[..., :2] - obj[..., :2], axis=-1),
            grip[..., 2] - obj[..., 2],
            obj[..., 2],
        ],
        axis=-1,
    )
    return np.concatenate(
        [grip_rel, macro_dir, grip_vel, object_vel, scalar], axis=-1
    ).astype(np.float32)


class EquivariantBallisticHWM(nn.Module):
    """Goal-frame event HWM with Fourier geometry and gated residual blocks.

    The endpoint head predicts displacement in a goal-aligned frame, removing
    global table orientation from the regression problem. JEPA latent dynamics
    remain in the original representation; only the physical planning head is
    constrained by the SE(2)-style inductive bias.
    """

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        hidden: int = 512,
        n_heads: int = 7,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        expanded = 5 * feature_dim
        self.input = nn.Sequential(
            nn.LayerNorm(latent_dim + expanded),
            nn.Linear(latent_dim + expanded, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden),
                "value": nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                       nn.Linear(2 * hidden, hidden)),
                "gate": nn.Linear(hidden, hidden),
            })
            for _ in range(n_blocks)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent_dim + 3))
            for _ in range(n_heads)
        ])

    def _expand(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                features,
                torch.sin(features),
                torch.cos(features),
                torch.sin(2.0 * features),
                torch.cos(2.0 * features),
            ],
            dim=-1,
        )

    def forward(
        self, z: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input(torch.cat([z, self._expand(features)], dim=-1))
        for block in self.blocks:
            normalized = block["norm"](h)
            h = h + torch.sigmoid(block["gate"](normalized)) * block["value"](normalized)
        pred = torch.stack([head(h) for head in self.heads], dim=1)
        z_next = z[:, None, :] + pred[..., : self.latent_dim]
        canonical_displacement = pred[..., self.latent_dim :]
        return z_next, canonical_displacement


def goal_relative_macro_candidates(
    object_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    angle_limit_deg: float = 45.0,
    angle_count: int = 31,
    amplitudes: int = 17,
    min_amplitude: float = 0.2,
    max_amplitude: float = 1.0,
    min_duration: int = 2,
    max_duration: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate low-dimensional strike macros around the live goal direction."""
    base = np.asarray(goal_xy, dtype=np.float32) - np.asarray(object_xy, dtype=np.float32)
    base_angle = float(np.arctan2(base[1], base[0]))
    offsets = np.deg2rad(np.linspace(-angle_limit_deg, angle_limit_deg, angle_count))
    amps = np.linspace(min_amplitude, max_amplitude, amplitudes)
    durations = np.arange(min_duration, max_duration + 1)
    macros, duration_steps = [], []
    for offset in offsets:
        direction = np.array([np.cos(base_angle + offset), np.sin(base_angle + offset)], np.float32)
        for amplitude in amps:
            for duration in durations:
                macros.append([direction[0], direction[1], amplitude, duration / max_duration])
                duration_steps.append(duration)
    return np.asarray(macros, np.float32), np.asarray(duration_steps, np.int64)
