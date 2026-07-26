"""Object-centric SE(3) utilities for dexterous manipulation controllers."""
from __future__ import annotations

import numpy as np
import torch


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo quaternions in ``(w, x, y, z)`` order."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float32)
    bw, bx, by, bz = np.asarray(b, dtype=np.float32)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.concatenate([q[:1], -q[1:]]).astype(np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / (float(np.linalg.norm(q)) + 1e-9)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion to a three-dimensional rotation vector."""
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, max(float(q[0]), 1e-8))
    return (q[1:] * (angle / sin_half)).astype(np.float32)


def quat_exp(rotation: np.ndarray) -> np.ndarray:
    """Three-dimensional rotation vector to a unit quaternion."""
    rotation = np.asarray(rotation, dtype=np.float32)
    angle = float(np.linalg.norm(rotation))
    if angle < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = rotation / angle
    return np.concatenate(
        [
            np.asarray([np.cos(0.5 * angle)], dtype=np.float32),
            axis * np.sin(0.5 * angle),
        ]
    ).astype(np.float32)


def relative_pose_features(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    position_scale: float = 0.05,
) -> np.ndarray:
    """Return scaled xyz error + shortest-arc SO(3) log error."""
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    q_error = quat_mul(
        quat_normalize(target_pose[3:]),
        quat_conjugate(quat_normalize(current_pose[3:])),
    )
    return np.concatenate(
        [
            (target_pose[:3] - current_pose[:3]) / float(position_scale),
            quat_log(q_error),
        ]
    ).astype(np.float32)


def step_pose(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    max_position_step: float,
    max_rotation_step: float,
) -> np.ndarray:
    """Take one bounded translation/SO(3) step toward a full object-pose goal."""
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    delta_position = target_pose[:3] - current_pose[:3]
    distance = float(np.linalg.norm(delta_position))
    if distance > max_position_step:
        delta_position *= float(max_position_step) / (distance + 1e-9)
    q_current = quat_normalize(current_pose[3:])
    q_error = quat_mul(quat_normalize(target_pose[3:]), quat_conjugate(q_current))
    rotation = quat_log(q_error)
    angle = float(np.linalg.norm(rotation))
    if angle > max_rotation_step:
        rotation *= float(max_rotation_step) / (angle + 1e-9)
    q_step = quat_mul(quat_exp(rotation), q_current)
    return np.concatenate([current_pose[:3] + delta_position, quat_normalize(q_step)]).astype(
        np.float32
    )


def pose_cost(achieved: np.ndarray, desired: np.ndarray) -> float:
    """HandManipulate's dense geometric surrogate: ``10*xyz + SO(3)``."""
    features = relative_pose_features(achieved, desired)
    return float(
        10.0 * np.linalg.norm(np.asarray(achieved)[:3] - np.asarray(desired)[:3])
        + np.linalg.norm(features[3:])
    )


class DexterousTransitionMemory:
    """Retrieve state-compatible action chunks from reward-free experience.

    Retrieval is deliberately hierarchical: first find transitions whose
    observed SE(3) displacement matches the requested local displacement, then
    rank that pool by standardized JEPA-latent distance to the live hand state.
    This avoids asking a generative prior to average across incompatible
    contact modes while still grounding every proposal in a real transition.
    """

    def __init__(
        self,
        latents: torch.Tensor,
        pose_deltas: torch.Tensor,
        action_chunks: torch.Tensor,
    ) -> None:
        if not (len(latents) == len(pose_deltas) == len(action_chunks)):
            raise ValueError("transition-memory arrays must have equal length")
        self.latents = latents
        self.pose_deltas = pose_deltas
        self.action_chunks = action_chunks
        self.latent_mean = latents.mean(0)
        self.latent_std = latents.std(0).clamp_min(1e-3)
        self.standardized_latents = (
            latents - self.latent_mean
        ) / self.latent_std

    @torch.no_grad()
    def query(
        self,
        latent: torch.Tensor,
        pose_delta: torch.Tensor,
        *,
        candidates: int,
        pose_pool: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return chunks and indices matching pose first, contact state second."""
        pose_pool = min(max(candidates, pose_pool), len(self.pose_deltas))
        pose_error = torch.square(
            self.pose_deltas - pose_delta.reshape(1, -1)
        ).sum(-1)
        pool = torch.topk(
            pose_error, k=pose_pool, largest=False, sorted=False
        ).indices
        query = (latent.reshape(-1) - self.latent_mean) / self.latent_std
        latent_error = torch.square(
            self.standardized_latents[pool] - query
        ).mean(-1)
        selected = pool[
            torch.topk(
                latent_error,
                k=min(candidates, pose_pool),
                largest=False,
                sorted=True,
            ).indices
        ]
        return self.action_chunks[selected], selected
