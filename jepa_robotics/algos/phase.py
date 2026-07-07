"""Self-supervised phase utilities for hierarchical JEPA control."""
from __future__ import annotations

import numpy as np
import torch


def phase_id(t: int, horizon_steps: int, n_phases: int) -> int:
    progress = float(t) / float(max(1, horizon_steps))
    return int(np.clip(np.floor(progress * n_phases), 0, n_phases - 1))


def phase_features(
    cur_phase: int,
    target_phase: int,
    cur_progress: float,
    target_progress: float,
    n_phases: int,
    device,
) -> torch.Tensor:
    feat = torch.zeros(2 * n_phases + 2, dtype=torch.float32, device=device)
    feat[cur_phase] = 1.0
    feat[n_phases + target_phase] = 1.0
    feat[-2] = float(cur_progress)
    feat[-1] = float(target_progress)
    return feat


def batch_phase_features(
    cur_phase: np.ndarray,
    target_phase: np.ndarray,
    cur_progress: np.ndarray,
    target_progress: np.ndarray,
    n_phases: int,
) -> np.ndarray:
    feat = np.zeros((len(cur_phase), 2 * n_phases + 2), dtype=np.float32)
    rows = np.arange(len(cur_phase))
    feat[rows, cur_phase.astype(np.int64)] = 1.0
    feat[rows, n_phases + target_phase.astype(np.int64)] = 1.0
    feat[:, -2] = cur_progress.astype(np.float32)
    feat[:, -1] = target_progress.astype(np.float32)
    return feat


class PhaseFutureIndex:
    """Nearest-future lookup constrained by self-supervised temporal phase."""

    def __init__(self, ckpt: dict, normalizer) -> None:
        self.states = np.asarray(ckpt["bank_states"], dtype=np.float32)
        self.futures = np.asarray(ckpt["bank_futures"], dtype=np.float32)
        self.phase = np.asarray(ckpt["bank_phase"], dtype=np.int64)
        self.target_phase = np.asarray(ckpt["bank_target_phase"], dtype=np.int64)
        self.progress = np.asarray(ckpt["bank_progress"], dtype=np.float32)
        self.target_progress = np.asarray(ckpt["bank_target_progress"], dtype=np.float32)
        self.norm_states = normalizer.encode(self.states).astype(np.float32)
        self.by_phase = {int(p): np.flatnonzero(self.phase == p) for p in np.unique(self.phase)}
        try:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(self.norm_states)
            self.phase_trees = {p: cKDTree(self.norm_states[idx]) for p, idx in self.by_phase.items() if len(idx)}
        except Exception:
            self.tree = None
            self.phase_trees = {}

    def estimate_phase(self, state: np.ndarray, normalizer, last_phase: int, monotone: bool) -> int:
        x = normalizer.encode(state).astype(np.float32)
        if self.tree is not None:
            _dist, idx = self.tree.query(x, k=1)
            phase = int(self.phase[int(idx)])
        else:
            idx = int(np.argmin(np.linalg.norm(self.norm_states - x[None], axis=1)))
            phase = int(self.phase[idx])
        return max(last_phase, phase) if monotone else phase

    def query(self, state: np.ndarray, normalizer, phase: int, window: int) -> tuple[np.ndarray, int, int, float, float]:
        x = normalizer.encode(state).astype(np.float32)
        phases = [p for p in range(phase - window, phase + window + 1) if p in self.by_phase]
        if not phases:
            phases = list(self.by_phase)
        best_global = None
        best_dist = float("inf")
        for p in phases:
            idxs = self.by_phase[p]
            if len(idxs) == 0:
                continue
            if p in self.phase_trees:
                dist, local = self.phase_trees[p].query(x, k=1)
                gi = int(idxs[int(local)])
                dist = float(dist)
            else:
                local_d = np.linalg.norm(self.norm_states[idxs] - x[None], axis=1)
                li = int(np.argmin(local_d))
                gi = int(idxs[li])
                dist = float(local_d[li])
            if dist < best_dist:
                best_dist = dist
                best_global = gi
        if best_global is None:
            best_global = int(np.argmin(np.linalg.norm(self.norm_states - x[None], axis=1)))
        return (
            self.futures[best_global],
            int(self.phase[best_global]),
            int(self.target_phase[best_global]),
            float(self.progress[best_global]),
            float(self.target_progress[best_global]),
        )
