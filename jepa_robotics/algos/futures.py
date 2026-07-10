"""Future-target selection strategies for SSL latent planning."""
from __future__ import annotations

import numpy as np


class DemoLockedFutureIndex:
    """Trajectory-locked future lookup for receding-horizon SSL planning.

    Nearest-state future retrieval is memoryless: consecutive queries can jump
    between demos, so the requested futures are not causally consecutive and
    the action prior is asked to chase a different trajectory every replan.
    This index locks onto a single demo episode when an eval episode starts
    (nearest initial state) and re-localizes the current state within that
    demo on every query, so the future target advances along one executable
    trajectory. If execution deviates -- for example the ball is dropped --
    re-localization falls back to the demo phase whose geometry matches the
    current state, which restarts the reach/grasp segment instead of asking
    for an unreachable transport future.
    """

    def __init__(
        self,
        episodes: list[np.ndarray],
        normalizer,
        *,
        horizon: int,
        locality_weight: float = 0.0,
        predicate_dims: tuple[int, int] | None = None,
        predicate_threshold: float = 0.06,
        geom_dims: tuple[int, int] | None = None,
        geom_weight: float = 1.0,
        relock_margin: float = 0.0,
    ) -> None:
        self.episodes = [np.asarray(ep, dtype=np.float32) for ep in episodes if len(ep) > 1]
        if not self.episodes:
            raise ValueError("DemoLockedFutureIndex needs at least one episode")
        self.norm_episodes = [normalizer.encode(ep).astype(np.float32) for ep in self.episodes]
        self.starts = np.stack([ne[0] for ne in self.norm_episodes])
        state_dim = self.starts.shape[1]
        self.dim_weights = np.ones(state_dim, dtype=np.float32)
        self.geom_dims = geom_dims
        if geom_dims is not None and geom_weight != 1.0:
            lo, hi = geom_dims
            # Upweight the task-geometry dims (e.g. palm-ball / ball-target
            # vectors) so demo matching tracks grasp feasibility rather than
            # being dominated by joint-angle dims.
            self.dim_weights[lo:hi] = float(geom_weight)
        self.relock_margin = float(relock_margin)
        self.all_states: np.ndarray | None = None
        self.all_ep_ids: np.ndarray | None = None
        self.all_step_ids: np.ndarray | None = None
        if self.relock_margin > 0:
            self._build_flat_bank()
        self.horizon = int(horizon)
        self.locality_weight = float(locality_weight)
        self.predicate_dims = predicate_dims
        self.predicate_threshold = float(predicate_threshold)
        if predicate_dims is not None:
            lo, hi = predicate_dims
            self.first_satisfied = []
            for ep in self.episodes:
                held = np.linalg.norm(ep[:, lo:hi], axis=-1) < self.predicate_threshold
                self.first_satisfied.append(int(np.argmax(held)) if held.any() else len(ep) - 1)
        else:
            self.first_satisfied = None
        self.locked: int | None = None
        self.prev_idx = 0

    def _build_flat_bank(self) -> None:
        if self.all_states is None:
            self.all_states = np.concatenate([ne for ne in self.norm_episodes], axis=0)
            self.all_ep_ids = np.concatenate(
                [np.full(len(ne), i, dtype=np.int64) for i, ne in enumerate(self.norm_episodes)]
            )
            self.all_step_ids = np.concatenate(
                [np.arange(len(ne), dtype=np.int64) for ne in self.norm_episodes]
            )

    def reset(self) -> None:
        self.locked = None
        self.prev_idx = 0

    def query(self, state: np.ndarray, normalizer) -> np.ndarray:
        x = normalizer.encode(state).astype(np.float32)
        w = self.dim_weights[None]
        if self.locked is None:
            self.locked = int(np.argmin(np.linalg.norm((self.starts - x[None]) * w, axis=1)))
            self.prev_idx = 0
        ne = self.norm_episodes[self.locked]
        dist = np.linalg.norm((ne - x[None]) * w, axis=1)
        if self.relock_margin > 0:
            all_dist = np.linalg.norm((self.all_states - x[None]) * w, axis=1)
            best_global = int(np.argmin(all_dist))
            if all_dist[best_global] < float(dist.min()) - self.relock_margin and int(self.all_ep_ids[best_global]) != self.locked:
                self.locked = int(self.all_ep_ids[best_global])
                self.prev_idx = 0
                ne = self.norm_episodes[self.locked]
                dist = np.linalg.norm((ne - x[None]) * w, axis=1)
        if self.locality_weight > 0:
            steps = np.abs(np.arange(len(ne)) - (self.prev_idx + 1))
            dist = dist + self.locality_weight * steps / max(1, len(ne) - 1)
        idx = int(np.argmin(dist))
        self.prev_idx = idx
        ep = self.episodes[self.locked]
        target_idx = min(idx + self.horizon, len(ep) - 1)
        if self.predicate_dims is not None:
            # Advance-on-predicate gating: while the live state does not satisfy
            # the predicate (e.g. ball not yet possessed), do not request demo
            # futures beyond the frame where the demo first satisfies it. This
            # keeps the requested future causally reachable instead of asking
            # for transport before the grasp is secured.
            lo, hi = self.predicate_dims
            satisfied = float(np.linalg.norm(state[lo:hi])) < self.predicate_threshold
            if not satisfied:
                target_idx = min(target_idx, max(self.first_satisfied[self.locked], min(idx + 1, len(ep) - 1)))
        return ep[target_idx]

    def match_distance(self, state: np.ndarray, normalizer) -> float:
        """Weighted distance from ``state`` to the nearest state in ANY of this
        index's demo segments (non-mutating; does not touch the lock).

        Used by an order-agnostic high level to score how reachable this
        subtask is from the current live state, so the scheduler can pick the
        most-reachable uncompleted subtask instead of following a fixed order.
        Distance is computed on the geom slice only (the reachability-determining
        feature, e.g. arm pose) when one was configured, so object-state dims do
        not dominate the reachability judgement.
        """
        x = normalizer.encode(state).astype(np.float32)
        if self.geom_dims is not None:
            lo, hi = self.geom_dims
            xs = x[lo:hi][None]
            best = min(float(np.linalg.norm(ne[:, lo:hi] - xs, axis=1).min()) for ne in self.norm_episodes)
        else:
            w = self.dim_weights[None]
            best = min(float(np.linalg.norm((ne - x[None]) * w, axis=1).min()) for ne in self.norm_episodes)
        return best

    def query_topk(self, state: np.ndarray, normalizer, k: int) -> list[np.ndarray]:
        """Futures from the locked demo plus the best-matching other demos.

        Returns up to ``k`` future states, each from a distinct episode: the
        locked demo's future first (identical to ``query``), then futures from
        the episodes whose weighted nearest state best matches the live state.
        Every candidate is a real demonstrated future, so downstream chunk
        proposals stay deterministic per candidate - diversity comes from demo
        coverage, not sampling noise.
        """

        primary = self.query(state, normalizer)
        if k <= 1:
            return [primary]
        self._build_flat_bank()
        x = normalizer.encode(state).astype(np.float32)
        d = np.linalg.norm((self.all_states - x[None]) * self.dim_weights[None], axis=1)
        order = np.argsort(d)
        futures = [primary]
        seen = {self.locked}
        for flat_idx in order:
            ep_id = int(self.all_ep_ids[flat_idx])
            if ep_id in seen:
                continue
            seen.add(ep_id)
            ep = self.episodes[ep_id]
            futures.append(ep[min(int(self.all_step_ids[flat_idx]) + self.horizon, len(ep) - 1)])
            if len(futures) >= k:
                break
        return futures
