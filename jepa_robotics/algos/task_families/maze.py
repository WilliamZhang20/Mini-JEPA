"""Dataset construction shared by maze walker trainers."""
from __future__ import annotations

import numpy as np


def build_chunks(episodes, spec, normalizer, chunk, future_goal_frac, rng):
    """Build padded action chunks with optional hindsight goal relabeling."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    states, chunks = [], []
    for episode in episodes:
        S, A = episode.states, episode.actions
        for t in range(len(A)):
            state = S[t].copy()
            if rng.random() < future_goal_frac:
                future = int(rng.integers(t + 1, len(S)))
                state[ds:de] = S[future, gs:ge]
            action_chunk = A[t : t + chunk]
            if len(action_chunk) < chunk:
                action_chunk = np.concatenate(
                    [action_chunk, np.repeat(action_chunk[-1:], chunk - len(action_chunk), axis=0)], axis=0
                )
            states.append(state)
            chunks.append(action_chunk)
    return normalizer.encode(np.asarray(states, np.float32)), np.asarray(chunks, np.float32)


def build_directed_chunks(
    episodes, spec, normalizer, chunk, rng, *, max_relabel_h=60, min_progress=0.15
):
    """Build goal-directed chunks that make measured progress toward a nearby future."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    states, chunks, progresses = [], [], []
    for episode in episodes:
        S, A = episode.states, episode.actions
        for t in range(len(A)):
            future = min(t + int(rng.integers(1, max_relabel_h + 1)), len(S) - 1)
            goal = S[future, gs:ge]
            end = min(t + chunk, len(S) - 1)
            progress = float(np.linalg.norm(S[t, gs:ge] - goal) - np.linalg.norm(S[end, gs:ge] - goal))
            if progress < min_progress:
                continue
            state = S[t].copy()
            state[ds:de] = goal
            action_chunk = A[t : t + chunk]
            if len(action_chunk) < chunk:
                action_chunk = np.concatenate(
                    [action_chunk, np.repeat(action_chunk[-1:], chunk - len(action_chunk), axis=0)], axis=0
                )
            states.append(state)
            chunks.append(action_chunk)
            progresses.append(progress)
    return (
        normalizer.encode(np.asarray(states, np.float32)),
        np.asarray(chunks, np.float32),
        np.asarray(progresses, np.float32),
    )


def build_auxiliary_chunks(
    episodes,
    spec,
    normalizer,
    chunk,
    rng,
    *,
    goal_pool,
    goal_copies=2,
):
    """Build state-conditioned chunks from an auxiliary behavior bank.

    Every chunk is paired with multiple independently sampled navigation goals.
    This teaches the flow that behavior demanded by these states is invariant to
    the current route target. The runtime model can therefore infer the useful
    action mode continuously from state, without a hand-authored posture
    predicate or a separate specialist network.
    """
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    states, chunks = [], []
    goals = np.asarray(goal_pool, dtype=np.float32)
    if len(goals) == 0:
        raise ValueError("goal_pool must contain at least one navigation goal")
    copies = max(1, int(goal_copies))
    for episode in episodes:
        S, A = episode.states, episode.actions
        for t in range(len(A)):
            action_chunk = A[t : t + chunk]
            if len(action_chunk) < chunk:
                action_chunk = np.concatenate(
                    [action_chunk, np.repeat(action_chunk[-1:], chunk - len(action_chunk), axis=0)],
                    axis=0,
                )
            for _ in range(copies):
                state = S[t].copy()
                state[ds:de] = goals[int(rng.integers(0, len(goals)))]
                states.append(state)
                chunks.append(action_chunk)
    return (
        normalizer.encode(np.asarray(states, np.float32)),
        np.asarray(chunks, np.float32),
    )
