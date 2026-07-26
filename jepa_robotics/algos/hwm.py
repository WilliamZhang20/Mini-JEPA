"""Same-latent hierarchical world model components.

This is the HWM-style interface: a high-level model predicts future states in the
same latent space as the frozen low-level JEPA, conditioned on learned
macro-actions encoded from primitive action chunks.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from jepa_robotics.models import MLP


class MacroActionEncoder(nn.Module):
    """Compress a primitive action chunk into one continuous macro-action."""

    def __init__(self, action_dim: int, macro_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(action_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, macro_dim)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(chunk)
        return self.head(h[-1])


class LatentMacroPredictor(nn.Module):
    """Residual high-level dynamics in the frozen low-level JEPA latent space."""

    def __init__(self, latent_dim: int, macro_dim: int, hidden: int = 512, n_blocks: int = 3) -> None:
        super().__init__()
        layers: list[int] = [latent_dim + macro_dim]
        layers.extend([hidden] * max(1, n_blocks))
        layers.append(latent_dim)
        self.net = MLP(layers, layer_norm=True)

    def forward(self, z: torch.Tensor, macro_action: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, macro_action], dim=-1))


class HighEncoder(nn.Module):
    """Compress a frozen low-level JEPA latent into an abstract HWM state."""

    def __init__(self, low_dim: int, abstract_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = MLP([low_dim, hidden, hidden, abstract_dim], layer_norm=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class MacroEncoder(MacroActionEncoder):
    """Compatibility name for the abstract HWM macro-action encoder."""


class MacroPredictor(nn.Module):
    """Residual dynamics in the abstract high-level latent."""

    def __init__(self, abstract_dim: int, macro_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = MLP([abstract_dim + macro_dim, hidden, hidden, abstract_dim], layer_norm=True)

    def forward(self, latent: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
        return latent + self.net(torch.cat([latent, macro], dim=-1))


class SubgoalDecoder(nn.Module):
    """Decode an abstract HWM latent into achieved-goal coordinates."""

    def __init__(self, abstract_dim: int, goal_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = MLP([abstract_dim, hidden, hidden, goal_dim], layer_norm=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TopologyScorer(nn.Module):
    """Predict demonstration-time distance from an abstract state to a goal.

    Euclidean distance is not a valid high-level objective in a maze: the first
    correct move around a concave wall can increase straight-line distance.
    This scorer receives the learned HWM state plus explicit current/goal
    geometry and regresses temporal distance using only within-trajectory future
    pairs. It is therefore a self-supervised route potential, not a reward
    critic.
    """

    def __init__(self, abstract_dim: int, goal_dim: int, hidden: int = 256) -> None:
        super().__init__()
        # z, current, goal, signed delta, absolute delta
        input_dim = abstract_dim + 4 * goal_dim
        self.net = MLP(
            [input_dim, hidden, hidden, hidden, 1],
            layer_norm=True,
        )

    def forward(
        self,
        latent: torch.Tensor,
        current_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        delta = goal_pos - current_pos
        features = torch.cat(
            [latent, current_pos, goal_pos, delta, delta.abs()],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class CoordinateTopologyScorer(nn.Module):
    """Topology potential using only current and goal coordinates.

    This variant can rank directly sampled waypoint coordinates without asking a
    latent dynamics model to hallucinate the waypoint's abstract state.
    """

    def __init__(self, goal_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = MLP(
            [4 * goal_dim, hidden, hidden, hidden, 1],
            layer_norm=True,
        )

    def forward(
        self,
        latent: torch.Tensor,
        current_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        del latent
        delta = goal_pos - current_pos
        return self.net(
            torch.cat([current_pos, goal_pos, delta, delta.abs()], dim=-1)
        ).squeeze(-1)


class GoalConditionedWaypointMemory:
    """Retrieve locally feasible waypoints from successful demonstrations.

    Every candidate is an observed local transition. Goal conditioning selects
    transitions from trajectories solving the requested route, avoiding the
    through-wall endpoints produced by a continuous latent decoder. This is a
    one-step retrieval policy, not a hand-authored map or graph search.
    """

    def __init__(
        self,
        current: np.ndarray,
        goals: np.ndarray,
        waypoints: np.ndarray,
        current_weight: float = 1.0,
        goal_weight: float = 4.0,
        neighbors: int = 7,
    ) -> None:
        self.current = np.asarray(current, dtype=np.float32)
        self.goals = np.asarray(goals, dtype=np.float32)
        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        self.current_weight = float(current_weight)
        self.goal_weight = float(goal_weight)
        self.neighbors = int(neighbors)
        if not (
            len(self.current) == len(self.goals) == len(self.waypoints)
            and len(self.current) > 0
        ):
            raise ValueError("Waypoint memory arrays must be non-empty and aligned")

    def query(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        current_pos = np.asarray(current_pos, dtype=np.float32)
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        distance = (
            self.current_weight
            * np.square(self.current - current_pos[None]).sum(axis=1)
            + self.goal_weight
            * np.square(self.goals - goal_pos[None]).sum(axis=1)
        )
        k = min(max(1, self.neighbors), len(distance))
        indices = np.argpartition(distance, k - 1)[:k]
        return np.median(self.waypoints[indices], axis=0).astype(np.float32)


class DiscreteTopologyRouter(nn.Module):
    """Predict the next discrete region on a route to the requested goal."""

    def __init__(self, region_count: int, hidden: int = 128) -> None:
        super().__init__()
        # normalized current xy, goal xy, and signed displacement
        self.net = MLP([6, hidden, hidden, region_count], layer_norm=True)

    def forward(
        self, current_pos: torch.Tensor, goal_pos: torch.Tensor
    ) -> torch.Tensor:
        current = current_pos / 4.0
        goal = goal_pos / 4.0
        delta = (goal_pos - current_pos) / 8.0
        return self.net(torch.cat([current, goal, delta], dim=-1))


def build_macro_data(episodes, spec, normalizer, world_model, device, stride, overlap=2):
    """Encode primitive trajectory windows into frozen low-level macro data."""
    goal_start, goal_end = spec.obs_dim, spec.obs_dim + spec.goal_dim
    states, futures, chunks, positions = [], [], [], []
    step = max(1, stride // overlap)
    for episode in episodes:
        for t in range(0, len(episode.actions) - stride, step):
            states.append(episode.states[t])
            futures.append(episode.states[t + stride])
            chunks.append(episode.actions[t : t + stride])
            positions.append(episode.states[t, goal_start:goal_end])
    states = normalizer.encode(np.asarray(states, np.float32))
    futures = normalizer.encode(np.asarray(futures, np.float32))
    with torch.no_grad():
        z = torch.cat([
            world_model.encode(torch.from_numpy(states[i : i + 16384]).to(device))
            for i in range(0, len(states), 16384)
        ])
        z_future = torch.cat([
            world_model.encode(torch.from_numpy(futures[i : i + 16384]).to(device))
            for i in range(0, len(futures), 16384)
        ])
    return (
        z,
        z_future,
        torch.from_numpy(np.asarray(chunks, np.float32)).to(device),
        torch.from_numpy(np.asarray(positions, np.float32)).to(device),
    )


def sample_macro_dataset(episodes, stride: int, overlap: int = 2):
    step = max(1, stride // max(1, overlap))
    states, futures, chunks, starts, finals = [], [], [], [], []
    for ep in episodes:
        states_ep = ep.states.astype(np.float32)
        actions_ep = ep.actions.astype(np.float32)
        T = len(actions_ep)
        if T < stride:
            continue
        starts.append(states_ep[0])
        finals.append(states_ep[-1])
        for t in range(0, T - stride + 1, step):
            states.append(states_ep[t])
            futures.append(states_ep[t + stride])
            chunks.append(actions_ep[t : t + stride])
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(futures, dtype=np.float32),
        np.asarray(chunks, dtype=np.float32),
        np.asarray(starts, dtype=np.float32),
        np.asarray(finals, dtype=np.float32),
    )
