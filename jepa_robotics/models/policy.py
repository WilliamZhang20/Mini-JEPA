from __future__ import annotations

import torch
from torch import nn

from .mlp import MLP


class GoalConditionedPolicy(nn.Module):
    """A small action prior learned on the JEPA latent representation.

    The JEPA encoder maps the full observation (which already includes the
    desired goal) to a latent ``z``; this policy maps ``z`` to an action. It is
    trained by behaviour cloning on the collected trajectories, i.e. purely
    self-supervised from the same data the world model uses. It is not meant to
    be the final controller on its own - it is the *proposal* that the
    world-model MPC refines, which is what makes precise contact skills like
    grasping reliable.
    """

    def __init__(self, *, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP([latent_dim, hidden_dim, hidden_dim, action_dim], layer_norm=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))


class LatentSubgoalActor(nn.Module):
    """Actor optimized through a frozen world model toward latent subgoals.

    Unlike ``GoalConditionedPolicy``, this module is not trained to copy action
    labels. Its input is the current JEPA latent and a desired future latent; the
    training objective lives on the world model's predicted future latent.
    """

    def __init__(self, *, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP([2 * latent_dim, hidden_dim, hidden_dim, action_dim], layer_norm=True)

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([z, z_goal], dim=-1)))
