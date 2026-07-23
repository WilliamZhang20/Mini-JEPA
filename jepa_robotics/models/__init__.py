"""JEPA model components, split by responsibility.

``ActionConditionedJEPA`` is the shared action-conditioned world model used
across all tasks (Fetch, PointMaze, ...). Per-task behaviour lives in the
data experts (data.py), task configs (tasks.py) and planning objectives
(algos/planning/objectives/),
not in the model itself. Components are split into submodules for clarity and
to make Roadmap-A extensions (ensemble heads, stochastic latent) additive.
"""

from .mlp import MLP
from .world_model import ActionConditionedJEPA
from .dexterous import DexterousJEPA, DexterousFlowPrior
from .policy import GoalConditionedPolicy, LatentSubgoalActor
from .regularizers import (
    covariance_regularizer,
    normalized_mse,
    variance_regularizer,
)

__all__ = [
    "MLP",
    "ActionConditionedJEPA",
    "DexterousJEPA",
    "DexterousFlowPrior",
    "GoalConditionedPolicy",
    "LatentSubgoalActor",
    "normalized_mse",
    "variance_regularizer",
    "covariance_regularizer",
]
