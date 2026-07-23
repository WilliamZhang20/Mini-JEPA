"""Compatibility imports for ballistic world models.

New code should import from :mod:`jepa_robotics.algos.world_models.ballistic`.
"""

from .world_models.ballistic import (
    BallisticHWM,
    EquivariantBallisticHWM,
    canonical_ballistic_features,
    fetch_slide_ready,
    goal_to_world_frame,
    goal_relative_macro_candidates,
    slide_macro_action,
    world_to_goal_frame,
)

__all__ = [
    "BallisticHWM",
    "EquivariantBallisticHWM",
    "canonical_ballistic_features",
    "fetch_slide_ready",
    "goal_to_world_frame",
    "goal_relative_macro_candidates",
    "slide_macro_action",
    "world_to_goal_frame",
]
