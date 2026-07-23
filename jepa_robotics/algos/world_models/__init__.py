"""Specialized reusable world models above the shared JEPA dynamics."""

from .ballistic import (
    BallisticHWM,
    EquivariantBallisticHWM,
    goal_relative_macro_candidates,
)

__all__ = ["BallisticHWM", "EquivariantBallisticHWM", "goal_relative_macro_candidates"]
