"""Compatibility facade for planning objectives.

New code should import :mod:`jepa_robotics.algos.planning.objectives`.
"""

from ..algos.planning.objectives import CommonScoringMixin, GoalScoringMixin, ManipScoringMixin, StrikeScoringMixin

__all__ = ["CommonScoringMixin", "ManipScoringMixin", "StrikeScoringMixin", "GoalScoringMixin"]
