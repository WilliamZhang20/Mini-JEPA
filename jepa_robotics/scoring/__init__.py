"""Per-task MPC scoring strategies, split out of evaluate.py.

Each mixin provides one ``_*_scores`` method; ``JEPAMPCPolicy`` composes them
and dispatches on ``score_mode``. Add a new task's score as a new mixin here
(e.g. a maze sub-goal score) without touching the planner core.
"""

from .common import CommonScoringMixin
from .manip import ManipScoringMixin
from .strike import StrikeScoringMixin
from .goal import GoalScoringMixin

__all__ = ["CommonScoringMixin", "ManipScoringMixin", "StrikeScoringMixin", "GoalScoringMixin"]
