"""Compatibility entry point for the categorized Kitchen evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval.eval_kitchen_subtask_inverse import main


if __name__ == "__main__":
    main()
