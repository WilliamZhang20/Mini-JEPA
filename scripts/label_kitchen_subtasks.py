"""Compatibility entry point for the categorized Kitchen labeler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data.label_kitchen_subtasks import main


if __name__ == "__main__":
    main()
