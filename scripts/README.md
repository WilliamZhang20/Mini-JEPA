# Scripts

This directory is the command-line layer, not the algorithm library.

- `train/`: optimization and checkpoint-writing entry points.
- `eval/`: benchmark and rollout entry points.
- `data/`: collection, labeling, and conversion entry points.
- `ops/`: Slurm and pipeline launchers (introduced as those files migrate).
- root-level files: legacy compatibility entry points. Do not add new algorithm
  implementations here.

New code must import reusable networks, planners, samplers, task geometry, and
completion predicates from `jepa_robotics`. A script must not import another
train/eval script; keep a root wrapper only when old commands or run records
need the filename.
