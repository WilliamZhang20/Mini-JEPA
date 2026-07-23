# Architecture

The repo is now organized around a simple boundary:

- `jepa_robotics/` is the importable package. Shared models, data adapters,
  environment wrappers, planning objectives, and reusable SSL-control
  algorithms belong here.
- `scripts/` is the CLI layer, split into `data/`, `train/`, and `eval/`.
  Scripts may parse arguments, load checkpoints, launch loops, and write
  artifacts; they must not be imported as algorithm libraries.
- `runs/` is artifact storage. It is intentionally outside source control except
  for small notes/videos that are already present in the working tree.
- `docs/` holds project status, experiment ledgers, and design notes. `CLAUDE.md`
  is now kept as a concise agent operating guide.

## Package Layout

- `jepa_robotics/models/`: action-conditioned JEPA, policy networks, MLP blocks,
  and representation regularizers.
- `jepa_robotics/data.py`: episode collection, scripted experts, offline npz
  loading, and normalization.
- `jepa_robotics/envs.py`: Gymnasium-Robotics registration, observation specs,
  flattening, and task-specific environment fixes.
- `jepa_robotics/tasks.py`: task presets.
- `jepa_robotics/evaluate.py`: common JEPA MPC and baseline evaluation code.
- `jepa_robotics/algos/`: reusable SSL latent-control pieces, categorized as
  the number of algorithms grows:
  - `control/`: learned runtime predicates and other controller mechanisms.
  - `planning/objectives/`: goal, manipulation, strike, and common trajectory
    objectives used by MPC. These are planner algorithms, not evaluation
    scripts. `jepa_robotics/scoring/` is now only a compatibility facade.
  - `task_families/`: shared Fetch/Kitchen/maze geometry, metadata, collection
    adapters, and Kitchen's demonstration-derived task-handoff graph.
  - `world_models/`: specialized models above primitive dynamics, including
    the goal-frame equivariant event-conditioned ballistic HWM.
  - `priors.py`: inverse action-chunk priors and diffusion/flow action-prior
    networks, plus the shared sampler.
  - `futures.py`: future-target selection strategies — the demo-locked future
    index (receding-horizon demo tracking) with per-subtask reachability
    matching.
  - `maze_low_level.py`: goal-conditioned maze low levels (flow walker, BC,
    inverse) used under the HWM flow-macro high level.
  - `phase.py`: self-supervised progress phase features and phase-constrained
    future lookup.
  - `hwm.py`: hierarchical world model components (macro-action encoder + latent
    macro predictor) for the HWM high level (arXiv:2604.03208).

`jepa_robotics/models/` already follows a model hierarchy: the world model,
policy models, reusable MLP blocks, regularizers, and dexterous variants are
separate modules. New task-specific policy or dynamics implementations should
go under `algos/`, not beside a CLI.

## Script Policy

When adding a new experiment:

1. Put reusable networks, indexes, samplers, objectives, task geometry, and
   planners in the appropriate `jepa_robotics/algos` category.
2. Add new CLIs under `scripts/data`, `scripts/train`, or `scripts/eval`.
3. Do not import one CLI from another. Shared code is a package dependency.
4. Root-level legacy filenames may be retained only as thin compatibility
   launchers because run records cite exact commands.

The FetchSlide and Kitchen pipelines are the first fully migrated vertical
slices. Existing root scripts are being migrated incrementally; the rule above
prevents further growth of the flat directory while preserving old commands.

Retired (2026-07-10): the Dijkstra subgoal-graph maze controller
(`eval_hjepa_maze.py`, `eval_hjepa2.py`, `record_hjepa_maze.py`) — replaced on
every maze by the neural HWM flow-macro high level (`eval/eval_hjepa_hwm.py` +
`train/train_hwm_macro_flow.py`) over the directed flow walker; the Relocate
contact-scoring experiments (`contact.py`, `train_relocate_contact_*.py`,
`*flow_residual_refiner.py`) — all measured neutral/negative, not part of the
canonical dual-specialist controller; and the failed Kitchen latent-dynamics /
Dreamer / latent-HWM / MPC scripts. The canonical controllers are the
subtask-specialist inverse (Kitchen), dual possession-specialist inverse
(Relocate), and HWM flow-macro + flow walker (mazes).

## Runs Layout

Adroit explore artifacts were merged under their task directories:

- `runs/adroit_door/explore`
- `runs/adroit_hammer/explore`
- `runs/adroit_pen/explore`
- `runs/adroit_relocate/explore`

Explore runs live only under `runs/adroit_*/explore`. Canonical JEPA model
symlinks live in each task's `checkpoints/`
directory.
