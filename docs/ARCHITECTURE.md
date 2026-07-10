# Architecture

The repo is now organized around a simple boundary:

- `jepa_robotics/` is the importable package. Shared models, data adapters,
  environment wrappers, scoring functions, and reusable SSL-control algorithms
  belong here.
- `scripts/` is the CLI layer. Scripts may parse arguments, load checkpoints,
  launch training/eval loops, and write artifacts, but repeated model or
  planning code should move into `jepa_robotics/`.
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
- `jepa_robotics/scoring/`: reusable task score components for goal,
  manipulation, and striking tasks.
- `jepa_robotics/algos/`: reusable SSL latent-control pieces:
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

## Script Policy

When adding a new experiment:

1. Put the reusable network/index/planning primitive in `jepa_robotics/algos` or
   the appropriate package module.
2. Keep the script focused on argument parsing, checkpoint IO, dataset assembly,
   and calling the reusable primitive.
3. Do not import train/eval classes from another script unless it is a short-term
   compatibility shim. Promote duplicated code into the package first.
4. Preserve existing script filenames when possible because many run records and
   docs cite exact commands.

Retired (2026-07-10): the Dijkstra subgoal-graph maze controller
(`eval_hjepa_maze.py`, `eval_hjepa2.py`, `record_hjepa_maze.py`) — replaced on
every maze by the neural HWM flow-macro high level (`eval_hjepa_hwm.py` +
`train_hwm_macro_flow.py`) over the directed flow walker; the Relocate
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

The old top-level `runs/adroit_*_explore` paths remain as symlinks for old
commands. Canonical JEPA model symlinks live in each task's `checkpoints/`
directory.
