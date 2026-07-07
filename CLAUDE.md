# CLAUDE.md — Agent Guide

This repo studies compact action-conditioned JEPA world models for
Gymnasium-Robotics control. The active direction is self-supervised control in
latent space, not policy cloning as the final algorithm.

## Core Paradigm

Use JEPA as a latent world model:

```text
z_t = encoder(o_t)
z_future = target_encoder(o_future)
p(z_t, a_{t:t+H-1}) -> z_future
```

Demos specify desirable futures. Trials teach which action chunks cause which
futures. A future-conditioned action prior proposes chunks from `(z_t,
z_future)`, and an action-conditioned JEPA model predicts or verifies whether
the chunk realizes the latent subgoal.

BC/RL artifacts are retained only where the SSL latent-control replacement has
not matched them.

## Where To Look

- `README.md`: public overview and headline results.
- `docs/PROJECT_STATUS.md`: current solved/open task status.
- `docs/EXPERIMENT_LEDGER.md`: directions tried and why unsolved cases remain
  below previous best/SOTA.
- `docs/ARCHITECTURE.md`: package/script boundary and directory organization.

## Repo Boundary

- Put reusable code in `jepa_robotics/`.
- Put reusable SSL-control algorithm pieces in `jepa_robotics/algos/`.
- Keep `scripts/` as thin command-line entry points for training, eval, record,
  conversion, and artifact IO.
- Do not duplicate model/index/planning logic across scripts. Promote it into
  the package first.
- Preserve existing script filenames unless there is a compatibility wrapper,
  because run records cite exact commands.

## Current Truth

- FetchPush and FetchPickAndPlace are clean SSL latent-planning base cases:
  future-conditioned flow/inverse action priors plus JEPA chunk selection reach
  1.00 success.
- Door, Hammer, and Pen have SSL replacements. Their old Adroit BC artifacts
  were removed after fresh validation.
- Relocate is still open. Generic future inverse/flow and phase schedules remain
  below retained BC.
- PointMaze is solved with H-JEPA plus SSL inverse low level on checked runs.
- AntMaze Medium/Large are not currently reproducible from historical HIQL logs.
- FrankaKitchen is open. The old 0.90 full-4 log is not reproducible; raw flow
  around two subtasks is the current fallback.
- FetchSlide remains RL-retained because ballistic strike control has not yet
  been matched by the SSL action-prior planners.

## Environment Notes

- Use conda env `myenv`.
- Use `MUJOCO_GL=egl` for MuJoCo rendering.
- Prefer `--device auto` or `--device cuda` on the H200 GPU.
- Experiment artifacts live under `runs/`.
- Adroit explore artifacts are canonical under `runs/adroit_*/explore`; old
  `runs/adroit_*_explore` paths are symlinks.

## Validation Rules

- Do not mark a task solved from a smoke test.
- Prefer at least a fresh 30-episode eval for solved Adroit/Fetch claims.
- For Kitchen and AntMaze, explicitly report episode counts and seeds because
  reproducibility has been fragile.
- If JEPA rollout scoring worsens a contact-rich task, record that result
  instead of assuming more tuning will fix it.

## Development Rules

- Use `rg` for search.
- Use `apply_patch` for manual file edits.
- Do not revert user or prior-run changes unless explicitly asked.
- Keep generated artifacts out of source changes unless they are videos or notes
  the user explicitly requested.
- When replacing BC/RL with SSL control, update both the docs and the artifact
  cleanup state.
