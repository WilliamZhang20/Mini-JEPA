# Project Status

This repo is transitioning from JEPA representations plus BC/RL execution
policies to self-supervised, future-conditioned latent action planning.

The working paradigm is:

```text
z_t = encoder(o_t)
z_future = target_encoder(o_future)
action prior proposes a_{t:t+H-1} from (z_t, z_future)
action-conditioned JEPA predicts z_hat_{t+H}
planner selects chunks that realize demo-derived latent futures
```

Demos specify desirable futures. Trials teach which actions cause which futures.
JEPA predicts in latent space. The actor or planner chooses actions to realize
latent subgoals.

## Current Best Results

| Area | Status | Current best checked result |
| --- | --- | --- |
| FetchReach | Solved | JEPA MPC, 0.95-1.00 |
| FetchPush | SSL replacement solved | flow prior + JEPA chunk selection, 1.00 |
| FetchPickAndPlace | SSL replacement solved | inverse prior + JEPA chunk selection, 1.00 |
| FetchSlide | Not SSL-replaced | JEPA-latent TQC/HER retained, 0.83 |
| PointMaze UMaze/Medium/Large | SSL replacement solved on checked runs | H-JEPA + inverse low level, 1.00/1.00/1.00 |
| AntMaze UMaze | Partially solved | H-JEPA BC low level 0.93 historical; SSL inverse direct low-level can solve short UMaze checks |
| AntMaze Medium/Large | Open | historical HIQL logs are non-reproducible in current env |
| Adroit Door | SSL replacement solved | schedule-phase inverse, 1.00/30; old BC removed |
| Adroit Hammer | SSL replacement solved | p4 schedule-phase inverse, 1.00/30 fresh validation; old BC removed |
| Adroit Pen | SSL replacement solved | raw+latent future flow, 0.90/30; old BC removed |
| Adroit Relocate | Open (very close, gap ~0.045) | retained BC 1.00; best SSL: dual possession-specialist inverse (firm 0.045 switch) on demo-locked futures with palm-ball-emphasis reach + ball-target-emphasis held specialists, 0.957/210 on held-out seeds |
| FrankaKitchen | Open | raw flow fallback around 2.05-2.12/4 subtasks; full-4 success not reproducible |

## Replacement Rules

- Mark a task solved only after a fresh eval that matches or beats the retained
  BC/RL baseline on a meaningful episode slice.
- Delete old BC/RL artifacts only when the SSL controller has matched or beaten
  the retained controller and the old artifact is no longer needed for a cited
  comparison.
- For contact-rich tasks, do not assume JEPA rollout scoring helps. Use measured
  evidence because predictor exploitation has repeatedly hurt Slide, Adroit, and
  Kitchen.
- For long-horizon tasks, prefer hierarchy over longer flat MPC.

## Current Open Problems

- **Adroit Relocate:** still open but very close (gap ~0.045). The strongest
  SSL controller is a dual possession-specialist inverse: a reach specialist and
  a held/transport specialist (each trained only on its contact regime) switched
  on the live palm-ball predicate at a firm 0.045, both tracking a demo-locked
  h8 future index. Both carry input-feature emphasis for the live-vs-demo
  offset: the reach specialist duplicates the live palm-ball vector (dims 30:33)
  8x to servo grasp to the live ball, and the held specialist duplicates the
  live ball-target vector (dims 36:39) 8x to servo placement to the live target.
  **0.957/210 on held-out seeds 81/82/83/85/87/88/89000** (0.955/330 over 11
  seeds), vs retained BC 1.00. The firm 0.045 switch removed most transport
  drops (the transport specialist never inherits a marginal grasp); the
  ball-target emphasis removed most placement near-misses. Remaining failures
  are a diverse long tail (residual reach misses on outlier ball positions,
  residual placement offset, rare mid-transport drops). Neutral-or-harmful:
  scoring branches (barrier, contact dynamics, CVAE), warm-start sampling,
  DAgger/noise-robustified retraining, "leave" hysteresis, exec-k=1,
  enter-delay reach-hold, held palm-ball emphasis, geometry weighting,
  multi-demo candidate ranking. See `docs/HANDOFF_RELOCATE_SSL_CONTACT.md` for
  the 2026-07-09 session log and next directions.
- **AntMaze Medium/Large:** needs a reproducible walker and a neural high-level
  that respects wall feasibility. Current failures look like low-level control
  and checkpoint/environment drift, not just graph planning.
- **FrankaKitchen:** needs a reliable sequential hierarchy. Raw flow can complete
  about two subtasks, but full sequence success is not stable under current code.

## Canonical Docs

- `docs/ARCHITECTURE.md`: package/script boundary and directory organization.
- `docs/EXPERIMENT_LEDGER.md`: tried axes, results, and why unsolved cases remain
  below previous best/SOTA.
- `docs/HANDOFF_RELOCATE_SSL_CONTACT.md`: downstream handoff for the unsolved
  Relocate SSL contact-planning gap.
