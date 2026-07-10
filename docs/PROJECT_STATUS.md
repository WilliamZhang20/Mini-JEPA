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
| PointMaze UMaze/Medium/Large | SSL replacement solved (Dijkstra retired) | HWM flow-macro-prior high level + directed flow walker, 1.00/1.00/1.00 (matches the legacy graph+inverse; no Dijkstra) |
| AntMaze UMaze | SSL replacement solved | HWM flow-macro-prior high level + directed flow walker (no Dijkstra), 0.867/60 |
| AntMaze Medium | Improved (open) | HWM flow-macro-prior high level + directed flow walker, 0.39/80 (beats graph 0.26 and Gaussian-CEM 0.067); below historical HIQL ~0.77 |
| AntMaze Large | Improved (open) | HWM flow-macro-prior high level + directed flow walker, 0.18/60 reproducible (was 0.00) |
| Adroit Door | SSL replacement solved | schedule-phase inverse, 1.00/30; old BC removed |
| Adroit Hammer | SSL replacement solved | p4 schedule-phase inverse, 1.00/30 fresh validation; old BC removed |
| Adroit Pen | SSL replacement solved | raw+latent future flow, 0.90/30; old BC removed |
| Adroit Relocate | Open (very close, gap ~0.045) | retained BC 1.00; best SSL: dual possession-specialist inverse (firm 0.045 switch) on demo-locked futures with palm-ball-emphasis reach + ball-target-emphasis held specialists, 0.957/210 on held-out seeds |
| FrankaKitchen | SSL replacement solved (reproducible) | subtask-specialist inverse controller: 0.813 full-4 / ~3.63 mean tasks over 150 eps (6 seeds); held-out 5-seed 0.816/125 |

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
- **AntMaze Medium/Large (improved 2026-07-10, reproducible):** two SSL
  upgrades give the first reproducible non-zero Medium in the current env.
  (1) A **directed-motion goal-delta-emphasis flow-matching walker**
  (`train_flow_walker.py --directed --emphasis-repeat`) as the low level:
  UMaze 0.85/60 (was 0.33). (2) A **HWM neural high level with a flow prior over
  macro-actions** (arXiv:2604.03208 + `train_hwm_macro_flow.py`): sampling macros
  from a flow conditioned on `(z_high, goal_xy)` keeps the macro search on the
  feasible demonstrated manifold, fixing the Gaussian-CEM wall-crossing
  hallucination. On Medium the high-level comparison (same directed walker) is
  **flow-macro 0.39/80 > empirical graph 0.26 > Gaussian CEM 0.067** -- the
  neural flow high level is now the best in the repo AND generalizes (no stored
  reachability table). **This flow-macro HWM + directed flow walker is now the
  canonical AntMaze controller across all three mazes, replacing the Dijkstra
  graph: UMaze 0.867/60, Medium 0.39/80, Large 0.18/60** (all first-reproducible
  or best-in-repo, fully neural). K-step macro lookahead did not help (horizon-2
  0.25 < horizon-1 0.39 — g's rollout compounds; the greedy 1-hop from feasible
  flow samples is best). The remaining gap to historical HIQL (Medium ~0.77) is
  the walker's raw gait speed: far-goal episodes time out within the 1000-step
  budget. Next: a faster gait (chunk-16 and aggressive filters both failed — the
  SSL walker is at its imitation-speed ceiling, so this needs an RL fine-tune or
  faster demos, not another knob). **PointMaze is now also fully migrated off the
  Dijkstra graph** to the same flow-macro HWM + directed flow walker paradigm
  (U/M/L 1.00/1.00/1.00, matching the old graph+inverse), so no maze task uses
  Dijkstra anymore. See EXPERIMENT_LEDGER for recipes and A/Bs.
- **FrankaKitchen (SOLVED this session, reproducible):** the Relocate recipe
  ported directly to the sequential task closes it. Four **segment-pure
  per-subtask inverse specialists** (microwave/kettle/light switch/slide
  cabinet), each trained ONLY on transitions the labeler tagged as working
  toward that subtask (law 1, no blurring), track a **per-subtask demo-locked
  future index** built from that subtask's demo segments (law 2), and a **firm
  scheduler** advances to the next specialist only when the env reports the
  current subtask complete (law 4, firm predicate switch, using ground-truth
  `info['step_task_completions']`). Two extra levers from this task:
  (a) **feature-matched emphasis** — emphasize the feature carrying that
  subtask's dominant servo error: the **robot arm joints (obs 0:9)** for the
  approach-dominated light switch, the **object qpos dims** for the
  manipulation-dominated microwave/kettle/slide (law 3, refined: the object dim
  is near-constant until manipulated, so it carries no approach signal);
  (b) **arm-pose demo matching + re-lock** (`--match-dims 0,9 --relock-margin
  0.3`) so the demo lock keys on reachability (arm pose) and recovers from a bad
  handoff lock — this was the single largest lever (0.65 -> 0.80). A wider light
  demo bank (`kitchen_subtask_light_arm_wide.pt`, 800 segments) then lifted it to
  **0.813 full-4 / 150 eps** (6 seeds, 25 eps each: 0.80/0.84/0.76/0.76/0.92/
  0.80), held-out 5-seed 0.816/125, vs the reproducible raw-flow baseline of
  0.00 full-4 / 2.05 mean. Task order is not free (an order-agnostic greedy
  scheduler loses to the fixed canonical order): the demos encode physical
  dependencies (light switch only reachable from the post-kettle pose).
  Scripts: `scripts/train_kitchen_subtask_inverse.py`,
  `scripts/eval_kitchen_subtask_inverse.py`. See
  `docs/HANDOFF_KITCHEN_SUBTASK_SSL.md`.

## Canonical Docs

- `docs/ARCHITECTURE.md`: package/script boundary and directory organization.
- `docs/EXPERIMENT_LEDGER.md`: tried axes, results, and why unsolved cases remain
  below previous best/SOTA.
- `docs/HANDOFF_RELOCATE_SSL_CONTACT.md`: downstream handoff for the unsolved
  Relocate SSL contact-planning gap.
