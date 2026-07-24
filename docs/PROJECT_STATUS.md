# Project Status

This repo uses self-supervised, future-conditioned latent action planning.

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

This is not a demo-free doctrine. Demonstrations are compatible with the JEPA
view when treated as observed trajectories, desirable futures, and evidence for
the world model/inverse dynamics. The distinction in this repo is between that
use and training the retained runtime controller as a direct supervised
`state -> expert action` clone. LeCun's proposed architecture also contains an
actor; demonstrations are evidence about desirable futures and inverse dynamics,
not a restriction on the source of observed behavior.

## Current Best Results

| Area | Status | Current best checked result |
| --- | --- | --- |
| FetchReach | Solved | JEPA MPC, 0.95-1.00 |
| FetchPush | SSL replacement solved | flow prior + JEPA chunk selection, 1.00 |
| FetchPickAndPlace | SSL replacement solved | inverse prior + JEPA chunk selection, 1.00 |
| FetchSlide | Solved | goal-frame equivariant ballistic JEPA HWM, 0.848 + 0.857 over independent 1000-episode blocks (0.853/2000) |
| PointMaze UMaze/Medium/Large | SSL replacement solved (Dijkstra retired) | HWM flow-macro-prior high level + directed flow walker, 1.00/1.00/1.00 (matches the legacy graph+inverse; no Dijkstra) |
| AntMaze UMaze | SSL replacement solved | HWM flow-macro + unified condition-modulated walker, 1.00/60 (1.00/1.00/1.00) |
| AntMaze Medium | Improved | same unified walker, progress condition 0.8, 0.775/80 (0.75/0.75/0.80/0.80) |
| AntMaze Large | Improved | unified walker with additional auxiliary-behavior mixing, 0.533/60 (0.55/0.45/0.60) |
| Adroit Door | SSL replacement solved | schedule-phase inverse, 1.00/30; old BC removed |
| Adroit Hammer | SSL replacement solved | p4 schedule-phase inverse, 1.00/30 fresh validation; old BC removed |
| Adroit Pen | SSL replacement solved | raw+latent future flow, 0.90/30; old BC removed |
| Adroit Relocate | SSL replacement accepted | dual possession-specialist inverse (firm 0.045 switch) on demo-locked futures with palm-ball-emphasis reach + ball-target-emphasis held specialists, 0.957/210 held-out; former BC removed |
| FrankaKitchen | Four-task SSL solved; reordered/all-task generalization in progress | demo-handoff graph + live demo re-locking + all-task completion probe: 0.80/100 on a scrambled four-task request across four fresh seeds; full-7 remains open |
| HandManipulate Block/Egg (full) | Open, controller-bound | demo-free SSL; WM solved, but MPC/flow controllers ~0 on full-pose (see ledger) |
| HandManipulateBlockRotateZ (stepping stone) | Open, ~0.05-0.10 demo-free SSL | geodesic-supervised DexterousJEPA (7.9° H=8) + future-conditioned flow; flow breaks the finger-gaiting regrasp ceiling (65° sustained rotation) but directed control caps ~30-40° closed; fine CEM converts a minority of episodes. Stronger models (DiT/CFG/rotation-weighted/self-goaling) did not crack directional chaining |
| HandManipulateBlock + continuous touch | Open, controller-bound | 60k-step reward-free tactile dataset + 17-patch anatomical tactile JEPA; grouped tactile encoding improves median goal cost 2.357 → 2.023 but remains 0/5 |

## Replacement Rules

- Mark a task solved only after a fresh, meaningful evaluation slice.
- Keep runtime controllers within the self-supervised world-model,
  future-conditioned-prior, and hierarchy thesis.
- For contact-rich tasks, do not assume JEPA rollout scoring helps. Use measured
  evidence because predictor exploitation has repeatedly hurt Slide, Adroit, and
  Kitchen.
- For long-horizon tasks, prefer hierarchy over longer flat MPC.

- **FetchSlide:** the generic receding-horizon formulation was the wrong
  hierarchy. The new controller plans once at the contact event: a ballistic
  HWM predicts the absorbing post-coast latent and puck endpoint from the
  pre-impact JEPA latent plus a compact strike macro, enumerates goal-relative
  macros, commits to one strike, and then does not replan during the coast.
  The first coordinate MLP improved 0.42 -> 0.735. A goal-frame equivariant
  successor adds canonical contact/velocity features, Fourier strike features,
  gated residual blocks, and distance-balanced geometric training. One
  self-supervised on-policy recalibration round then reaches **0.848 and 0.857**
  on independent 1000-episode blocks.
  There is no reward/value/policy-gradient training. Scripted pre-contact
  alignment remains and should not be confused with fully learned end-to-end
  manipulation.

## Current Open Problems

- **Shadow Hand smoothness/tactile Block (2026-07-23):** iCEM now constrains
  candidates relative to the previously executed action before world-model
  scoring. Across three RotateZ seed blocks it preserves 1/9 success while
  reducing action-delta RMS about 0.50 → 0.22 and jerk about 0.75 → 0.29.
  The continuous-touch Block variant adds 92 taxels and a grouped encoder with
  17 anatomical palm/phalanx patches (93 total tokens instead of 168). It beats
  scalar tactile tokenization on median terminal goal cost (2.023 vs 2.357)
  but remains unsolved and slightly worse than no-touch Block (1.867). The
  checkpoint is retained because it is the correct base for future
  cross-object/contact adaptation, not because tactile sensing alone fixed the
  long-horizon controller.

- **Adroit Relocate residual tail:** replacement is accepted. The strongest
  SSL controller is a dual possession-specialist inverse: a reach specialist and
  a held/transport specialist (each trained only on its contact regime) switched
  on the live palm-ball predicate at a firm 0.045, both tracking a demo-locked
  h8 future index. Both carry input-feature emphasis for the live-vs-demo
  offset: the reach specialist duplicates the live palm-ball vector (dims 30:33)
  8x to servo grasp to the live ball, and the held specialist duplicates the
  live ball-target vector (dims 36:39) 8x to servo placement to the live target.
  **0.957/210 on held-out seeds 81/82/83/85/87/88/89000** (0.955/330 over 11
  seeds). The former BC's 1.00 development result is treated as statistically
  indistinguishable at this evaluation scale, and its checkpoint has been
  removed. The firm 0.045 switch removed most transport
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
  (`train/train_flow_walker.py --directed --emphasis-repeat`) as the low level:
  UMaze 0.85/60 (was 0.33). (2) A **HWM neural high level with a flow prior over
  macro-actions** (arXiv:2604.03208 + `train/train_hwm_macro_flow.py`): sampling macros
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
  flow samples is best). The remaining gap to the historical Medium result
  (~0.77) is
  the walker's raw gait speed: far-goal episodes time out within the 1000-step
  budget. Next: a faster gait (chunk-16 and aggressive filters both failed — the
  SSL walker is at its imitation-speed ceiling, so this needs faster
  demonstrations, not another knob). **PointMaze is now also fully migrated off the
  Dijkstra graph** to the same flow-macro HWM + directed flow walker paradigm
  (U/M/L 1.00/1.00/1.00, matching the old graph+inverse), so no maze task uses
  Dijkstra anymore. See EXPERIMENT_LEDGER for recipes and A/Bs.
- **AntMaze unified low level (2026-07-23, reproducible):** the old solution
  correctly identified falling as the locomotion bottleneck, but encoded that
  knowledge as a torso-quaternion threshold and a separate self-righting
  network. The replacement trains one FiLM-modulated residual flow over
  directed gait chunks and 794 clean self-righting demonstrations. Each
  auxiliary chunk is paired with multiple random navigation goals, so the
  model learns from state when that behavior is useful rather than receiving a
  runtime recovery flag. The old thresholds, specialist/risk networks, and
  collection scripts are removed. Fully seeded flow and environment sweeps:
  **UMaze 1.00/60 (1.00/1.00/1.00), Medium 0.775/80
  (0.75/0.75/0.80/0.80), Large 0.533/60 (0.55/0.45/0.60)**. This improves both
  the mean and seed-block spread over the specialist configuration.
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
  A 2026-07-23 follow-up removes the runtime completion oracle. A small probe
  consumes frozen JEPA latents plus normalized live state and is trained on
  aligned demonstration replays. With a 0.99 threshold and three-frame
  debounce it matches the 0.80/25 environment-switched dev slice with zero
  premature switches and scores **0.784/125 full-4, 3.58/4 mean** on held-out
  seeds 50000-90000, versus 0.816/125 for the environment switch.
  `info['step_task_completions']` is scoring-only in this variant.
  A second follow-up generalizes the high level rather than pretending physical
  task order is arbitrary. Directed handoff costs are learned from specialist
  demo terminal/start arm poses, and a dynamic program routes through any
  requested subset. A scrambled four-task request reaches **0.80/100** across
  four fresh seed blocks (0.88/0.76/0.80/0.76) with the all-task learned
  completion probe. Live demo re-locking with a 0.10 margin fixes the weak
  handoff tail; without it, fresh seeds averaged 0.62 and the former 0.87
  single-seed block was optimistic. The entire pipeline now accepts all seven
  environment tasks. A fresh three-seed full-seven validation uses a dynamic
  490-step budget and reaches **0.017/60 full-7, 4.22/7 mean**
  (0.05/0.00/0.00). Hinge cabinet is the remaining dominant low-level failure,
  so full-seven is supported but open.

## Canonical Docs

- `docs/ARCHITECTURE.md`: package/script boundary and directory organization.
- `docs/EXPERIMENT_LEDGER.md`: tried axes, results, and why unsolved cases remain
  below previous best/SOTA.
- `docs/HANDOFF_RELOCATE_SSL_CONTACT.md`: downstream handoff for the unsolved
  Relocate SSL contact-planning gap.
