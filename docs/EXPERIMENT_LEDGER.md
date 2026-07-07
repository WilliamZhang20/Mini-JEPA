# Experiment Ledger

This ledger records directions that were tried while moving from BC/RL execution
policies to SSL latent action planning. It focuses on unsolved or recently
solved cases and explains why failures remain below the previous best controller.

## Adroit

### Door

- Direction: self-supervised schedule-phase inverse prior over demo-derived
  future latents.
- Result: solved, 1.00/30.
- Decision: old explore-WM BC artifact removed.

Why it worked: Door has a stable temporal structure and a fixed articulated
object. A progress phase plus future-conditioned inverse chunk can realize the
same latent sequence without copying action labels at runtime.

### Hammer

- Direction: phase inverse priors with different phase counts and direct
  schedule execution.
- Results:
  - p4 direct schedule: 1.00/30 on seed 64000.
  - p6 schedule: about 0.90/30.
  - progress-guided candidate scoring: reached 0.966/30 in an earlier check but
    was slower than direct p4 execution.
- Decision: p4 phase inverse is the retained SSL controller; old Hammer BC
  artifact removed.

Why it worked: Hammer is contact-rich, but the successful demo manifold is
phase-ordered: reach/tool alignment, lift/position, strike. A small number of
self-supervised phases preserves this structure without requiring RL fine-tune.

### Pen

- Direction: raw+latent future-conditioned flow action prior.
- Result: solved enough to replace BC, 0.90/30.
- Decision: old explore-WM BC artifact removed.

Why it worked: Pen benefits from a stochastic action prior because in-hand
orientation is multimodal. Adding raw current/future state to the latent
condition preserved proprioceptive precision the compact latent alone can smooth
away.

### Relocate

Previous best/SOTA in this repo: retained JEPA-latent BC on offline demos, 1.00
on checked runs.

Best SSL result so far: flat inverse/flow variants around 0.40/10. Later phase
and horizon variants fell to 0.00-0.10/10.

Directions tried:

- Flat future inverse from `(z_t, z_future)` to action chunks.
- Flat future flow from `(z_t, z_future)` to action chunks.
- Raw+latent conditioning for high-DoF precision.
- Horizon changes around h8/h16.
- Action scale changes around 0.8/1.2.
- Self-supervised phase inverse with small demo bank.
- Cached full-demo phase inverse with large pair bank.
- Phase execution with `exec-k=1` and `exec-k=4`.

Why these do not work yet relative to the 1.00 BC baseline:

- Relocate is not just temporal progress. It requires a latent predicate for
  possession: grasp ball, keep ball in hand during transport, then release/place.
  Generic demo-nearest future retrieval often selects visually plausible futures
  without ensuring the hand has controllable possession.
- The action prior can imitate short chunks locally but loses the contact mode
  switch. Once the ball is missed or dropped, future-conditioned chunks continue
  toward the desired future but no longer have a causal path to it.
- JEPA latent rollout scoring is not reliable enough under high-DoF contact to
  filter these failures. It tends to reward latent proximity even when the
  physical grasp state is wrong.

Most promising next direction: object/contact-aware SSL structure. Add learned
or self-labeled predicates for ball contact/possession/height, train separate
future priors for grasp, transport, and place, and make the high level advance
only when the predicate is satisfied.

## AntMaze And PointMaze

### PointMaze

- Direction: H-JEPA graph plus SSL inverse low level.
- Result: solved on checked UMaze/Medium/Large runs, 1.00/1.00/1.00.

Why it works: the point agent has simple dynamics, so a future-conditioned
inverse low level can reliably move between subgoals. The graph handles walls
and sparse reward by planning through empirically reachable landmarks.

### AntMaze UMaze

Previous best in this repo: H-JEPA with BC low level around 0.93.

Directions tried:

- SSL inverse low level direct to goal.
- H-JEPA graph with inverse low level.
- Relaxed graph connectivity.
- Comparisons against HIQL checkpoints.

Current read: direct SSL inverse can solve very short UMaze checks, while the
graph can hurt when the direct low level already reaches the visible goal. The
hierarchy should become conditional: use direct control when reachability is
easy, and graph planning only when walls block the route.

### AntMaze Medium/Large

Previous best/SOTA in this repo: historical HIQL logs around 0.77 Medium and
0.54 Large, with a 0.66 50-episode Large peak.

Current checked result: old HIQL Medium/Large checkpoints are not reproducible in
the current environment and can score 0/20 on old and fresh seeds.

Directions tried:

- Retesting tuned HIQL checkpoints.
- Retesting H-JEPA graph variants.
- SSL inverse low-level checks.
- Neural high-level HWM-style variants in the maze stack.

Why these do not work yet relative to the historical HIQL best:

- The ant walker is the bottleneck. A graph can propose useful subgoals, but the
  low level must reliably move an 8-DoF ant between them.
- The hard graph is benchmark-reliable because it stores empirical reachability,
  but it does not generalize. The neural high level is more general in principle,
  but current rollouts compound error and can hallucinate wall-crossing
  feasibility.
- The historical HIQL result appears sensitive to environment/checkpoint drift,
  so it should not be claimed as current SOTA until reproduced.

Most promising next direction: rebuild a reproducible SSL-conditioned walker,
then use a hybrid high level: neural reachability proposals constrained by an
empirical graph or learned wall-feasibility classifier.

## FrankaKitchen

Previous best/SOTA in this repo: historical `kitchen_flow_skill_ft3.pt` logs
claimed up to 0.90 full-4, but this is not reproducible under the current
code/environment.

Current best reproducible fallback:

- Raw/chunked flow prior: about 2.12/4 subtasks with 0.12 full-4 on an 8-episode
  sweep, and about 2.05/4 with 0.00 full-4 on a 20-episode validation.

Directions tried:

- Re-evaluating the old skill hierarchy and self-imitation checkpoints.
- Raw flow action chunks with different `exec-k` values.
- Lower flow initial noise.
- JEPA reward-head chunk selection over flow candidates.
- Flat future inverse.
- Same-latent HWM macro model plus future-conditioned inverse low level.
- Short smoke HWM training to validate the code path after arXiv:2604.03208.
- Action-prior and scheduler variations.

Why these do not work yet relative to the old 0.90 log:

- The old 0.90 checkpoint is not reproducible, so the current target is not a
  stable artifact. Current checks collapse to 0.10-0.20/4 mean tasks on logged
  seeds.
- Raw flow can solve early subtasks, but full-4 requires reliable sequencing.
  It often stalls around the third task, especially the light switch.
- JEPA reward-head selection was worse than raw flow, indicating the predictor
  or reward probe is not precise enough for contact-rich subtask boundaries.
- The HWM attempt is algorithmically aligned with the paradigm but not yet
  trained at sufficient scale. The tiny smoke model scored 0 tasks and only
  proves integration, not capability.

Most promising next direction: keep the same-latent hierarchy, but make it
subtask-aware and faster to train. Use replay-labeled subtask predicates, cache
latent banks, train the macro model on successful segments, and gate low-level
inverse execution by predicted subtask completion instead of terminal latent
distance alone.

## FetchSlide

Previous best in this repo: JEPA-latent TQC/HER around 0.83; reference TQC is in
the high 0.8 range.

Directions tried:

- Long-horizon flow priors.
- RL-trial flow priors.
- Future-conditioned inverse priors.
- Goal-conditioned inverse variants.
- JEPA ranking/refinement.
- Action scaling and execution horizon changes.

Why these do not work yet relative to the retained RL baseline:

- Slide is ballistic. The key control decision is the initial strike impulse,
  after which the agent has little corrective authority.
- Receding-horizon JEPA ranking favors locally plausible contact but does not
  reliably predict long puck coasting under friction.
- The task likely needs a strike-specific latent objective or impulse prior
  rather than generic future-conditioned chunk imitation.
