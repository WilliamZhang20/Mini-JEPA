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
- Coupled flow+diffusion residual refinement:
  `a_flow ~ flow(a | z_t, z_future)`, then
  `residual(a_demo - a_flow | z_t, z_future, a_flow)`.
- Horizon changes around h8/h16.
- Action scale changes around 0.8/1.2.
- Self-supervised phase inverse with small demo bank.
- Cached full-demo phase inverse with large pair bank.
- Phase execution with `exec-k=1` and `exec-k=4`.
- Relocate-specific decoded-state scoring using observation dims `30:33`
  palm-to-ball and `36:39` ball-to-target.
- A specialized self-supervised Relocate contact probe trained on frozen true
  JEPA latents to predict palm-ball and ball-target distances.
- JEPA-gradient action refinement initialized from flow samples, with anchor and
  smoothness penalties.
- Stronger h8 raw+latent flow retraining on 888k expert transition pairs.
- Shorter h4 raw+latent flow retraining.
- SSL self-imitation: collect 25 successful rollouts from the SSL flow policy,
  train success-only inverse/flow, and train expert+SSL-success mixed flow.
- BC-as-data bridge: collect 59/60 successful trajectories from the retained BC,
  then train future-conditioned flow/inverse priors on those trajectories. The
  runtime controller is still the SSL prior, not BC.
- Horizon-matched h8 target training, where the future bank uses `t+8` instead
  of asking an 8-step chunk to pursue a `t+16` future.
- Rollout-calibrated contact probe: train the palm-ball / ball-target probe on
  `wm.predict_rollout(z_t, action_chunk)` latents instead of true encoder
  latents.
- Contact-trace-conditioned inverse: retrieve a demo contact-distance trajectory
  and condition the inverse prior on that trace.
- Action-conditioned contact dynamics head trained on all outcomes from 80
  SSL-flow Relocate rollouts. It predicts the contact trace directly from
  `(z_t, raw_t, action_chunk)` and is used as a candidate-ranking energy.
- Conditional contact VAE trained on the same SSL contact-trace target:
  `p(contact_trace | z_t, raw_t, action_chunk)`. This models multimodal
  contact outcomes for candidate scoring instead of using a deterministic
  contact-distance head.

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
- The first coupled flow+residual attempt did not beat raw flow. On seed 67000,
  raw flow with 4 candidates scored 0.70/10; the best gated deterministic
  residual refiner also scored 0.70/10. Stochastic DDPM residuals over-corrected
  and dropped to 0.00-0.33 on 3-episode slices. A decoded-state contact score
  using the current JEPA state probe worsened selection to 0.33/3 on the easy
  seed-66000 slice where raw flow hit 1.00/3, so the state probe is not yet a
  reliable possession/contact oracle.
- A specialized contact probe fit true demo latents well
  (`smooth_l1` around `5e-5` after 3000 steps on 60k states), but using it to
  score JEPA-predicted rollout latents still scored only 0.33/3 on seed 66000.
  This suggests the probe is calibrated on encoder latents but not on imagined
  dynamics latents, so the missing piece is contact-aware rollout consistency or
  training probes on predicted latents, not another terminal-state scoring weight.
- JEPA-gradient refinement through the action-conditioned model exploited the
  contact-rich model and collapsed to 0.00/3 on seed 66000, even with a strong
  action anchor. This confirms that direct primitive-action optimization through
  the Relocate JEPA predictor is not reliable enough.
- Stronger expert-flow retraining did not improve the ceiling: the new h8 model
  scored 0.70/10 on seed 67000 and 0.60/10 on seed 68000, matching the old
  checkpoint. h4 control was worse: 0.50/10 on seed 67000 and 0.20-0.50/10 on
  seed 68000.
- SSL self-imitation without the expert distribution collapsed. Success-only
  inverse scored 0.00-0.10/10; success-only flow scored 0.00-0.10/10. Oversampling
  SSL successes into expert data also failed to beat the old flow: 0.70/10 on
  seed 67000 and 0.40-0.50/10 on seed 68000.
- BC-generated trajectories make the training loss tiny, but do not solve the
  future-conditioned replacement. BC-trial flow scored only 0.30-0.40/10 and had
  very low action delta, indicating mode averaging/smoothing. BC-trial inverse
  reached 0.60/10 deterministic and 0.70/10 with small-noise JEPA ranking. A
  horizon-matched h8 BC-trial inverse collapsed to 0.30/10 on seed 67000 and
  0.00/10 on seed 68000.
- The rollout-calibrated contact probe finally improved one slice: weight 0.1
  scored 0.80/10 on seed 68000, above the previous 0.60-0.70 local ceiling. It
  did not generalize: seed 67000 scored 0.40/10 and seed 69000 scored 0.50/10.
  This confirms that calibrating probes on predicted latents is necessary but
  not sufficient.
- The contact-trace-conditioned inverse also failed despite near-zero training
  loss: 0.20/10 on seed 67000 and 0.00/10 on seed 68000. The likely issue is
  retrieval/recovery, not local fitting: a retrieved contact trace from a nearby
  successful trajectory can still be unreachable from the current hand/object
  state after small deviations.
- The action-conditioned contact dynamics head is the best new direction but
  still not a replacement. Training on 80 all-outcome SSL-flow rollouts fit the
  observed contact traces well (`smooth_l1` reached about `1e-5`). It improved
  some 10-episode slices: 0.80/10 on seed 68000 and 0.90/10 on seed 69000 at
  weight 0.1; weight 0.2 lifted seed 67000 to 0.80/10 but hurt the other seeds.
  A 30-episode validation at weight 0.1 scored 0.567, below the retained BC and
  not enough to delete it.
- VAE assessment: the contact VAE is now wired as the stochastic version of the
  contact scorer. Full and small training runs were too slow in this environment
  before reaching useful checkpoints. A tiny smoke checkpoint
  (`relocate_contact_vae_tiny.pt`, 5k pairs, 500 steps, latent dim 4) evaluated
  end-to-end. With a deliberately cheap planner budget (`episodes=3`,
  `candidates=2`, `exec-k=8`, `flow-steps=1`, `contact-vae-samples=1`) it scored
  0.333 on seed 68000, while the same setting without the VAE scored 0.000.
  That is a positive integration signal but far below the retained 1.00 BC and
  weaker than the deterministic contact-dynamics branch. The next VAE step would
  need faster offline/cached scoring, a trained checkpoint rather than the tiny
  smoke model, and possibly a discrete contact-mode bottleneck before it can
  replace the deterministic scalar contact scorer.
- Staged contact-trust scoring was tried after a research pass over D-MPC,
  Contact-Grounded Policy, DexWM, and contact trust-region work. The scorer
  uses action-conditioned contact dynamics but changes the ranking energy by
  inferred mode: reach/grasp, transport while maintaining contact, then
  place/settle. It produced a promising 1.00/3 cheap diagnostic on seed 67000
  with `candidates=4`, `flow-steps=4`, `exec-k=4`, and no JEPA rollout scoring,
  but expanded to only 0.40/10 on the same seed. A hybrid with scalar
  contact/target terms and predicate future lookup also scored 0.40/5. This
  does not beat the deterministic contact-dynamics baseline. See
  `docs/HANDOFF_RELOCATE_SSL_CONTACT.md` for exact commands and the downstream
  plan.

Research check, 2026-07-07:

- Diffusion Model Predictive Control supports the factorization we are using:
  a multi-step action proposal plus runtime model-based planning. It also argues
  that multi-step dynamics/proposals reduce compounding error, but its benchmark
  evidence is mostly D4RL locomotion, not high-DoF hand contact.
- Implicit Contact Diffuser and Hierarchical Diffusion Policy both point to
  contact subgoals as the right abstraction for contact-rich tasks. The key
  difference from our failed trace attempt is that they plan over richer contact
  descriptors or phased objective contacts and explicitly track/remove contact
  subgoals, instead of retrieving a fixed trace from nearest state.
- DexWM-style dexterous world models emphasize fine-grained hand action
  representations and hand-consistency losses. This matches our finding that a
  compact state JEPA can encode Relocate but its imagined contact latents are not
  reliable enough for action optimization.

Session 2026-07-08 (planner-structure pass):

- **Eval nondeterminism found and fixed.** The eval scripts never seeded torch,
  so stochastic flow sampling made 10-episode numbers irreproducible: an exact
  reconstruction of the recorded 0.80/10 contact-dynamics config scored
  0.30/10. `--torch-seed` was added to both eval scripts. Prior 10-episode
  Relocate numbers should be read with roughly +/-0.2-0.3 run-to-run noise, and
  tuning-seed numbers (66000-69000) are additionally selection-inflated: the
  0.567/30 record was measured on tuning seed 67000.
- Possession trust barrier (`possession_trust_barrier` in
  `jepa_robotics/algos/contact.py`): mode-aware hard barrier on predicted
  contact traces instead of a weak additive term. Tuning aggregate 0.57/30 vs
  0.53/30 base - neutral. A hardened variant (barrier 30, keep 0.075, with
  demo-locked futures) reached 0.73/30 tuning but the flow stack still
  measured only ~0.53-0.58 on fresh seeds.
- DAgger-style contact-head recalibration on the planner's own rollouts
  (rounds 1 and 2, 180-280 pooled episodes): neutral on tuning seeds; fresh
  seed moved 0.53 -> 0.58/100. Not sufficient alone.
- Warm-start (re-noised previous plan) flow sampling: harmful, 0.23/30 tuning
  with memoryless retrieval (plan lock-in feedback loop); neutral-to-negative
  with demo-locked futures. Grasp-phase commitment (exec-k burst through the
  contact switch): neutral-to-harmful.
- Full-size contact CVAE trained in minutes on an idle H200 (the earlier "too
  slow" was environment load, not the code), but the KL collapsed (~1e-7), so
  it is effectively deterministic and was not pursued further.
- **Demo-locked future index worked** (`DemoLockedFutureIndex` in
  `jepa_robotics/algos/futures.py`): lock onto one demo at episode start,
  re-localize the current state within that demo every replan, and target the
  state `h` steps ahead. Futures become causally consecutive instead of
  jumping between demos, and after a drop the re-localization naturally falls
  back to the reach/grasp phase. Flow stack: 0.63/30 tuning vs 0.53 memoryless.
- **Deterministic inverse + demo-locked h8 tracking is the new best SSL
  controller.** `relocate_flat_inverse_h8_raw_strong.pt` with
  `--future-index demo_locked --target-horizon 8 --candidates 1 --exec-k 2`:
  0.87/30 tuning (0.8/0.9/0.9) and 0.67/90 fresh-seed
  (0.77/0.67/0.57 on seeds 75000/77000/78000, 30 episodes each). Adding
  sampling noise plus JEPA ranking on top is worse (0.67 tuning), and a
  DAgger inverse retrain on 280 pooled trial episodes is worse (0.67 tuning) -
  trial actions dilute the expert manifold, echoing the earlier self-imitation
  collapse.
- Failure anatomy (per-episode diagnostics, `--log-episodes`): all failures
  are possession failures - roughly even split between never grasping and
  dropping during transport; zero near-misses. When possession survives
  transport, placement succeeds.

Why this still does not replace the 1.00 BC: the inverse prior conditioned on
demo futures executes the demonstrated grasp/transport pattern, but has no
mechanism to correct marginal grasps - runtime scoring (JEPA rollout, contact
dynamics, barriers) does not reliably distinguish chunks that keep possession
from chunks that lose it, so the ~0.3 fresh-seed failure mass stays.

Session 2026-07-08 (evening follow-up, two closures):

- **Possession-gated future advance: neutral.** `DemoLockedFutureIndex` gained
  optional predicate gating (cap the future target at the demo's
  first-possession frame while the live palm-ball distance exceeds the
  threshold; `--future-possession-gate`). Tuning 0.9/0.9/0.7 (0.83/30), fresh
  seeds 75000/77000/78000 0.67/0.67/0.60 (0.64/90) - within noise of the
  ungated 0.67/90. Re-localization already keeps the matched index in the
  grasp phase while un-possessed, so explicit gating adds little.
- **Input-noise robustified inverse: harmful.** Retraining the inverse with
  Gaussian noise on the current-state input only (clean futures and action
  targets; `--input-noise-std` in `train_flat_future_inverse.py`) scored
  0.67/30 tuning at std 0.05 and 0.53/30 at std 0.10, vs 0.87/30 for the
  noise-free checkpoint. Interpretation: state noise spans several demo
  frames, so one chunk is forced to fit a neighborhood whose correct chunks
  differ - temporal smearing of the fast grasp closure. Together with the
  DAgger-inverse and self-imitation failures this is now a consistent law for
  Relocate: **any training-time blurring of the expert action manifold loses.**

Session 2026-07-08/09 (possession-specialist round):

- **Segment-pure dual-specialist inverse is the new best SSL controller:
  0.78/90 fresh-seed** (0.77/0.73/0.83 on seeds 75000/77000/78000), 0.93/30
  tuning with two perfect 10-episode slices. Two inverses trained with
  `--require-possession` in `train_flat_future_inverse.py`: a *free/reach*
  specialist (pairs whose current frame is not in possession, which includes
  the closure transitions) and a *held/transport* specialist (current and
  future frames both held). Runtime switches on the live palm-ball predicate
  (`--inverse-possession-path`, threshold 0.06). Both track the demo-locked
  h8 future index with `exec-k 2`. This confirms the "no blurring" law from
  the positive side: splitting the bimodal regime beats one global inverse
  (0.78 vs 0.67 fresh) with zero change to targets or futures.
- Failure anatomy after the split (seed 77000 diagnostics): 6/8 failures are
  marginal grasps with min palm-ball 0.066-0.109 - the hand parks just outside
  the possession boundary and closure never captures; 2/8 are drops, one at
  4.9 cm from the target.
- Refinements that did NOT move fresh-seed performance beyond noise:
  switch hysteresis 0.03 (0.74/90), geometry-weighted demo matching x3 on dims
  30:39 (0.78/90), re-lock-on-drop margin 0.5 (worse on tuning), a held
  specialist trained at threshold 0.09 with switch 0.075 (0.77/90), and
  4-candidate multi-demo futures ranked by the recalibrated contact-dynamics
  head (0.79/90). The multi-demo ranking machinery
  (`DemoLockedFutureIndex.query_topk` + `--demo-candidates`) is retained in
  the eval script but is not part of the canonical controller.

Session 2026-07-09 (closure micro-correction — palm-ball emphasis):

- **Palm-ball-emphasis reach specialist is the new best SSL controller:
  0.93/90 on untouched validation seeds** (0.90/1.00/0.90 on 80000/81000/82000,
  30 eps each), the biggest fresh-seed gain of the effort. The reach specialist
  was retrained with the live palm-ball vector (raw dims 30:33) duplicated 8x
  in its conditioning (`--emphasis-dims 30,33 --emphasis-repeat 8` in
  `train_flat_future_inverse.py`; eval replicates the suffix per-specialist from
  the checkpoint via `_append_emphasis`). The held specialist is unchanged. The
  diagnosis from the possession-specialist round was that marginal grasps came
  from the reach chunk tracking the *demo* ball geometry embedded in `z_future`,
  not the *live* ball; upweighting the live palm-ball input makes the closure
  chunk servo to the live ball. Zero change to action targets or futures, so the
  no-blurring and future-coherence laws both hold — this is input-feature
  emphasis, not target relabeling, future retrieval, or runtime action
  optimization.
- Same-seed comparison against the non-emphasis reach specialist: on the three
  untouched seeds, emphasis 0.93/90 vs 0.79/90; on the original fresh seeds
  75000/77000/78000, emphasis 0.84/90 (0.80/0.867/0.867) vs 0.78/90
  (0.77/0.73/0.83). Across all six fresh seeds, emphasis 0.89/180 vs 0.78/180.
  Every seed improved; none regressed, from a single untuned `--emphasis-repeat
  8`.
- Failure anatomy shifted (seed 77000: reach failures collapsed, 29/30 grasp;
  seed 82000 diagnostics, 3 failures/30): one still-marginal reach miss
  (min palm-ball 0.062, at the boundary), one wider reach miss (0.093), one
  early transport drop right after the reach->held switch. The post-switch drop
  is now a leading failure and is a held-specialist/switch-boundary problem, not
  a reach problem. See `docs/HANDOFF_RELOCATE_SSL_CONTACT.md` for the emphasis
  sweep / post-switch-drop next directions.
- Still below the retained BC 1.00 (gap ~0.07), so
  `adroit_relocate_bc_on_explorewm.pt` stays.

Prior most-promising directions from the possession-specialist round (a
grasp-conditioned low level; contact-consistency JEPA finetuning, DexWM-style
hand-consistency losses) remain open; the emphasis result is a cheaper win that
did not require either.

Session 2026-07-09 (continued — firm-switch + placement emphasis, gap to ~0.045):

- **New best SSL controller: 0.957/210 on held-out seeds (0.955/330 over 11
  seeds).** Two further root-cause fixes on top of the reach emphasis, each
  attacking the failure mode the previous fix exposed:
  - *Firmer 0.045 possession switch* removes transport drops. Retraining both
    specialists at possession threshold 0.045 and switching there (vs 0.06)
    lets the reach specialist keep tightening the grasp before handoff, so the
    transport specialist never inherits a marginal grasp it cannot maintain.
    Dev 4-seed 0.86 -> 0.95. This is the opposite polarity of the earlier
    harmful "leave" hysteresis: a firmer *enter* condition, not stickier
    possession. `exec-k=1` (0.82-0.85) and an `enter-delay` reach-hold
    (0.87-0.88) were both neutral-to-harmful, confirming the fix is the
    threshold, not the switching dynamics.
  - *Ball-target emphasis on the held specialist* fixes placement near-misses.
    Diagnostics after the drop fix revealed a distinct failure the grasp fixes
    had masked: episodes where the ball is held firmly the whole trajectory
    (170+ possession steps, palm-ball ~0.01-0.03) but delivered ~11 cm from
    the target. This is the placement analog of the reach miss - the held chunk
    tracks the *demo* target in `z_future`, not the *live* target. Duplicating
    the live ball-target vector (raw dims 36:39) 8x in the held specialist's
    conditioning (`--emphasis-dims 36,39`) servos placement to the live target.
    Held-out 7-seed 0.948 -> 0.957 (fixed placement on 81/82/88).
- Canonical checkpoints: `relocate_flat_inverse_h8_raw_free_emph_t045.pt`
  (reach, palm-ball emphasis, threshold 0.045) +
  `relocate_flat_inverse_h8_raw_held_bt_t045.pt` (held, ball-target emphasis,
  threshold 0.045), switch 0.045. Full progression this task: single global
  inverse 0.67 -> segment-pure split 0.78 -> reach emphasis 0.93 -> firm switch
  -> placement emphasis 0.957 held-out.
- This generalizes law 3: input-feature emphasis fixes live-vs-demo offsets for
  *placement* as well as *grasp*, both by upweighting the relevant live
  relative-geometry vector with zero target/future modification. Adds a control
  law: switch to the transport specialist only on a firm grasp.
- Still below the retained BC 1.00 (gap ~0.045); remaining failures are a
  diverse long tail (residual reach misses on outlier ball positions, residual
  placement offset, rare mid-transport drops). `adroit_relocate_bc_on_explorewm.pt`
  stays. Cheap planner/emphasis knobs are saturated; see
  `docs/HANDOFF_RELOCATE_SSL_CONTACT.md` for next directions (emphasis/threshold
  micro-sweep with strict held-out discipline, wider demo bank for reach
  outliers, contact-consistency JEPA finetune).

## AntMaze And PointMaze

### PointMaze

- Direction: H-JEPA graph plus SSL inverse low level.
- Result: solved on checked UMaze/Medium/Large runs, 1.00/1.00/1.00.

Why it works: the point agent has simple dynamics, so a future-conditioned
inverse low level can reliably move between subgoals. The graph handles walls
and sparse reward by planning through empirically reachable landmarks.

**Dijkstra retired (2026-07-10):** migrated to the same flow-macro HWM + directed
flow walker paradigm as AntMaze (no demo npz existed, so demos were collected via
the scripted maze controller: `collect_episodes(scripted_fraction=0.95)`,
~1.5k-3.4k eps/maze). Flow-macro HWM eval: **UMaze/Medium/Large 1.00/1.00/1.00**
(seeds 30000/31000, 20 eps), matching the legacy graph+inverse. So no maze task
in the repo uses the Dijkstra graph anymore -- all mazes run the neural
flow-macro HWM high level. Checkpoints per maze:
`point_{umaze,medium,large}_{flow_directed,hwm_s*,hwm_macroflow}.pt`.

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

Session 2026-07-10 (flow-walker goal-delta emphasis + directed-motion data):

Architecture = the user's target: **H-JEPA subgoal graph high level + rectified
flow-matching low-level walker** (`--low-type flow`), with two Relocate-style
upgrades to the walker.

- **Goal-delta emphasis (law 3, locomotion analog).** Duplicate the
  **(desired_goal - achieved_goal) xy vector** N x in the flow conditioning
  (`train_flow_walker.py --emphasis-repeat`, applied in `eval_hjepa_maze.py`
  `LowLevelFlow`) -- the servo DIRECTION, analogous to the palm-ball vector, so
  the sampled gait chunk heads at the live subgoal instead of averaging over HER
  directions. UMaze A/B (matched 100k walkers): FLAT 0.40 -> 0.60, H-JEPA
  0.267 -> 0.333.
- **Directed-motion data (law 1, no blurring, for a gait).** Default HER
  relabels to far/arbitrary goals and mixes in wandering/standing segments, so
  the walker learns a slow, undirected gait -- the AntMaze bottleneck.
  `build_directed_chunks` (`--directed --max-relabel-h --min-progress`) relabels
  the goal to a NEARBY future achieved position and keeps a sample only if the
  ant closes >= `min_progress` xy distance to it over the chunk. This de-blurs
  the walker toward decisive locomotion and, combined with the graph, is what
  finally makes Medium non-zero.
- **Results (H-JEPA graph, directed walker `antmaze_medium_flow_directed.pt`,
  reach-radius 1.0 / subgoal-timeout 90 / landmarks 40):**
  - **UMaze: 0.85 / 60 eps** (0.90/0.90/0.75 on seeds 30000/31000/32000), up
    from 0.33 with the non-directed emphasis walker; near the historical ~0.93.
  - **Medium: 0.26 / 80 eps** (0.40/0.20/0.20/0.25 on seeds 30000-33000), up
    from the reproducible 0.00. First reproducible Medium success in the current
    env, below historical HIQL (~0.77).
- An even more aggressive "fast" filter (`--min-progress 0.35 --max-relabel-h
  24`) was WORSE on Medium (0.125/80) -- too little/too-narrow data. The
  directed walker (`--min-progress 0.2 --max-relabel-h 40`) is the keeper.
  Denser graph (landmarks 80) and graph-param retunes were within noise.
- Diagnosis of the remaining Medium gap: the walker locomotes decisively now but
  its top xy speed is still bounded by the (wandering) demos, so far-goal
  episodes exhaust the 1000-step budget before the graph path completes -- the
  ~0.26 is dominated by near/mid goals succeeding and far goals timing out. The
  ant walker's raw speed remains the limiter; the graph and emphasis are not the
  bottleneck. Next: a faster gait (better demos / a speed-shaped objective / a
  stronger AntMaze JEPA), then re-test on Medium/Large.
- Faster-gait push (all negative — the SSL walker is at its imitation-speed
  ceiling): (a) **chunk-16** directed walker collapsed on Medium (0.00-0.10/20 vs
  chunk-8's 0.39) — a 16-step open-loop chunk overshoots a nearby subgoal;
  (b) **hard fast-filter** (min_progress 0.35) → 0.125/80 — starves the turning
  segments; (c) **speed-weighted sampling** (min_progress 0.1 + p~progress^2, so
  turns are kept but fast segments are upweighted) → 0.34/80, within noise of the
  uniform directed walker's 0.39. All three confirm the walker cannot exceed the
  wandering D4RL demos' top speed by reweighting/filtering imitation data. The
  high level is solved (flow-macro beats the graph), so the residual Medium/Large
  gap to HIQL (~0.77) needs a genuinely faster gait SOURCE — a goal-conditioned
  RL fine-tune of the walker (the one lever not yet pulled) — not another
  imitation-data knob. chunk-8 uniform directed walker is retained as the SSL
  ceiling.

HWM neural high level (arXiv:2604.03208, "Hierarchical Planning with Latent
World Models"), 2026-07-10:

- Trained the repo's HWM stack (`train_hjepa_hwm.py`: HighEncoder psi +
  MacroEncoder GRU + residual MacroPredictor g with VICReg/stop-grad +
  SubgoalDecoder) on the Medium demos at stride 40. It fits the paper's claim of
  generalizing macro-dynamics -- **holdout next-macro-latent error ~0.0018**,
  decoder ~0.0015.
- Ran it as the paper prescribes: **CEM over K macro-actions rolled through g,
  scored by decoded terminal position vs goal**, first macro decoded to an xy
  subgoal, on top of the (unchanged) directed flow walker, with reach-based
  replanning (`eval_hjepa_hwm.py --low-type flow`, K in 1/2/4, reach-radius 1.0,
  low-timeout 90).
- **Result: 0.00-0.067/15 on Medium -- WORSE than the empirical Dijkstra graph
  (0.26).** Failure mode is exactly the documented one: the decoded-position CEM
  has no wall model, so it proposes macro-actions whose decoded subgoal heads
  straight at the goal THROUGH walls; the walker drives into the wall and stalls.
  K=1 (single feasible hop, minimal compounding) did not help (0.067). The
  macro-predictor generalizes (low holdout error) but planning against it
  hallucinates wall-crossing feasibility.
- On a SINGLE known maze the empirical graph's stored wall-aware reachability
  beats the pure-latent Gaussian-CEM HWM. The fix is NOT to fall back to the
  graph but to constrain the macro search to the feasible manifold -- see below.

**Flow macro-action prior (the decisive fix), 2026-07-10:** replace the
Gaussian-CEM macro sampler with a **rectified-flow prior over macro-actions**
conditioned on `(z_high, goal_xy)` (`train_hwm_macro_flow.py`;
`eval_hjepa_hwm.py --macro-flow`). Trained on the frozen HWM's
`(psi(z_t), achieved_goal_{t+k*N}, macro_encoder(a_{t:t+N}))` tuples, so every
sampled macro is an ON-MANIFOLD demonstrated transition. Plan = sample N macros
from the flow, roll each through the frozen g, decode the next subgoal, pick the
one closest to the goal. Because demos never cross walls, the sampled macros
never propose wall-crossing subgoals -- the exact failure Gaussian CEM had.

- **Result (directed flow walker low level, 20 eps/seed, seeds 30000-33000):**
  - Gaussian CEM (paper as-is): 0.067
  - Dijkstra empirical graph: 0.26 (mean over the same seeds)
  - **Flow macro-prior, n=16: 0.39 / 80** (0.55/0.30/0.40/0.30); n=32: 0.34/80.
  The flow high level **beats both the vanilla paper's CEM (6x) and the
  empirical graph** on Medium, and is a fully neural high level (no stored
  reachability table -> generalizes to unseen layouts, the paper's win regime).
- This confirms the flow-on-the-manifold theme one level up: the reason the
  neural high level "hallucinated wall-crossing feasibility" was the GLOBAL
  Gaussian macro proposal, not the world model g -- constrain the proposal to
  the demonstrated macro distribution with a flow prior and the neural high
  level becomes the best AntMaze high level in the repo. **Canonical AntMaze
  controller (ALL three mazes) is now HWM + flow macro-prior + directed flow
  walker, replacing the Dijkstra graph:** UMaze 0.867/60, Medium 0.39/80,
  Large 0.18/60. Per-maze checkpoints `antmaze_{umaze,medium,large}_hwm_s40.pt`
  (psi/macro/g/dec) + `..._hwm_macroflow.pt` (macro flow) +
  `..._flow_directed.pt` (walker); eval `eval_hjepa_hwm.py --low-type flow
  --macro-flow ... --macro-flow-horizon 1 --reach-radius 1.0 --low-timeout 90`.
  Remaining gap to HIQL (~0.77 Medium) is still the walker's gait speed on
  far-goal episodes.

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

### SOLVED 2026-07-10: subtask-specialist SSL controller (Relocate recipe port)

Direction: port the four Relocate laws to the sequential task.
`scripts/train_kitchen_subtask_inverse.py` +
`scripts/eval_kitchen_subtask_inverse.py`.

- **Law 1 (no blurring) -> segment-pure per-subtask specialists.** Four inverse
  priors, each trained ONLY on transitions the labeler
  (`kitchen_labeled_v4.npz`, 1233 eps, per-transition subtask one-hot) tagged as
  working toward that subtask. One global policy blurs four action manifolds and
  stalls; splitting them is the base win.
- **Law 2 (future coherence) -> per-subtask demo-locked future index.** Each
  specialist stores its own demo *segments* and tracks a `DemoLockedFutureIndex`
  h8 ahead within them.
- **Law 4 (firm predicate switch) -> ground-truth completion scheduler.** A
  trivial fixed-order high level advances to the next specialist only when the
  env reports the current subtask complete (`info['step_task_completions']`).
  Each subtask becomes a fresh short-horizon problem.
- **Law 3 refined (feature-matched emphasis).** Emphasize the feature carrying
  that subtask's dominant servo error, NOT blindly the object dim. The object
  qpos is near-constant until the object is manipulated, so for the
  approach-dominated **light switch** the useful emphasis is the **robot arm
  joints (obs 0:9)** (they determine EE reachability of the fixed switch);
  for the manipulation-dominated microwave/kettle/slide the **object qpos dims**
  are right. Same-seed: light with object emphasis 0.55, light with arm emphasis
  0.65; all-arm 0.50 (arm emphasis hurts the manipulation subtasks). This
  extends law 3: emphasis dim = the live feature whose error the chunk must
  null, which is task-phase dependent.
- **New lever, decisive: arm-pose demo matching + re-lock.** The demo-locked
  index must key on the reachability-determining feature. Matching on the
  near-constant object dim (`--geom-weight 3` on obj) scored 0.40; matching on
  the **arm joints** (`--match-dims 0,9 --geom-weight 3`) scored 0.55; adding
  **`--relock-margin 0.3`** (re-lock to a better-matching demo when the arm
  deviates from the locked one) scored **0.80** — recovering the bad
  kettle->light handoff locks that were the dominant failure. This is the
  Relocate "re-localization after a drop" idea applied to a subtask handoff.

Obs layout note (cost a retrain): the flat 59-D obs is
`[robot_obs(18), obj_qpos(21), obj_qvel(20)]`, and `OBS_ELEMENT_INDICES` are
into the full qpos, so an object at qpos index j lands at **obs index 9+j**
(microwave 31, kettle 32:39, light switch 26:28, slide 28). The first run used
the raw qpos indices by mistake.

Neutral-or-harmful: uniform demo matching (`--geom-weight 1`, 0.10-0.30);
object-dim matching (0.40); patience-based subtask rotation (budget-limited,
0.00-0.45; low patience disrupts working subtasks); slide-before-light ordering
(`--order 0,1,3,2`, 0.00 full-4 — light is *worse* approached after slide);
all-subtask arm emphasis (0.50); exec-k=1 (0.10-0.30).

Result: **0.813 full-4 / 150 eps** (6 seeds 25 eps:
0.80/0.84/0.76/0.76/0.92/0.80; held-out 5-seed 0.816/125), mean ~3.63/4, vs
reproducible raw-flow baseline 0.00 full-4 / 2.05 mean. First reproducible
full-4 success in the repo. Canonical config:
`--match-dims 0,9 --geom-weight 3.0 --relock-margin 0.3 --exec-k 2
--target-horizon 8`, specialists mw/kettle/slide = object emphasis, light =
arm emphasis. See `docs/HANDOFF_KITCHEN_SUBTASK_SSL.md`.

Two follow-on findings (2026-07-10 push):

- **Wider light demo bank: 0.773 -> 0.813** (`kitchen_subtask_light_arm_wide.pt`,
  `--max-segments 800 --seg-pad 4 --emphasis-repeat 12`). The residual failures
  were all light-switch approach misses from post-kettle poses the 400-segment
  bank covered poorly; doubling the bank (relocate next-direction #2) covers
  more of them. Every seed matched or improved.
- **Task order is NOT free for this controller (order-agnostic greedy loses).**
  A greedy scheduler that at every (re)selection picks the most-reachable
  uncompleted subtask by demo-match distance (arm-pose slice) scored only
  0.25-0.40 vs 0.75-0.85 for the fixed canonical order. The demos encode strong
  physical dependencies -- the light switch is only reliably reachable from the
  post-kettle arm pose, so doing slide/light out of the demonstrated order ruins
  the approach (light-after-slide is ~0.00). The canonical order that matches
  the dominant demo order is near-optimal; free ordering is limited by demo
  coverage, not by the scheduler. `--scheduler greedy` is retained for the
  record but is not the canonical controller.

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
