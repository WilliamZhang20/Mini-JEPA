# JEPA Mini Robotics

Small action-conditioned JEPA (Joint-Embedding Predictive Architecture)
experiments for Gymnasium Robotics. The question: **how far can a compact
self-supervised latent world model go on real robot-control tasks without
becoming a giant foundation model?** Every world model here is ≤ ~6 M parameters
and trains on low-dimensional state.

It started as three Fetch tasks and now spans four difficulty tiers, from
table-top manipulation to long-horizon maze navigation and a 28-DoF dexterous
hand:

| Tier | Task(s) | Best agent | Success |
| --- | --- | --- | ---: |
| base | FetchReach / Push / PickAndPlace | JEPA MPC / flow-prior+JEPA selection / inverse-prior+JEPA selection | 1.00 / 1.00 / 1.00 |
| 1 | **FetchSlide** (ballistic strike) | goal-frame equivariant ballistic JEPA HWM | **0.853/2000** |
| 2 | **PointMaze** UMaze / Medium / Large | **H-JEPA + inverse low level** | 1.00 / 1.00 / 1.00 |
| 2 | **AntMaze** UMaze / Medium / Large | H-JEPA + unified conditional flow | 1.00 / 0.775 / 0.533 |
| 3 | **Adroit** Door / Hammer / Pen / Relocate | SSL future-conditioned specialists | 1.00 / 1.00 / 0.90 / 0.957 |
| 4 | **FrankaKitchen** | demo-handoff graph + live-relocking inverse specialists + learned completion | 0.80/100 reordered four-task multi-seed check; full-7 remains open |

The repo is deliberately organized around self-supervised world models rather
than reward-optimized agents. Demos/trials specify desirable futures and transition
evidence, action priors propose chunks conditioned on `(z_t, z_future)`, and JEPA
dynamics selects or verifies the chunk that realizes the latent future. The experiments
below map out *which* controller class wins where, and why. Two findings recur
and are documented in
["What the world model is good for"](#what-the-world-model-is-good-for):

1. **JEPA's encoder carries control; its predictor carries planning — but only on
   smooth dynamics.** Planning through the learned predictor works on smooth tasks
   (Fetch reach/push, and the *high level* of the maze hierarchy) and *fails* on
   contact-rich ones (slide, Adroit) due to model exploitation.
2. **Long-horizon tasks need hierarchy, not a longer flat plan.** A two-level
   Hierarchical JEPA beats flat goal-conditioned control on every maze tested.

## What Is Inside

- `jepa_robotics/train.py`: collects trajectories, trains the JEPA world model,
  and writes a checkpoint/model artifact.
- `jepa_robotics/train_policy.py`: historical behaviour-cloned goal-conditioned
  action prior on the frozen JEPA latent. Kept for baselines and tasks not yet
  replaced by SSL action-prior control.
- `jepa_robotics/evaluate.py`: compares random actions, a scripted controller,
  the learned policy on its own, and the policy-seeded JEPA+MPC planner. It can
  also record MP4 rollouts.
- `scripts/train_fetch_flow_prior.py` / `scripts/eval_fetch_flow_jepa.py`: FetchPush's
  LeCun-compatible replacement controller: a conditional flow proposes action
  chunks from `(z_t, z_future)`, and JEPA dynamics selects the chunk that realizes
  the encoded demo/goal future.
- `scripts/train_fetch_inverse_prior.py` / `scripts/eval_fetch_inverse_jepa.py`:
  FetchPickAndPlace's SSL inverse-chunk replacement: train
  `inverse(z_t, z_future) -> a_{t:t+H-1}` from trial transitions and rank noisy
  candidates through the JEPA world model. The same inverse-prior artifact can
  now serve as an H-JEPA low level under `scripts/eval/eval_hjepa_hwm.py`.
- `scripts/train_flat_future_inverse.py` / `scripts/eval_flat_future_inverse.py`:
  flat-task Adroit experiment: retrieve a nearby demo future state, condition the
  inverse prior on its latent, and execute/rank chunks without copying the demo
  action label at runtime.
- `scripts/train_flat_future_flow.py` / `scripts/eval_flat_future_flow.py`:
  flat-task flow version of the same idea. It samples action chunks from
  `flow(a_chunk | z_t, z_future)` and optionally appends normalized raw
  current/future states for high-DoF proprioceptive precision. Maze low levels
  use the same inverse/flow family under `scripts/eval/eval_hjepa_hwm.py`.
- `scripts/train_phase_future_inverse.py` / `scripts/eval_phase_future_inverse.py`:
  hierarchical Adroit variant. Demonstrations are split into self-supervised
  progress phases, the high level advances a phase/subgoal schedule, and the low
  level predicts inverse chunks from `(z_t, z_future, phase_t, phase_future)`.
- `scripts/train_phase_future_inverse_fast.py`: cached/vectorized version of the
  phase inverse trainer. It batches latent encoding across all demonstrations
  before optimization, avoiding the per-episode encoder bottleneck on large
  Adroit datasets.
- `jepa_robotics/algos/`: shared SSL-control utilities factored out of scripts.
  `algos.phase` owns self-supervised phase features and phase-constrained future
  indexing; `algos.priors` owns reusable inverse and flow/diffusion action-prior
  networks; `algos.hwm` owns same-latent macro-action HWM components.
- `jepa_robotics/algos/world_models/ballistic.py` with
  `scripts/{data,train,eval}/*slide_ballistic_hwm.py`: FetchSlide's
  event-conditioned hierarchy. It learns the absorbing post-coast latent and
  endpoint from a pre-impact latent plus strike macro, then commits without
  receding-horizon replanning.
- `scripts/train_latent_hwm.py` / `scripts/eval_latent_hwm.py`: HWM-style
  same-latent hierarchy attempt inspired by arXiv:2604.03208. A high-level
  macro world model predicts future JEPA latents from learned macro-actions; the
  predicted latent subgoal is passed to a future-conditioned inverse low level.
- `scripts/train_flow_residual_refiner.py` / `scripts/eval_flow_residual_refiner.py`:
  coupled flow+diffusion experiment. Flow proposes global future-conditioned
  action chunks; a residual diffusion/flow refiner predicts local corrections
  conditioned on `(z_t, z_future, a_flow)`, and JEPA/state scoring gates refined
  vs unrefined candidates.
- `scripts/train_relocate_contact_probe.py`: self-supervised Relocate probe
  predicting palm-ball and ball-target distances from frozen JEPA latents. This
  is a fine-state scoring attempt, not an action-label policy.
- `scripts/collect_flat_future_flow.py` / `scripts/collect_adroit_bc_rollouts.py`:
  data-generation utilities for SSL replacement attempts. They save successful
  trajectories so future-conditioned flow/inverse priors can train on transition
  evidence; runtime evaluation still uses the SSL prior, not the collector
  policy.
- `scripts/train_relocate_rollout_contact_probe.py`: trains the Relocate
  fine-state probe on JEPA-predicted rollout latents rather than true encoder
  latents.
- `scripts/train_relocate_contact_dynamics.py`: trains an action-conditioned
  contact dynamics head on SSL rollouts, predicting palm-ball and ball-target
  traces directly from `(z_t, raw_t, action_chunk)`.
- `scripts/train_relocate_contact_vae.py`: trains a self-supervised conditional
  VAE over Relocate contact traces,
  `p(contact_trace | z_t, raw_t, action_chunk)`. Evaluation can load it through
  `scripts/eval_flow_residual_refiner.py --relocate-contact-vae-path` to score
  candidate chunks by sampled contact-mode futures. This is a contact/reachability
  scorer, not a BC action policy.
- `scripts/train_relocate_contact_trace_inverse.py` /
  `scripts/eval_relocate_contact_trace_inverse.py`: contact-trace-conditioned
  inverse prior. The high level retrieves a demo contact trajectory and the low
  level predicts an action chunk from `(z_t, z_future, raw_t, raw_future,
  contact_trace)`.
- `jepa_robotics/models/`: the action-conditioned JEPA model (recurrent latent
  dynamics, optional K-head ensemble) and the `GoalConditionedPolicy` action
  prior, split into `world_model.py` / `policy.py` / `mlp.py` / `regularizers.py`.
- `jepa_robotics/algos/planning/objectives/`: per-task MPC objectives
  (`manip`/`strike`/`goal`); `jepa_robotics/scoring/` is a compatibility facade.
- `jepa_robotics/data.py`: trajectory collection, scripted experts
  (reach/push/pick/slide/maze), offline-npz loading, and normalization.
- `jepa_robotics/envs.py`: Gymnasium Robotics registration, observation
  flattening, maze/AntMaze goal-env handling.
- `jepa_robotics/tasks.py`: task presets for Fetch, FetchSlide, PointMaze,
  AntMaze, and the Adroit suite.
- `scripts/{data,train,eval}/`: categorized CLI entry points; legacy root names
  are compatibility launchers during migration. `scripts/` also contains Slurm
  entry points, the `train_eval_object_v2.sh` pipeline, the
  Hierarchical-JEPA evaluator (`eval/eval_hjepa_hwm.py`), the offline-demo adapter
  (`minari_to_npz.py`), world-model accuracy probe (`eval_wm_rollout.py`), and
  video recorders.
- `docs/`: architecture/status notes and the experiment ledger for failed or
  unresolved directions.

Experiment outputs are intentionally ignored by Git. Checkpoints, videos, logs,
and JSONL eval files are written under `runs/` by default.

## SSL Controller Status

Controller learning uses self-supervised, future-conditioned action planning in
latent space:

| Task | Current controller |
| --- | --- |
| FetchPush | Flow prior + JEPA chunk selection: 1.00 over 30 episodes, mean final distance 0.008. |
| FetchPickAndPlace | Inverse prior + JEPA chunk selection: 1.00 over 30 episodes, mean final distance 0.011. |
| FetchSlide | Goal-frame equivariant event HWM: 0.848 and 0.857 on independent 1000-episode blocks (0.853/2000 aggregate). |
| PointMaze | HWM flow-macro high level + inverse/flow low level: UMaze/Medium/Large = 1.00/1.00/1.00. |
| AntMaze | HWM flow-macro high level + unified condition-modulated flow walker; see the checked per-layout results below. |
| Adroit | Door 1.00/30, Hammer 1.00/30, Pen 0.90/30, Relocate 0.957/210 with future-conditioned specialists. |
| FrankaKitchen | Demo-handoff graph + live-relocking inverse specialists + learned completion: reordered four-task 0.80/100 over four fresh seeds; full-seven remains open. |

## A World Model Is Not A Controller

The manipulation tasks taught the central lesson of this repo. Three things had
to be true to match a conventional scripted controller, in order of leverage:

1. **Data quality dominates.** The world model only learns dynamics it sees. The
   original scripted experts succeeded ~7% (pick) / ~3% (push), so the data
   almost never contained a real grasp or push and no planner could recover the
   skill. The rewritten experts in `data.py` solve their tasks ~100% (run
   `python scripts/check_experts.py`), which is what makes the collected data
   contain grasps and pushes in the first place.

2. **The JEPA world model is an accurate predictor, not a controller.** After
   training on good data, the model predicts a grasp-and-lift to within ~6 mm
   over 16 steps. But sampling-based MPC (CEM) with a hand-shaped distance cost
   still could not *discover* the grasp — a precise, temporally-extended action
   is a needle in action-sequence space, and the object-to-goal cost is flat
   until the object is already grasped. Cost shaping alone plateaued near 40%.

3. **The controller needs its own self-supervision.** Early runs used a small
   BC `GoalConditionedPolicy` on the frozen JEPA latent. The current replacement
   is stricter: train future-conditioned action priors from transitions rather
   than copying `state -> action` labels. Push uses a flow prior; PickAndPlace
   uses an inverse chunk prior. JEPA remains the latent world model that scores
   whether a proposed action chunk realizes the future.

The payoff (FetchPickAndPlace, 30 episodes): the old BC policy is gone; the
inverse prior plus JEPA chunk selection matches it at 1.00 success and reaches
mean final distance 0.011. See [Results Snapshot](#results-snapshot).

## Beyond Fetch: Tiers 1–3

The same compact-JEPA recipe scales to much harder tasks, each requiring one new
ingredient on top of the world model.

### Tier 1 — FetchSlide (ballistic strike): 0.853

The gripper is locked and the goal is *out of reach*, so the arm must impart the
right momentum in one strike and then watch. Receding-horizon MPC is
structurally wrong for this ("commit then watch"), and flat CEM scored 0.00.
The new **goal-frame equivariant ballistic HWM** plans at the contact timescale:
it canonicalizes contact geometry and strike direction around the live
puck-to-goal axis, expands those features with Fourier terms, and predicts the
absorbing post-coast endpoint with a gated residual ensemble. It chooses once
and does not replan during the coast. Transition trials and on-policy
recalibration use no reward, value, critic, or policy gradients. Two independent
1000-episode blocks score **0.848 and 0.857 (0.853/2000 aggregate)**, up from the
0.735 coordinate-MLP HWM and the 0.40 scripted expert.

### Tier 2 — Mazes: Hierarchical JEPA

Long mazes break flat goal-conditioned control: a straight-line-to-goal policy
walks into walls because it cannot represent route topology.

**Hierarchical JEPA (`scripts/eval/eval_hjepa_hwm.py`)** splits control into two timescales:

- **Low level** — a self-supervised inverse or directed-flow controller,
  `inverse(z_t, z_subgoal) -> a_{t:t+H-1}`, trained from transition trials.
- **High level** — a neural HWM predicts abstract macro futures. A conditional
  flow proposes only demonstrated, wall-feasible macro transitions; the HWM
  predicts their endpoints and selects the next subgoal. No stored graph or
  Dijkstra route is used.

AntMaze uses one condition-modulated residual flow for both gait and
self-righting behavior. The self-righting trajectories are mixed into ordinary
training with randomized route goals, so the model infers the useful action
mode from its JEPA/state condition. There is no posture threshold, named
recovery state, or specialist switch at runtime.

| Maze | **Hierarchical JEPA** | Seed blocks |
| --- | ---: | --- |
| PointMaze UMaze / Medium / Large | **1.00 / 1.00 / 1.00** | checked per layout |
| AntMaze UMaze | **1.00** | 1.00 / 1.00 / 1.00 (60 episodes) |
| AntMaze Medium | **0.775** | 0.75 / 0.75 / 0.80 / 0.80 (80 episodes) |
| AntMaze Large | **0.533** | 0.55 / 0.45 / 0.60 (60 episodes) |

The flow-macro hierarchy solves route topology; remaining Medium/Large failures
are long-horizon locomotion stalls. The narrow block ranges are from fully
seeded environment and flow sampling, rather than a selected rollout.

### Tier 3 — Adroit dexterous hand (24–30-DoF): mixed

The Adroit hand breaks the scripted-expert data engine — there is no simple
geometric controller for finger coordination, and the observation is flat (no
goal). The data source is offline D4RL expert demonstrations (via
[Minari](https://minari.farama.org/)),
behaviour-cloned on the frozen JEPA latent
(`minari_to_npz.py` → `train_policy.py --episodes-npz`). The current SSL
transition treats those demos as desirable future states instead of action labels
where possible.

| Adroit task | best checked controller |
| --- | ---: |
| Door | **1.00 schedule-phase inverse** over 30 |
| Hammer | **1.00 schedule-phase inverse** over 30 |
| Pen (in-hand reorientation) | **0.90 raw+latent future flow** over 30 |
| Relocate | **0.957 dual possession-specialist inverse** over 210 held-out episodes |

Notably the old BC controllers and the new Pen flow prior run on the *same*
exploratory world model (trained on random data). The encoder is
information-preserving enough (its state-probe loss forces the latent to
reconstruct the full state) that demos can define either action policies or
future latents. The open problem is not representation; it is turning
contact-rich high-DoF demo futures into reliable latent-space control without
falling back to copied action labels. Door and Hammer now do this with
phase-scheduled inverse priors; Relocate now uses the SSL dual-specialist
replacement.

2026-07-07 Relocate follow-up: a phase-conditioned inverse prior was trained in
two forms. A 120-demo smoke model reached only 0.10/10 with `exec-k=1`; a
full-demo cached model (`train_phase_future_inverse_fast.py`, 222k pairs,
60k-bank, hidden 128) reached 0.00/10 for both `exec-k=1` and `exec-k=4`.
This points away from generic temporal phases for Relocate. The next plausible
direction is object/contact-aware structure: explicit grasp/transport/place
latent predicates or a stronger low-level contact prior, not more nearest-future
phase scheduling.

Second 2026-07-07 orthogonal sweep:
Relocate horizon/scale variants stayed at 0.00/3 (`h8` at action-scale 0.8,
`h16` at action-scale 1.2). Later contact-aware scorers improved some short
slices, but the best 30-episode contact dynamics run was 0.567 and a tiny
contact-CVAE scorer is only a smoke-test branch so far. Kitchen raw flow with
more commitment regressed to
1.40/4 over 5 episodes; Kitchen flat inverse and tiny same-latent HWM both
scored 0. Hammer phase inverse p4 later validated at 1.00/30, replacing the old
BC artifact; p6 was weaker. AntMaze UMaze showed
that hierarchy should be conditional: the SSL inverse low level reached 1.00/2
when pointed directly at the goal, while the relaxed H-JEPA graph reached 0.50/2.
AntMaze Medium remains limited by low-level locomotion reliability.

See `docs/EXPERIMENT_LEDGER.md` for the tried axes, negative results, and the
current remaining gaps.

### Tier 4 — FrankaKitchen: reordered four-task control and all seven options

A 9-DoF arm must complete an ordered chain of microwave, kettle, light switch,
and slide cabinet. Flat single-step and long open-loop variants scored zero full
sequences because the failure is chaining. The reproducible controller uses
four segment-pure future-conditioned inverse specialists, demo-locked future
tracking, feature-matched emphasis, and a firm high-level switch. With the
environment completion event it reaches 0.813/150 full-4. Replacing that runtime
oracle with a frozen JEPA+raw completion probe trained from aligned demo replay
retains **0.784/125 full-4 and 3.58/4 mean tasks**. The newer high level no
longer assumes the supplied order: it learns task-to-task handoff costs from
specialist demo boundaries and solves the minimum-cost route over the requested
set. A deliberately scrambled four-task request scores **0.80/100** across four
fresh seed blocks (0.88/0.76/0.80/0.76). Specialists may re-lock to a different
demonstrated trajectory only when it matches the live handoff state materially
better. The environment’s full seven-task vocabulary
now has specialists and completion heads. A fresh three-seed validation with a
task-count-scaled 490-step budget scores **0.017/60 complete and 4.22/7 mean**
(0.05/0.00/0.00 blocks). This is API and coverage generalization, not a claim
that the seven-task chain is solved—the hinge cabinet remains the dominant
failure.

The older flow/self-imitation progression below is retained as experiment
history, not as the current result; its reported 0.90 checkpoint was not
reproducible in the current environment.

The still-useful recipe and its historical logged progression:

| step | what | full-4 |
| --- | --- | ---: |
| flat single-step baselines | single-step control on the latent | 0.00 |
| **control-aware encoder** + **action-chunked flow** policy | inverse-dynamics head keeps contact detail; chunked flow on `[raw ⊕ latent]` stops compounding/averaging | 0.00 (1.86/4 tasks) |
| **+ subtask skill-hierarchy** | label demos by sub-task (env-replay), condition the flow *skill* on a target one-hot, drive with a next-incomplete scheduler | 0.28 |
| **+ self-imitation / DAgger** ×2 (offline) | harvest the policy's own full-4 successes (19 expert demos → 500+), retrain | 0.68 |
| **+ online self-imitation fine-tuning** | warm-start, collect successes, fine-tune, iterate | **0.90** |

1. **Control-aware JEPA encoder.** A plain VICReg/predictive latent *trails raw obs*
   on contact control (it smooths away fine detail). Adding an **inverse-dynamics head**
   (predict `aₜ` from `zₜ, zₜ₊₁`, `--inverse-dynamics`) forces the latent to keep the
   action-discriminative detail — lifting JEPA from 1.37 (worse than raw) to 1.86 (ties raw).
2. **Action-chunked flow-matching policy** on `[raw ⊕ JEPA latent]`. Chunking kills
   per-step compounding error; flow denoising represents the *multimodal* demos instead
   of averaging them to mush (and samples ~10× cheaper than diffusion at equal quality).
3. **Subtask skill-hierarchy.** Decomposing the 4-task chain so each sub-task is a fresh
   short-horizon problem for the strong flow skill is the unlock (0 → 0.28 full-4).
4. **Self-imitation (DAgger).** The policy already completes all 4 tasks sometimes; its
   own successful trajectories are exactly the scarce full-sequence data the 19 expert
   demos lacked. Two offline rounds: full-4 **0.28 → 0.57 → 0.68**.
5. **Online self-imitation fine-tuning.** Warm-start the policy, collect fresh successes,
   fine-tune, iterate — full-4 **0.68 → 0.81 → 0.87 → 0.90**.

Historical logged final: **3.88/4 sub-tasks on average, 0.90 full 4-task success**
(4 seeds), entirely JEPA-based. Treat this as an unreproduced log until the
hierarchy is repaired.

**Behaviour analysis** (`scripts/analyze_kitchen_behavior.py`) pinpoints the bottleneck: the **3rd
task, the light switch**. Microwave and kettle are always ~1.0; the early policy flipped the switch
only 0.33 of the time and stalled there, and since the policy runs a fixed
microwave→kettle→light-switch→slide-cabinet order, raising switch-completion to 0.88 cascaded into
full sequences. Filtered self-imitation uses an accumulating success buffer to
improve skill reliability without a reward-optimized controller.

*(Metric note: these are mean-subtask and full-sequence rates on the D4RL partial+complete offline
demos; "online fine-tuning" here means the policy's own successful rollouts. Published
FrankaKitchen "success rates" vary by metric and setting — full-sequence vs mean-subtask, offline vs
online — so direct cross-paper ranking needs matching the exact protocol.)*

## What The World Model Is Good For

Across all tiers, a consistent picture of where JEPA's *predictive* learning pays
off — verified by direct experiment, including several clean negative results.

**1. The encoder (representation) is load-bearing; ablation-verified.** The
Adroit/maze controllers act on `encode(obs)`, never the raw observation. Replacing
the trained encoder with a random-init encoder of identical shape collapses Adroit
Door from 0.90 → 0.00 — the policy genuinely routes through the learned latent and
is not a raw-observation "cheater".

**2. Planning through the predictor works on smooth dynamics, fails on
contact-rich ones.** Sampling-based MPC over the JEPA dynamics solves the smooth
Fetch reach/push tasks and drives the *high level* of the maze hierarchy (abstract
subgoal transitions are smooth). But on contact-rich tasks it actively *hurts*:

- FetchSlide MPC: 0.00 (the planner exploits world-model error instead of striking).
- Adroit Pen, BC vs BC+world-model MPC: **0.70 → 0.15** (aggressive planning) or
  → 0.70 (conservative planning just reproduces BC). An ensemble-disagreement
  penalty (Roadmap A) did not rescue it.

**3. You cannot "just make the world model more precise" for contact planning.**
We retrained the Pen world model on the *expert demo manifold* (where the planner
actually queries) instead of random data. It became a *worse* open-loop predictor
(rollout error went from ~4× better-than-static to *worse* than static off the
narrow expert distribution), and MPC still lost to BC. The reason is structural:
**a world model is only accurate where it was trained, but a planner's whole job
is to perturb *off* that distribution** to search for improvements — the
"perturbation frontier" is by definition the un-modeled region. Data-distribution
matching cannot fix this; it is why contact-rich model-based control is genuinely
hard, and why the maze *hierarchy* (which plans over smooth, in-distribution
macro-steps) is the regime where the JEPA predictor finally earns its keep.

This is the central scientific result of the repo: **JEPA's value is its
representation and — for smooth/abstract dynamics — its predictor; from-scratch
model-based planning is not a free win on contact-rich control.**

## Roadmap Progress

- **World-model upgrades:** ✅ K-head **ensemble dynamics** with an inter-head
  disagreement signal (`--ensemble-heads`); ✅ **VICReg** variance+covariance
  regularization (prevents latent collapse, monitored every run); ✅ better
  transition trials and **offline D4RL demos** for Adroit/AntMaze.
- **Hierarchical JEPA:** ✅ demonstrated across PointMaze + AntMaze (above).
- The code is modular by responsibility (`models/`, categorized `algos/`, and
  thin `scripts/{data,train,eval}` CLIs) so these were additive, not rewrites.

## Setup

Use a Python environment with MuJoCo/Gymnasium Robotics installed. The scripts
below assume a conda env named `myenv`, but any environment with the
requirements installed should work.

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install -r requirements.txt
# optional, for the offline-demo (Adroit / AntMaze) experiments:
pip install --no-deps minari h5py portion
```

On headless GPU machines, use EGL:

```bash
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
```

## Quick Smoke Test

This verifies imports, environment creation, a tiny training loop, checkpoint
writing, and a one-episode evaluation.

```bash
conda activate myenv
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
python -m jepa_robotics.train \
  --task fetch_reach \
  --output-root runs \
  --smoke \
  --device cpu
```

Expected outputs:

- `runs/fetch_reach/checkpoints/fetch_reach_jepa_checkpoint.pt`
- `runs/fetch_reach/checkpoints/fetch_reach_jepa_model.pt`

The smoke result is not meant to solve the task. It only checks that the code
runs.

## Train A Small FetchReach Model

```bash
conda activate myenv
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
python -m jepa_robotics.train \
  --task fetch_reach \
  --output-root runs \
  --collect-steps 100000 \
  --train-steps 15000 \
  --batch-size 256 \
  --horizons 1,2,4,8 \
  --latent-dim 64 \
  --hidden-dim 256 \
  --predictor-mode rollout \
  --lambda-pred-probe 0.15 \
  --lambda-pred-goal 0.15 \
  --device auto
```

By default this writes:

- `runs/fetch_reach/checkpoints/fetch_reach_jepa_checkpoint.pt`
- `runs/fetch_reach/checkpoints/fetch_reach_jepa_model.pt`

You can override paths with `--save-path` and `--model-path`.

## Train The Stronger Goal-Focused Model

This is the best pure-JEPA configuration tested in this repo so far. It adds
auxiliary losses that make the predicted future achieved-goal coordinates more
accurate. It still evaluates without teacher correction or scripted proposal
actions.

```bash
conda activate myenv
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
python -m jepa_robotics.train \
  --task fetch_reach \
  --output-root runs \
  --collect-steps 220000 \
  --scripted-fraction 0.45 \
  --action-noise 0.25 \
  --train-steps 100000 \
  --batch-size 512 \
  --horizons 1,2,4,8,16 \
  --latent-dim 128 \
  --hidden-dim 512 \
  --predictor-mode rollout \
  --lambda-pred-probe 0.2 \
  --lambda-pred-achieved 30.0 \
  --lambda-pred-goal 0.3 \
  --lambda-probe 0.08 \
  --lambda-achieved 5.0 \
  --lambda-goal 0.08 \
  --lambda-distance 0.1 \
  --device auto \
  --model-path runs/fetch_reach/checkpoints/reach_goal_focus_model.pt \
  --save-path runs/fetch_reach/checkpoints/reach_goal_focus_checkpoint.pt
```

## Evaluate

Evaluate random actions, the scripted controller, and pure JEPA+MPC on the same
seeds:

```bash
conda activate myenv
PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
python -m jepa_robotics.evaluate \
  --task fetch_reach \
  --output-root runs \
  --model-path runs/fetch_reach/checkpoints/reach_goal_focus_model.pt \
  --episodes 50 \
  --mpc-method grad \
  --mpc-score state \
  --mpc-candidates 64 \
  --mpc-horizon 8 \
  --grad-iters 25 \
  --grad-lr 0.06 \
  --action-l2-weight 0.02 \
  --action-delta-weight 0.1 \
  --execute-smoothing 0.2 \
  --teacher-correction-fraction 0.0 \
  --jepa-scripted-proposal-fraction 0.0 \
  --device auto \
  --out runs/fetch_reach/eval_results/pure_jepa_eval.jsonl \
  --video-policy jepa_mpc_grad_state_smooth \
  --video-dir runs/fetch_reach/videos
```

The important flags for a pure comparison are:

- `--teacher-correction-fraction 0.0`
- `--jepa-scripted-proposal-fraction 0.0`

Those keep evaluation from using the scripted controller as a crutch.

## Model Sizes

The action-conditioned JEPA world models are compact and efficient:

| Task | Model | Parameters |
| --- | --- | ---: |
| FetchReach-v4 | reach_goal_focus_deadline | 1.74M |
| FetchPush-v4 | push_v2 | 2.02M |
| FetchPickAndPlace-v4 | pickplace_v2 | 2.02M |
| **Total (all three production models)** | | **5.78M** |

Parameter counts include the encoder, target encoder, dynamics predictor (recurrent GRU-based), state/distance probes, and all embedding layers. Run `python scripts/count_params.py` to recount parameters in all checkpoints.

## Results Snapshot

These are local results from the development runs. They are included as context,
not as committed artifacts.

| Policy / setup | FetchReach-v4 success | Mean final distance | Mean action delta |
| --- | ---: | ---: | ---: |
| Random | 0.00 | 0.2362 | 1.5403 |
| Scripted proportional controller | 1.00 | 0.0023 | 0.0144 |
| Earlier pure rollout JEPA + grad MPC | 0.94 | 0.0250 | 0.3340 |
| Goal-focused pure rollout JEPA + state grad MPC | 0.94 | 0.0227 | 0.0921 |
| Teacher-corrected JEPA | 1.00 | 0.0023 | 0.0144 |

The teacher-corrected row is intentionally labeled. It uses
`--teacher-correction-fraction 1.0 --teacher-correction-threshold inf`, so it
matches the scripted controller by design. The more interesting result is the
pure row: same success as the earlier pure model, better final distance, and much
less shaky control.

**FetchPickAndPlace-v4** (30 episodes). Pick/place no longer needs the old BC
action prior. The replacement is a **self-supervised inverse action-chunk prior**
conditioned on `(z_t, z_goal)` plus exact Fetch geometry. Trials teach which
8-step action chunks cause which futures; the JEPA world model scores noisy
candidate chunks at evaluation:

| Policy / setup | Success | Mean final distance |
| --- | ---: | ---: |
| Random | 0.00 | 0.261 |
| Scripted controller (conventional reference) | 1.00 | 0.014 |
| Historical JEPA BC policy (learned, on latent) | 1.00 | 0.017 |
| Historical JEPA BC + world-model MPC | 1.00 | 0.011 |
| **Inverse prior + JEPA chunk selection** | **1.00** | **0.011** |

The direct inverse prior also reaches 1.00 success (mean final distance 0.014).
The old `pickplace_v2_policy.pt` BC artifact is no longer required.

**FetchPush-v4** (30 episodes). Push no longer needs a BC action prior. The
replacement is a **flow-matching action prior conditioned on `(z_t, z_future)`**:
demos define local desirable futures, the flow samples plausible 8-step action
chunks, and the JEPA dynamics model selects the chunk whose rollout best reaches
the encoded future. Multi-horizon future conditioning (`4,8,12,16`), exact Fetch
geometry side features, and combined latent+decoded-state scoring close the gap:

| Policy / setup | Success | Mean final distance |
| --- | ---: | ---: |
| Random | 0.07 | 0.184 |
| Scripted controller (conventional reference) | 0.97 | 0.031 |
| Historical JEPA BC policy (learned, on latent) | 0.93 | 0.033 |
| Historical JEPA BC + world-model MPC (reach term off) | 1.00 | 0.014 |
| **Flow prior + JEPA chunk selection** | **1.00** | **0.008** |

Push has a task-specific subtlety: a good push contacts the *far* side of the
object from the goal, so the gripper-to-object "reach" cost actively misleads the
planner (it pulls the gripper to the object centre). The flow+JEPA version avoids
copying action labels: `z_t = encoder(o_t)`, `z_future = target_encoder(o_{t+h})`,
`flow(a_{t:t+7} | z_t, z_future)` proposes chunks, and JEPA rolls out/scored chunks
online.

## Train The Manipulation Agent

`scripts/train_eval_object_v2.sh` runs the full pipeline for an object task:
collect data with the scripted expert and train the recurrent JEPA world model.
For `fetch_push`, it trains/evaluates the flow-prior + JEPA-selection replacement.
For `fetch_pick_place`, it trains/evaluates the inverse-prior + JEPA-selection
replacement. Neither path trains a state-to-action BC policy.

```bash
TASK_NAME=fetch_pick_place RUN_TAG=pickplace_v2 \
  bash scripts/train_eval_object_v2.sh
# or TASK_NAME=fetch_push
```

To evaluate the PickAndPlace inverse prior directly:

```bash
python scripts/eval_fetch_inverse_jepa.py \
  --task fetch_pick_place \
  --model-path runs/fetch_pick_place/checkpoints/pickplace_v2_model.pt \
  --inverse-path runs/fetch_pick_place/checkpoints/pickplace_inverse_prior_h8_goalgeom.pt \
  --goal-mode final --episodes 30 --candidates 64 --noise-std 0.05 \
  --exec-k 1 --target-horizon 8 --latent-weight 1.0 \
  --state-weight 5.0 --final-goal-weight 1.0 \
  --action-delta-weight 0.01 --device auto
```

This reports the JEPA-ranked inverse-chunk controller; direct inverse execution
is obtained with `--candidates 1 --latent-weight 0 --state-weight 0 --final-goal-weight 0`.

## Record A Video

Multi-episode showcase videos (with varied / mid-air goals) for the learned
agent and the scripted reference:

```bash
# Scripted reference controller
python scripts/record_expert.py --task fetch_pick_place --vary-goal --episodes 6
```

Any evaluation can also record the first episode for a selected policy:

```bash
python -m jepa_robotics.evaluate \
  --task fetch_reach \
  --model-path runs/fetch_reach/checkpoints/reach_goal_focus_model.pt \
  --episodes 1 \
  --mpc-method grad \
  --mpc-score state \
  --video-policy jepa_mpc_grad_state_smooth \
  --video-dir runs/fetch_reach/videos
```

Videos are written as MP4 files under the selected `--video-dir`.

## Slurm

The Slurm scripts in `scripts/` assume:

- conda is available at `/opt/anaconda3/etc/profile.d/conda.sh`
- the environment is named `myenv`, or `CONDA_ENV` is set
- the cluster supports the `--gres=gpu:1` option

Example:

```bash
CONDA_ENV=myenv sbatch scripts/train_fetchreach_rollout.slurm
```

Outputs are written under `runs/` and ignored by Git.

## Task Notes

- `fetch_reach`: goal-conditioned reaching; solved by pure JEPA + MPC (no policy
  needed).
- `fetch_push`: push an object to a goal on the table. Solved with a
  flow-matching action-chunk prior conditioned on current/future JEPA latents,
  with JEPA dynamics selecting the chunk to execute.
- `fetch_pick_place`: grasp and place, often at a mid-air goal. Solved (1.00
  success) with an SSL inverse action prior plus JEPA chunk selection;
  sampling-only MPC was not enough (see
  ["A world model is not a controller"](#a-world-model-is-not-a-controller)).
- `fetch_slide`: ballistic strike, gripper locked, goal out of reach. The
  goal-frame equivariant ballistic JEPA HWM reaches 0.853/2000; flat primitive
  MPC fails.
- `point_umaze` / `point_medium` / `point_large`, `antmaze_*`: maze navigation,
  solved by Hierarchical JEPA (Tier 2). AntMaze env ids must match the Minari
  D4RL dataset they were recorded with (`AntMaze_*_Diverse_GR-v4`).
- `adroit_door` / `adroit_hammer` / `adroit_pen` / `adroit_relocate`: 24–30-DoF
  dexterous hand, flat non-goal observation. Door is now controlled by a
  schedule-phase inverse prior; Hammer by the p4 schedule-phase inverse prior;
  Pen by a raw+latent future-conditioned flow prior, and Relocate by dual
  possession-specialist inverses. Explore
  world-model artifacts are merged under each task as
  `runs/adroit_*/explore`; canonical
  `runs/adroit_*/checkpoints/*_jepa_model.pt` links point at those checkpoints.

### Offline demos (Adroit / AntMaze)

Tier 3 and the AntMaze low levels use offline D4RL datasets via Minari:

```bash
pip install --no-deps minari h5py portion   # keep gymnasium/numpy pinned
export MINARI_DATASETS_PATH=$PWD/.cache/minari
python scripts/minari_to_npz.py --dataset D4RL/door/expert-v2 \
  --out runs/adroit_door/data/door_expert_demos.npz
python -m jepa_robotics.train_policy --task adroit_door \
  --model-path <jepa_wm>.pt --episodes-npz runs/adroit_door/data/door_expert_demos.npz \
  --train-steps 30000 --out runs/adroit_door/checkpoints/door_bc.pt
```

## Current Limitations

- This is low-dimensional state JEPA, not pixel JEPA.
- **Generic model-based primitive planning still does not solve the hardest
  contact control.** Slide's H24 flow/inverse and receding-horizon JEPA ranking
  failed; a goal-frame equivariant ballistic HWM reaches 0.853/2000 by moving
  planning to the strike/coast timescale. All four Adroit tasks use
  self-supervised
  future-conditioned priors; Relocate's 0.957/210 dual-specialist controller
  replaced the former BC checkpoint.
- Hierarchical JEPA's ceiling is the low-level controller's competence; AntMaze
  is solved only when the low-level walker is strong enough.
- The MPC refinement (where it helps) runs online, so policy + MPC is slower than
  the feed-forward policy alone.
- On contact-rich long-horizon manipulation (FrankaKitchen), the plain JEPA latent
  *trails raw obs*, and the plain JEPA predictor is worse than the raw action
  prior. Segment-pure inverse specialists solve 0.813/150 full-4;
  an oracle-free learned completion switch retains 0.784/125 held-out.
- Clean negatives logged this round (kitchen): classifier-free guidance hurts, progress/
  history conditioning and scheduler stall-rotation are within noise, and a single
  bigger-net "stronger push" regressed — the wins came from the four ingredients above,
  not from conditioning/capacity tweaks.
