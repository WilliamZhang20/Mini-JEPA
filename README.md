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
| base | FetchReach / Push / PickAndPlace | JEPA + learned policy + MPC | 1.00 / 1.00 / 1.00 |
| 1 | **FetchSlide** (ballistic strike) | JEPA-latent TQC + HER | 0.83 |
| 2 | **PointMaze** UMaze / Medium / Large | **Hierarchical JEPA** | 1.00 / 0.90 / 1.00 |
| 2 | **AntMaze** UMaze (8-DoF ant) | Hierarchical JEPA | 0.93 |
| 3 | **Adroit** Door / Hammer / Pen / Relocate | JEPA-latent BC on offline demos | 0.96 / 1.00 / 0.77 / 1.00 |
| 4 | **FrankaKitchen** (4 sequential sub-tasks) | control-aware-JEPA skill-hierarchy + online self-imitation | **0.90 full-4** (3.88/4 sub-tasks) |

The repo is deliberately not one monolithic RL agent. It trains a JEPA predictive
world model and a **goal-conditioned controller in its latent** — by behaviour
cloning, HER reinforcement learning, or a hierarchical subgoal planner, whichever
the task demands — and the experiments below map out *which* of those wins where,
and why. Two findings recur and are documented in
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
- `jepa_robotics/train_policy.py`: behaviour-clones a goal-conditioned action
  prior on the (frozen) JEPA latent. This is the "controller" half of the agent.
- `jepa_robotics/evaluate.py`: compares random actions, a scripted controller,
  the learned policy on its own, and the policy-seeded JEPA+MPC planner. It can
  also record MP4 rollouts.
- `jepa_robotics/models/`: the action-conditioned JEPA model (recurrent latent
  dynamics, optional K-head ensemble) and the `GoalConditionedPolicy` action
  prior, split into `world_model.py` / `policy.py` / `mlp.py` / `regularizers.py`.
- `jepa_robotics/scoring/`: per-task MPC cost mixins (`manip`/`strike`/`goal`).
- `jepa_robotics/data.py`: trajectory collection, scripted experts
  (reach/push/pick/slide/maze), offline-npz loading, and normalization.
- `jepa_robotics/envs.py`: Gymnasium Robotics registration, observation
  flattening, maze/AntMaze goal-env handling.
- `jepa_robotics/sb3_jepa.py`: SB3 feature extractors that put the JEPA latent
  under a TQC/SAC policy (HER, trainable-encoder, and concat variants).
- `jepa_robotics/tasks.py`: task presets for Fetch, FetchSlide, PointMaze,
  AntMaze, and the Adroit suite.
- `scripts/`: Slurm entry points, the `train_eval_object_v2.sh` pipeline, the
  Hierarchical-JEPA evaluator (`eval_hjepa_maze.py`), the offline-demo adapter
  (`minari_to_npz.py`), world-model accuracy probe (`eval_wm_rollout.py`), and
  video recorders.

Experiment outputs are intentionally ignored by Git. Checkpoints, videos, logs,
and JSONL eval files are written under `runs/` by default.

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

3. **The controller needs its own self-supervision.** We behaviour-clone a small
   `GoalConditionedPolicy` on the frozen JEPA latent (`train_policy.py`). This
   learned action prior knows the grasp choreography; the world-model MPC then
   *refines and verifies* it. This mirrors how modern world-model agents work
   (Dreamer, TD-MPC2, DINO-WM): a world model paired with a learned policy/value,
   not planning-by-sampling alone.

The payoff (FetchPickAndPlace, 30 episodes): the learned policy alone matches the
scripted controller, and policy + world-model MPC is slightly *more* precise than
scripted. See [Results Snapshot](#results-snapshot).

## Beyond Fetch: Tiers 1–3

The same compact-JEPA recipe scales to much harder tasks, each requiring one new
ingredient on top of the world model.

### Tier 1 — FetchSlide (ballistic strike): 0.83

The gripper is locked and the goal is *out of reach*, so the arm must impart the
right momentum in a single strike and then watch — there is no post-contact
correction. Receding-horizon MPC is structurally wrong for this ("commit then
watch"), and indeed flat CEM planning scored 0.00. The winner is a **TQC + HER
controller trained on the frozen JEPA latent** (deep-learning control, not
planning): 0.83 success, up from ~0.60, near the TQC reference ceiling (~0.87).
FetchSlide is a known-hard ballistic task where even SOTA tops out in the high
0.8s. (`train_jepa_sb3_policy.py`, demonstration-augmented HER.)

### Tier 2 — Mazes: Hierarchical JEPA

Long mazes break flat goal-conditioned control: a straight-line-to-goal policy
walks into walls, and the success signal is hundreds of steps away. Flat HER
plateaus around 0.70 (it reaches *visible* goals but cannot route around walls).

**Hierarchical JEPA (`eval_hjepa_maze.py`)** splits control into two timescales:

- **Low level** — the goal-conditioned policy (HER for PointMaze, HER-relabeled BC
  for AntMaze) acting on the JEPA latent; it reliably reaches *nearby* subgoals.
- **High level** — a **data-driven subgoal graph**: landmarks sampled in
  achieved-goal (x, y) space, with an edge between two landmarks only if the agent
  *empirically* got from one to the other within *k* steps. Edges therefore only
  exist where trajectories actually went — i.e. **around** walls. Dijkstra plans a
  subgoal path; the low level executes one subgoal at a time.

This beats flat on **every** maze with the *same* low level:

| Maze | Flat (low level → goal) | **Hierarchical JEPA** |
| --- | ---: | ---: |
| PointMaze UMaze | 0.70 | **1.00** |
| PointMaze Medium | 0.70 | **0.90** |
| PointMaze Large | 0.70 | **1.00** |
| AntMaze UMaze (8-DoF ant) | 0.70 | **0.93** |

The win magnitude is set by the low level's competence (the documented hierarchy
caveat — the hierarchy cannot reach subgoals the low level cannot). On the bigger
AntMaze layouts the offline-BC ant walker is the bottleneck, addressed by a
stronger offline→online TQC+HER low level (`uncap_antmaze_hjepa.slurm`).

### Tier 3 — Adroit dexterous hand (24–30-DoF): the whole suite

The Adroit hand breaks the scripted-expert data engine — there is no simple
geometric controller for finger coordination, and the observation is flat (no
goal). From-scratch RL on these sparse-success, contact-rich tasks gets ~0%
(see findings below). The fix is a **learned data source**: offline D4RL expert
demonstrations (via [Minari](https://minari.farama.org/)), behaviour-cloned on
the frozen JEPA latent (`minari_to_npz.py` → `train_policy.py --episodes-npz`).

| Adroit task | from-scratch RL | **JEPA-latent BC on offline demos** |
| --- | ---: | ---: |
| Door | 0.00 | **0.96** |
| Hammer | 0.00 | **1.00** |
| Pen (in-hand reorientation) | 0.00 | **0.77** |
| Relocate | 0.00 | **1.00** |

Notably the BC controllers run on the *same* exploratory world model (trained on
random data). The encoder is information-preserving enough (its state-probe loss
forces the latent to reconstruct the full state) that BC clones the experts
cleanly — the random-data latent only blocked RL *exploration*, never imitation.

### Tier 4 — FrankaKitchen (4 compositional sub-tasks): 0.90 full-success

The hardest tier cleared, and it needed the whole upgraded stack. A 9-DoF arm must
complete an *ordered set* of 4 kitchen sub-tasks (microwave, kettle, light switch,
slide cabinet). **Every flat controller scored 0** — BC, TD3+BC, IQL, CEM-MPC, and a
latent Dreamer all fail, because the wall is *chaining*, not single-step control (the
JEPA *predictor* is even worse-than-no-op here: contact-rich → model exploitation).
Five ingredients, each earned by diagnosis, take it from 0 to 0.90 full success:

| step | what | full-4 |
| --- | --- | ---: |
| flat (BC / TD3+BC / IQL / MPC / Dreamer) | single-step control on the latent | 0.00 |
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

Final: **3.88/4 sub-tasks on average, 0.90 full 4-task success** (4 seeds), entirely JEPA-based.
Videos: `runs/franka_kitchen/videos/kitchen_jepa_rl_tuned.mp4` (online-tuned, 0.90) and
`kitchen_jepa_skill_hierarchy.mp4` (offline, 0.68).

**Behaviour analysis** (`scripts/analyze_kitchen_behavior.py`) pinpoints the bottleneck: the **3rd
task, the light switch**. Microwave and kettle are always ~1.0; the early policy flipped the switch
only 0.33 of the time and stalled there, and since the policy runs a fixed
microwave→kettle→light-switch→slide-cabinet order, raising switch-completion to 0.88 cascaded into
full sequences. **Negative result on RL:** an *actual* advantage-weighted-regression objective
(`scripts/finetune_skill_rl.py`) matched self-imitation at its peak (0.93) but then *collapsed*
(on-policy instability) — the bottleneck is skill *reliability*, not exploration/credit-assignment,
so filtered self-imitation (an accumulating success buffer) is the more robust tool here.

*(Metric note: these are mean-subtask and full-sequence rates on the D4RL partial+complete offline
demos; "online fine-tuning" here means the policy's own rollouts, not reward-based RL. Published
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
  regularization (prevents latent collapse, monitored every run); ✅ better data
  (HER for slide/maze, **offline D4RL demos** for Adroit/AntMaze).
- **Hierarchical JEPA:** ✅ demonstrated across PointMaze + AntMaze (above).
- The code is modular by responsibility (`models/` package, per-task `scoring/`
  mixins) so these were additive, not rewrites.

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

The Tier 2/3 reinforcement-learning controllers also use `stable-baselines3` and
`sb3-contrib` (TQC).

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

**FetchPickAndPlace-v4** (30 episodes, recurrent JEPA world model + learned
policy on the latent). The learned controller matches the scripted reference,
and adding world-model MPC refinement makes it slightly more precise:

| Policy / setup | Success | Mean final distance |
| --- | ---: | ---: |
| Random | 0.00 | 0.261 |
| Scripted controller (conventional reference) | 1.00 | 0.014 |
| JEPA policy (learned, on latent) | 1.00 | 0.017 |
| JEPA policy + world-model MPC | 1.00 | **0.011** |

The earlier sampling-only planner (no learned policy) reached only ~0.40 success
on the same model — it never reliably grasped. The jump to 1.00 is entirely from
adding the learned action prior, not from changing the world model.

**FetchPush-v4** (30 episodes, same pipeline):

| Policy / setup | Success | Mean final distance |
| --- | ---: | ---: |
| Random | 0.07 | 0.184 |
| Scripted controller (conventional reference) | 0.97 | 0.031 |
| JEPA policy (learned, on latent) | 0.93 | 0.033 |
| JEPA policy + world-model MPC (reach term off) | **1.00** | **0.014** |

Push has a task-specific subtlety: a good push contacts the *far* side of the
object from the goal, so the gripper-to-object "reach" cost actively misleads the
planner (it pulls the gripper to the object centre). Turning that term off
(`--manip-reach-weight 0.0`) lets the world-model MPC refine the learned policy's
push and it beats the scripted controller on both success and precision.

## Train The Manipulation Agent (World Model + Policy + MPC)

`scripts/train_eval_object_v2.sh` runs the full pipeline for an object task:
collect data with the scripted expert, train the recurrent JEPA world model,
behaviour-clone the goal-conditioned policy on its latent, evaluate all four
policies, and record agent + reference videos.

```bash
TASK_NAME=fetch_pick_place RUN_TAG=pickplace_v2 \
  bash scripts/train_eval_object_v2.sh
# or TASK_NAME=fetch_push
```

To evaluate a trained model + policy directly:

```bash
python -m jepa_robotics.evaluate \
  --task fetch_pick_place \
  --model-path runs/fetch_pick_place/checkpoints/pickplace_v2_model.pt \
  --policy-path runs/fetch_pick_place/checkpoints/pickplace_v2_policy.pt \
  --policy-proposal-fraction 0.5 \
  --episodes 30 --mpc-method cem --mpc-score manip \
  --mpc-candidates 128 --mpc-horizon 12 --cem-iters 4 --action-std 0.5 \
  --manip-reach-weight 0.1 --manip-path-weight 0.3 --device auto
```

This reports `random`, `scripted`, `jepa_policy` (the learned prior alone), and
`jepa_mpc_..._policy50` (policy-seeded world-model MPC) on the same seeds.

## Record A Video

Multi-episode showcase videos (with varied / mid-air goals) for the learned
agent and the scripted reference:

```bash
# Learned JEPA agent (policy + world-model MPC)
python scripts/record_jepa.py --task fetch_pick_place --vary-goal --episodes 6 \
  --model-path runs/fetch_pick_place/checkpoints/pickplace_v2_model.pt \
  --policy-path runs/fetch_pick_place/checkpoints/pickplace_v2_policy.pt

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
- `fetch_push`: push an object to a goal on the table. Solved with the world
  model + learned policy pipeline.
- `fetch_pick_place`: grasp and place, often at a mid-air goal. Solved (1.00
  success) with the world model + learned policy + MPC; sampling-only MPC was
  not enough (see ["A world model is not a controller"](#a-world-model-is-not-a-controller)).
- `fetch_slide`: ballistic strike, gripper locked, goal out of reach. Solved
  (0.83) by a TQC+HER controller on the JEPA latent; flat MPC fails (Tier 1).
- `point_umaze` / `point_medium` / `point_large`, `antmaze_*`: maze navigation,
  solved by Hierarchical JEPA (Tier 2). AntMaze env ids must match the Minari
  D4RL dataset they were recorded with (`AntMaze_*_Diverse_GR-v4`).
- `adroit_door` / `adroit_hammer` / `adroit_pen` / `adroit_relocate`: 24–30-DoF
  dexterous hand, flat non-goal observation. Solved by behaviour cloning offline
  D4RL demos on the JEPA latent (Tier 3); from-scratch RL gets ~0%.

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
- **Model-based planning does not help contact-rich control** (slide, Adroit) — it
  exploits world-model error; those tasks are solved by learned control (BC/HER) on
  the JEPA latent, not by planning. See
  [What the world model is good for](#what-the-world-model-is-good-for).
- Hierarchical JEPA's ceiling is the low-level controller's competence; weak
  locomotion (offline-BC ant) caps the harder AntMaze layouts (~0.25–0.30, genuinely
  hard offline benchmarks where 1.0 is not realistic).
- The MPC refinement (where it helps) runs online, so policy + MPC is slower than
  the feed-forward policy alone.
- On contact-rich long-horizon manipulation (FrankaKitchen), the plain JEPA latent
  *trails raw obs* until an inverse-dynamics auxiliary makes it control-aware, and the
  JEPA *predictor* is worse-than-no-op (planning is hopeless) — control there comes from
  the encoder + an action-chunked flow skill-hierarchy + DAgger, not from the world model.
- Clean negatives logged this round (kitchen): classifier-free guidance hurts, progress/
  history conditioning and scheduler stall-rotation are within noise, and a single
  bigger-net "stronger push" regressed — the wins came from the four ingredients above,
  not from conditioning/capacity tweaks.
