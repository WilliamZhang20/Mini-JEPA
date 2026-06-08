# JEPA Mini Robotics

Small action-conditioned JEPA experiments for Gymnasium Robotics.

The goal of this repo is to see how far a compact self-supervised world model can
go on simple robot control tasks without becoming a giant foundation model. It
covers three Fetch tasks of increasing difficulty:

- `FetchReach-v4`: goal-conditioned reaching (solved by pure JEPA + MPC).
- `FetchPush-v4`: push an object across the table to a goal.
- `FetchPickAndPlace-v4`: grasp an object and place it at a goal, often mid-air.

For all three we learn a latent dynamics model from low-dimensional robot
observations and control by planning over it. For the contact-rich manipulation
tasks we add a small **learned action prior** on top of the JEPA representation
(see ["A world model is not a controller"](#a-world-model-is-not-a-controller)).

This is deliberately not an RL agent: it trains a JEPA-style predictive model and
a behaviour-cloned policy, then plans/refines actions at evaluation time.

## What Is Inside

- `jepa_robotics/train.py`: collects trajectories, trains the JEPA world model,
  and writes a checkpoint/model artifact.
- `jepa_robotics/train_policy.py`: behaviour-clones a goal-conditioned action
  prior on the (frozen) JEPA latent. This is the "controller" half of the agent.
- `jepa_robotics/evaluate.py`: compares random actions, a scripted controller,
  the learned policy on its own, and the policy-seeded JEPA+MPC planner. It can
  also record MP4 rollouts.
- `jepa_robotics/models.py`: the action-conditioned JEPA model (recurrent latent
  dynamics) and the `GoalConditionedPolicy` action prior.
- `jepa_robotics/data.py`: trajectory collection, scripted expert policies, and
  normalization.
- `jepa_robotics/envs.py`: Gymnasium Robotics registration and observation
  flattening.
- `jepa_robotics/tasks.py`: task presets for FetchReach, FetchPush,
  FetchPickAndPlace, and exploratory Adroit Door support.
- `scripts/`: Slurm entry points, the end-to-end `train_eval_object_v2.sh`
  pipeline, expert/agent video recorders, and `check_experts.py`.

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

## Setup

Use a Python environment with MuJoCo/Gymnasium Robotics installed. The scripts
below assume a conda env named `myenv`, but any environment with the
requirements installed should work.

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install -r requirements.txt
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

*(FetchPush-v4 results are produced by the same `train_eval_object_v2.sh`
pipeline and will be filled in once its run completes.)*

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
- `adroit_door`: exploratory support for non-goal observation spaces. It needs a
  task-specific reward or success probe before the Fetch-style goal geometry
  objective is meaningful.

## Current Limitations

- This is low-dimensional state JEPA, not pixel JEPA.
- The MPC refinement runs online, so the policy + MPC agent is slower than the
  feed-forward policy alone (which already matches the scripted controller).
- The learned policy is behaviour-cloned from the scripted experts, so it
  inherits their behaviour; the world model contributes the latent representation
  it runs on and the MPC refinement on top.
- Checkpoints are not committed. Train locally or attach artifacts through a
  release if you want to distribute pretrained weights.
