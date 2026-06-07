# JEPA Mini Robotics

Small action-conditioned JEPA experiments for Gymnasium Robotics.

The goal of this repo is to see how far a compact self-supervised world model can
go on simple robot control tasks without becoming a giant foundation model. The
main benchmark is `FetchReach-v4`: learn a latent dynamics model from
low-dimensional robot observations, then use model-predictive control (MPC) over
the learned model to reach the goal.

This is deliberately not an RL agent. It trains a JEPA-style predictive model,
then plans actions at evaluation time.

## What Is Inside

- `jepa_robotics/train.py`: collects trajectories, trains the JEPA world model,
  and writes a checkpoint/model artifact.
- `jepa_robotics/evaluate.py`: compares random actions, a simple scripted
  controller, optional SB3 policies, and JEPA+MPC planners. It can also record
  MP4 rollouts.
- `jepa_robotics/models.py`: the compact action-conditioned JEPA model.
- `jepa_robotics/data.py`: trajectory collection, scripted data policies, and
  normalization.
- `jepa_robotics/envs.py`: Gymnasium Robotics registration and observation
  flattening.
- `jepa_robotics/tasks.py`: task presets for FetchReach, FetchPickAndPlace, and
  exploratory Adroit Door support.
- `scripts/`: Slurm entry points for longer runs.

Experiment outputs are intentionally ignored by Git. Checkpoints, videos, logs,
and JSONL eval files are written under `runs/` by default.

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

## Record A Video

Any evaluation can record the first episode for a selected policy:

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

- `fetch_reach`: primary task; goal-conditioned Fetch reaching.
- `fetch_pick_place`: wired for collection/training, but the simple scripted
  data policy is not yet strong enough for good pick-and-place performance.
- `adroit_door`: exploratory support for non-goal observation spaces. It needs a
  task-specific reward or success probe before the Fetch-style goal geometry
  objective is meaningful.

## Current Limitations

- This is low-dimensional state JEPA, not pixel JEPA.
- The current planner is doing online MPC, so evaluation can be slower than a
  feed-forward policy.
- Pure JEPA has not matched the scripted controller exactly. The gap is small on
  success rate for FetchReach but still real in precision.
- Checkpoints are not committed. Train locally or attach artifacts through a
  release if you want to distribute pretrained weights.
