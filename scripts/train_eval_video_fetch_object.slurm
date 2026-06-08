#!/bin/bash
#SBATCH --job-name=jepa-fetch-object
#SBATCH --partition=YV2PANT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-myenv}"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export MUJOCO_GL=egl

TASK_NAME="${TASK_NAME:-fetch_push}"
case "$TASK_NAME" in
  fetch_pick_place)
    TASK_SLUG="fetch_pick_place"
    CONTROLLER_GAIN="${CONTROLLER_GAIN:-5.0}"
    SCRIPTED_FRACTION="${SCRIPTED_FRACTION:-0.65}"
    ACTION_NOISE="${ACTION_NOISE:-0.2}"
    ;;
  fetch_push)
    TASK_SLUG="fetch_push"
    CONTROLLER_GAIN="${CONTROLLER_GAIN:-3.0}"
    SCRIPTED_FRACTION="${SCRIPTED_FRACTION:-0.5}"
    ACTION_NOISE="${ACTION_NOISE:-0.3}"
    ;;
  *)
    echo "Unsupported TASK_NAME=$TASK_NAME. Use fetch_pick_place or fetch_push." >&2
    exit 2
    ;;
esac

RUN_TAG="${RUN_TAG:-${TASK_SLUG}_object_jepa_${SLURM_JOB_ID}}"
TASK_DIR="runs/$TASK_SLUG"
CHECKPOINT_DIR="$TASK_DIR/checkpoints"
LOG_DIR="$TASK_DIR/logs"
VIDEO_DIR="$TASK_DIR/videos"
EVAL_DIR="$TASK_DIR/eval_results"
MODEL_PATH="$CHECKPOINT_DIR/${RUN_TAG}_model.pt"
CHECKPOINT_PATH="$CHECKPOINT_DIR/${RUN_TAG}_checkpoint.pt"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.jsonl"
EVAL_LOG="$EVAL_DIR/${RUN_TAG}_eval.jsonl"

mkdir -p "$TMPDIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$VIDEO_DIR" "$EVAL_DIR"

python -m jepa_robotics.train \
  --task "$TASK_NAME" \
  --output-root runs \
  --seed "${SEED:-41}" \
  --collect-steps "${COLLECT_STEPS:-300000}" \
  --collect-log-every 10000 \
  --scripted-fraction "$SCRIPTED_FRACTION" \
  --controller-gain "$CONTROLLER_GAIN" \
  --action-noise "$ACTION_NOISE" \
  --train-steps "${TRAIN_STEPS:-90000}" \
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
  --eval-episodes 3 \
  --mpc-candidates 128 \
  --mpc-horizon 8 \
  --log-every 250 \
  --save-every 10000 \
  --device cuda \
  --model-path "$MODEL_PATH" \
  --save-path "$CHECKPOINT_PATH" \
  2>&1 | tee "$TRAIN_LOG"

python -m jepa_robotics.evaluate \
  --task "$TASK_NAME" \
  --output-root runs \
  --model-path "$MODEL_PATH" \
  --episodes "${EVAL_EPISODES:-20}" \
  --seed "${EVAL_SEED:-123}" \
  --mpc-method grad \
  --mpc-score state \
  --mpc-candidates "${MPC_CANDIDATES:-64}" \
  --mpc-horizon 8 \
  --grad-iters 25 \
  --grad-lr 0.06 \
  --action-l2-weight 0.02 \
  --action-delta-weight 0.1 \
  --execute-smoothing 0.2 \
  --teacher-correction-fraction 0.0 \
  --jepa-scripted-proposal-fraction 0.0 \
  --device cuda \
  --out "$EVAL_LOG" \
  --video-policy jepa_mpc_grad_state_smooth \
  --video-dir "$VIDEO_DIR" \
  --fps 30

echo "TASK_NAME=$TASK_NAME"
echo "MODEL_PATH=$MODEL_PATH"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "EVAL_LOG=$EVAL_LOG"
echo "VIDEO_DIR=$VIDEO_DIR"
