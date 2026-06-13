#!/bin/bash
# Retrain FetchReach with the CURRENT architecture so the checkpoint is
# compatible with the current eval/record code (the old reach checkpoints used
# an obsolete narrow state_probe and no longer load). Trains for ~2h (wall-clock
# capped), then re-records a high-res MPC video from the final checkpoint.
#
# Reach is solved by PURE MPC (no learned policy), so this trains only the world
# model and records via jepa_robotics.evaluate (which records the MPC agent).
#
# 1 GPU, any node: submitted to the default ALL partition with --gres=gpu:1.
#   sbatch scripts/retrain_reach_compat.slurm
#SBATCH --job-name=jepa-reach-retrain
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-myenv}"
set -u

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export MUJOCO_GL=egl

RUN_TAG="${RUN_TAG:-reach_compat_${SLURM_JOB_ID:-local}}"
TASK_DIR="runs/fetch_reach"
CKPT_DIR="$TASK_DIR/checkpoints"; LOG_DIR="$TASK_DIR/logs"
VIDEO_DIR="$TASK_DIR/videos"; EVAL_DIR="$TASK_DIR/eval_results"
MODEL_PATH="$CKPT_DIR/${RUN_TAG}_model.pt"
CKPT_PATH="$CKPT_DIR/${RUN_TAG}_checkpoint.pt"
EVAL_LOG="$EVAL_DIR/${RUN_TAG}_eval.jsonl"
TRAIN_LOG="$LOG_DIR/${RUN_TAG}_train.jsonl"
mkdir -p "$TMPDIR" "$CKPT_DIR" "$LOG_DIR" "$VIDEO_DIR" "$EVAL_DIR"

# Stage 1: world model, ~2h wall-clock cap. --save-every keeps the model artifact
# fresh, so the last save is a usable "final checkpoint" even when the cap fires.
# train-steps is set high so the 2h timeout is the limiter, not step count.
TRAIN_SECONDS="${TRAIN_SECONDS:-7200}"
set +e
timeout --signal=INT --kill-after=60 "$TRAIN_SECONDS" \
python -m jepa_robotics.train \
  --task fetch_reach \
  --output-root runs \
  --seed "${SEED:-19}" \
  --collect-steps "${COLLECT_STEPS:-220000}" \
  --collect-log-every 20000 \
  --scripted-fraction 0.45 \
  --action-noise 0.25 \
  --train-steps "${TRAIN_STEPS:-400000}" \
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
  --save-every 4000 \
  --device cuda \
  --model-path "$MODEL_PATH" \
  --save-path "$CKPT_PATH" \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e
echo "training stopped (rc=$TRAIN_RC; 124=timeout cap as intended)"

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: no model artifact at $MODEL_PATH; nothing to record." >&2
  exit 1
fi

# Stage 2: evaluate + re-record the PURE-MPC agent at high resolution (960x720).
python -m jepa_robotics.evaluate \
  --task fetch_reach \
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
  --device cuda \
  --out "$EVAL_LOG" \
  --video-policy jepa_mpc \
  --video-dir "$VIDEO_DIR" \
  --width 960 --height 720 \
  --fps 30

echo "DONE RUN_TAG=$RUN_TAG MODEL=$MODEL_PATH EVAL=$EVAL_LOG VIDEO_DIR=$VIDEO_DIR"
