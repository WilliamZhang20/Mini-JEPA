#!/bin/bash
# Train a MAX-PERFORMANCE FetchSlide reference: TQC + HER (RL-Zoo recipe) trained
# directly on FetchSlide-v4, then record a high-res MP4. Unlike the pretrained
# v1 zoo model (which only transfers to ~43% on v4), this trains on the exact
# env the JEPA agent uses, for a fair max-perf reference.
#
# 1 GPU, any node (default ALL partition).
#   sbatch scripts/train_eval_slide_reference.slurm
#SBATCH --job-name=tqc-slide-ref
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00
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

RUN_TAG="${RUN_TAG:-slide_tqc_ref_${SLURM_JOB_ID:-local}}"
TASK_DIR="runs/fetch_slide"
CKPT_DIR="$TASK_DIR/checkpoints"; LOG_DIR="$TASK_DIR/logs"; VIDEO_DIR="$TASK_DIR/videos"
mkdir -p "$TMPDIR" "$CKPT_DIR" "$LOG_DIR" "$VIDEO_DIR"

# ~3.5h training cap (3M-step budget; whichever comes first), then record.
python scripts/train_sb3_reference.py \
  --task fetch_slide \
  --total-steps "${TOTAL_STEPS:-3000000}" \
  --train-seconds "${TRAIN_SECONDS:-12600}" \
  --max-steps 50 \
  --seed "${SEED:-1}" \
  --eval-episodes "${EVAL_EPISODES:-30}" \
  --width 960 --height 720 \
  --save-model "$CKPT_DIR/${RUN_TAG}.zip" \
  --video-out "$VIDEO_DIR/reference_fetch_slide.mp4" \
  --device cuda 2>&1 | tee "$LOG_DIR/${RUN_TAG}.log"

echo "DONE RUN_TAG=$RUN_TAG MODEL=$CKPT_DIR/${RUN_TAG}.zip VIDEO=$VIDEO_DIR/reference_fetch_slide.mp4"
