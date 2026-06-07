#!/bin/bash
#SBATCH --job-name=jepa-cross-eval
#SBATCH --partition=YV2PANT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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
mkdir -p "$TMPDIR" runs/fetch_reach/eval_results runs/fetch_reach/videos

MODEL_PATH="${MODEL_PATH:-runs/fetch_reach/checkpoints/fetchreach_jepa_model.pt}"
python -m jepa_robotics.evaluate \
  --task fetch_reach \
  --output-root runs \
  --model-path "$MODEL_PATH" \
  --episodes 50 \
  --mpc-candidates 128 \
  --mpc-horizon 8 \
  --device auto \
  --video-policy all \
  --out "runs/fetch_reach/eval_results/fetchreach_cross_eval_${SLURM_JOB_ID}.jsonl" \
  --video-dir runs/fetch_reach/videos
