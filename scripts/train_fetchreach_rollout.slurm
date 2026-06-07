#!/bin/bash
#SBATCH --job-name=jepa-rollout
#SBATCH --partition=YV2PANT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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
mkdir -p "$TMPDIR" runs/fetch_reach/checkpoints

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
  --eval-episodes 3 \
  --mpc-candidates 128 \
  --mpc-horizon 8 \
  --log-every 100 \
  --device auto \
  --save-path "runs/fetch_reach/checkpoints/fetchreach_rollout_checkpoint_${SLURM_JOB_ID}.pt" \
  --model-path "runs/fetch_reach/checkpoints/fetchreach_rollout_model_${SLURM_JOB_ID}.pt"
