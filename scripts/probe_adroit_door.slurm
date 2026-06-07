#!/bin/bash
#SBATCH --job-name=jepa-adroit-probe
#SBATCH --partition=YV2PANT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
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
mkdir -p runs/adroit_door/checkpoints

python -m jepa_robotics.train \
  --task adroit_door \
  --output-root runs \
  --collect-steps 2000 \
  --train-steps 20 \
  --batch-size 64 \
  --latent-dim 64 \
  --hidden-dim 256 \
  --predictor-mode rollout \
  --device cpu \
  --save-path runs/adroit_door/checkpoints/probe_checkpoint.pt \
  --model-path runs/adroit_door/checkpoints/probe_model.pt
