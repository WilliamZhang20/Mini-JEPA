#!/bin/bash
#SBATCH --job-name=jepa-rollout-eval
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

MODEL_PATH="${MODEL_PATH:-runs/fetch_reach/checkpoints/fetchreach_rollout_model.pt}"
MPC_METHOD="${MPC_METHOD:-cem}"
MPC_SCORE="${MPC_SCORE:-latent}"
EPISODES="${EPISODES:-20}"
CANDIDATES="${CANDIDATES:-256}"
CEM_ITERS="${CEM_ITERS:-5}"
VIDEO_POLICY="${VIDEO_POLICY:-none}"
PROPOSAL_FRACTION="${PROPOSAL_FRACTION:-0.0}"
ACTION_L2_WEIGHT="${ACTION_L2_WEIGHT:-0.0}"
ACTION_DELTA_WEIGHT="${ACTION_DELTA_WEIGHT:-0.0}"
EXECUTE_SMOOTHING="${EXECUTE_SMOOTHING:-0.0}"
GRAD_ITERS="${GRAD_ITERS:-30}"
GRAD_LR="${GRAD_LR:-0.08}"

python -m jepa_robotics.evaluate \
  --task fetch_reach \
  --output-root runs \
  --model-path "$MODEL_PATH" \
  --episodes "$EPISODES" \
  --mpc-candidates "$CANDIDATES" \
  --mpc-horizon 8 \
  --mpc-method "$MPC_METHOD" \
  --mpc-score "$MPC_SCORE" \
  --cem-iters "$CEM_ITERS" \
  --grad-iters "$GRAD_ITERS" \
  --grad-lr "$GRAD_LR" \
  --elite-frac 0.1 \
  --action-std 0.7 \
  --action-l2-weight "$ACTION_L2_WEIGHT" \
  --action-delta-weight "$ACTION_DELTA_WEIGHT" \
  --execute-smoothing "$EXECUTE_SMOOTHING" \
  --jepa-scripted-proposal-fraction "$PROPOSAL_FRACTION" \
  --device auto \
  --video-policy "$VIDEO_POLICY" \
  --out "runs/fetch_reach/eval_results/rollout_${MPC_METHOD}_${MPC_SCORE}_${SLURM_JOB_ID}.jsonl" \
  --video-dir runs/fetch_reach/videos
