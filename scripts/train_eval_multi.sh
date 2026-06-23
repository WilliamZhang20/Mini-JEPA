#!/bin/bash
# Roadmap B end-to-end: ONE JEPA world model + ONE policy for reach + push +
# pick-and-place, via the canonical state adapter.
#
#   1. collect a diverse union (for the world model) and a clean union (for BC)
#   2. train the unified recurrent JEPA world model on the canonical 35-D state
#   3. behaviour-clone one goal-conditioned policy on its latent
#   4. evaluate all three sub-tasks with the single model+policy
#
# Usage: bash scripts/train_eval_multi.sh
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-myenv}"
set -u
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MUJOCO_GL=egl

RUN_TAG="${RUN_TAG:-fetch_multi}"
DIR="runs/fetch_multi"
DATA="$DIR/data"; CKPT="$DIR/checkpoints"; LOG="$DIR/logs"
mkdir -p "$DATA" "$CKPT" "$LOG" "$DIR/eval_results"
WM_NPZ="$DATA/wm_union.npz"
POLICY_NPZ="$DATA/policy_union.npz"
# Deliverables (final model + policy .pt and the videos) live together directly
# under the umbrella dir runs/fetch_multi/; only the bulky optimizer-bearing
# training checkpoint and intermediate data/logs go in subdirs.
MODEL="$DIR/${RUN_TAG}_model.pt"
POLICY="$DIR/${RUN_TAG}_policy.pt"
CKPT_PATH="$CKPT/${RUN_TAG}_checkpoint.pt"

# 1a. Diverse union for the world model (20% random for dynamics coverage).
python scripts/collect_fetch_multi.py \
  --collect-steps "${WM_COLLECT_STEPS:-600000}" --scripted-fraction 0.8 \
  --action-noise 0.2 --controller-gain 12.0 --out "$WM_NPZ"

# 1b. Clean union for behaviour cloning (mostly-expert, low noise).
python scripts/collect_fetch_multi.py \
  --collect-steps "${POLICY_COLLECT_STEPS:-300000}" --scripted-fraction 0.97 \
  --action-noise 0.1 --controller-gain 12.0 --out "$POLICY_NPZ"

# 2. Unified world model (canonical 35-D state read from the npz spec).
python -m jepa_robotics.train \
  --task fetch_multi --episodes-npz "$WM_NPZ" \
  --train-steps "${WM_TRAIN_STEPS:-150000}" --batch-size 512 \
  --horizons 1,2,4,8,16 --latent-dim 128 --hidden-dim 512 \
  --predictor-mode recurrent --lr 3e-4 --ema 0.996 \
  --lambda-pred-probe 0.5 --lambda-pred-achieved 30.0 --lambda-pred-goal 0.2 \
  --lambda-probe 0.2 --lambda-achieved 5.0 --lambda-goal 0.05 --lambda-distance 0.05 \
  --log-every 1000 --save-every 25000 --device cuda \
  --model-path "$MODEL" --save-path "$CKPT_PATH"

# 3. One goal-conditioned policy on the shared latent.
python -m jepa_robotics.train_policy \
  --task fetch_multi --episodes-npz "$POLICY_NPZ" \
  --model-path "$MODEL" --out "$POLICY" \
  --train-steps "${POLICY_TRAIN_STEPS:-40000}" --device cuda

# 4. Per-task evaluation with the single model + policy.
python scripts/eval_fetch_multi.py \
  --model-path "$MODEL" --policy-path "$POLICY" \
  --episodes "${EVAL_EPISODES:-30}" --mpc-candidates 128 --mpc-horizon 12 --cem-iters 4 \
  --video-dir "$DIR" --device cuda \
  --out "$DIR/eval_results/${RUN_TAG}_eval.jsonl"

echo "DONE fetch_multi MODEL=$MODEL POLICY=$POLICY"
