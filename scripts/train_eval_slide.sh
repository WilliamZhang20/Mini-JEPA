#!/bin/bash
# Train + evaluate + record FetchSlide with the v2 JEPA stack, beefed up for the
# ballistic striking dynamics: deeper recurrent latent dynamics
# (--transition-depth 2), longer horizons (up to 24 steps) so the world model
# sees the puck coast after contact, and the calibrated scripted "slide" striker
# for data. Sized to run in ~2h on a single RTX 6000 Ada.
#
# Usage: bash scripts/train_eval_slide.sh
set -eo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-myenv}"
set -u
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MUJOCO_GL=egl

TASK_NAME=fetch_slide
TASK_SLUG=fetch_slide
RUN_TAG="${RUN_TAG:-slide_v1}"
CONTROLLER_GAIN="${CONTROLLER_GAIN:-12.0}"
TASK_DIR="runs/$TASK_SLUG"
CKPT_DIR="$TASK_DIR/checkpoints"; LOG_DIR="$TASK_DIR/logs"
VIDEO_DIR="$TASK_DIR/videos"; EVAL_DIR="$TASK_DIR/eval_results"
MODEL_PATH="$CKPT_DIR/${RUN_TAG}_model.pt"
CKPT_PATH="$CKPT_DIR/${RUN_TAG}_checkpoint.pt"
POLICY_PATH="$CKPT_DIR/${RUN_TAG}_policy.pt"
EVAL_LOG="$EVAL_DIR/${RUN_TAG}_eval.jsonl"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$VIDEO_DIR" "$EVAL_DIR"

TRAIN_EXTRA_ARGS=()
if [ -n "${RESUME_PATH:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--resume-path "$RESUME_PATH")
fi
if [ "${NO_RESUME_OPTIMIZER:-0}" = "1" ]; then
  TRAIN_EXTRA_ARGS+=(--no-resume-optimizer)
fi

# Stage 1: world model. Deeper recurrent dynamics + long horizons for coasting.
python -m jepa_robotics.train \
  --task "$TASK_NAME" \
  --output-root runs \
  --seed "${SEED:-41}" \
  --collect-steps "${COLLECT_STEPS:-250000}" \
  --collect-log-every 25000 \
  --scripted-fraction "${SCRIPTED_FRACTION:-0.7}" \
  --controller-gain "$CONTROLLER_GAIN" \
  --action-noise "${ACTION_NOISE:-0.2}" \
  --train-steps "${TRAIN_STEPS:-55000}" \
  --batch-size 512 \
  --horizons 1,2,4,8,16,24 \
  --latent-dim 192 \
  --hidden-dim 512 \
  --predictor-mode recurrent \
  --transition-depth 2 \
  --lr 3e-4 \
  --ema 0.996 \
  --lambda-var "${LAMBDA_VAR:-0.02}" \
  --lambda-cov "${LAMBDA_COV:-0.0}" \
  --lambda-pred-probe 0.5 \
  --lambda-pred-achieved 30.0 \
  --lambda-pred-goal 0.2 \
  --lambda-pred-cov "${LAMBDA_PRED_COV:-0.0}" \
  --lambda-probe 0.2 \
  --lambda-achieved 5.0 \
  --lambda-goal 0.05 \
  --lambda-distance 0.05 \
  --eval-episodes 5 \
  --mpc-candidates 128 \
  --mpc-horizon 20 \
  --log-every 500 \
  --save-every 10000 \
  --device cuda \
  --model-path "$MODEL_PATH" \
  --save-path "$CKPT_PATH" \
  "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/${RUN_TAG}_train.log"

# Stage 2: behaviour-clone the goal-conditioned action prior (the striker skill).
python -m jepa_robotics.train_policy \
  --task "$TASK_NAME" \
  --model-path "$MODEL_PATH" \
  --out "$POLICY_PATH" \
  --collect-steps "${POLICY_COLLECT_STEPS:-150000}" \
  --train-steps "${POLICY_TRAIN_STEPS:-30000}" \
  --scripted-fraction 0.97 \
  --controller-gain "$CONTROLLER_GAIN" \
  --action-noise 0.1 \
  --device cuda 2>&1 | tee "$LOG_DIR/${RUN_TAG}_policy.log"

# Stage 3: evaluate random / scripted / policy-only / policy+world-model-MPC.
# Slide is a push-like contact task: no reach term (contact the far side), score
# on the object->goal distance over a long planning horizon.
python -m jepa_robotics.evaluate \
  --task "$TASK_NAME" \
  --output-root runs \
  --model-path "$MODEL_PATH" \
  --policy-path "$POLICY_PATH" \
  --policy-proposal-fraction 0.5 \
  --episodes "${EVAL_EPISODES:-20}" \
  --seed "${EVAL_SEED:-123}" \
  --mpc-method cem \
  --mpc-score manip \
  --mpc-candidates "${MPC_CANDIDATES:-96}" \
  --mpc-horizon 20 \
  --cem-iters 3 \
  --elite-frac 0.1 \
  --action-std "${ACTION_STD:-0.4}" \
  --manip-reach-weight 0.0 \
  --manip-path-weight 0.3 \
  --device cuda \
  --out "$EVAL_LOG" \
  --video-policy none \
  --video-dir "$VIDEO_DIR" \
  --fps 30 2>&1 | tee "$LOG_DIR/${RUN_TAG}_eval.log"

echo "DONE TASK=$TASK_NAME MODEL=$MODEL_PATH POLICY=$POLICY_PATH EVAL=$EVAL_LOG VIDEO_DIR=$VIDEO_DIR"
