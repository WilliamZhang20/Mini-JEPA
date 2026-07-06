#!/bin/bash
# Train + evaluate + record video for a Fetch object-manipulation task using the
# v2 JEPA stack: recurrent latent dynamics, accurate state decoder, fixed
# scripted experts for data, and the manipulation-aware MPC planner.
#
# Usage: TASK_NAME=fetch_pick_place RUN_TAG=pickplace_v2 bash scripts/train_eval_object_v2.sh
set -eo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-myenv}"
set -u
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MUJOCO_GL=egl

TASK_NAME="${TASK_NAME:-fetch_pick_place}"
case "$TASK_NAME" in
  fetch_pick_place)
    TASK_SLUG="fetch_pick_place"
    SCRIPTED_FRACTION="${SCRIPTED_FRACTION:-0.8}"
    ACTION_NOISE="${ACTION_NOISE:-0.2}"
    MANIP_REACH_WEIGHT="${MANIP_REACH_WEIGHT:-0.1}"
    ACTION_STD="${ACTION_STD:-0.5}"
    ;;
  fetch_push)
    TASK_SLUG="fetch_push"
    SCRIPTED_FRACTION="${SCRIPTED_FRACTION:-0.75}"
    ACTION_NOISE="${ACTION_NOISE:-0.25}"
    # Push must NOT use the gripper->object reach term: a good push contacts the
    # *far* side of the object, so pulling the gripper to the object centre
    # misleads the planner. The learned policy already knows the push approach.
    MANIP_REACH_WEIGHT="${MANIP_REACH_WEIGHT:-0.0}"
    ACTION_STD="${ACTION_STD:-0.3}"
    ;;
  *)
    echo "Unsupported TASK_NAME=$TASK_NAME" >&2; exit 2;;
esac

CONTROLLER_GAIN="${CONTROLLER_GAIN:-12.0}"
RUN_TAG="${RUN_TAG:-${TASK_SLUG}_v2}"
TASK_DIR="runs/$TASK_SLUG"
CKPT_DIR="$TASK_DIR/checkpoints"; LOG_DIR="$TASK_DIR/logs"
VIDEO_DIR="$TASK_DIR/videos"; EVAL_DIR="$TASK_DIR/eval_results"
MODEL_PATH="$CKPT_DIR/${RUN_TAG}_model.pt"
CKPT_PATH="$CKPT_DIR/${RUN_TAG}_checkpoint.pt"
EVAL_LOG="$EVAL_DIR/${RUN_TAG}_eval.jsonl"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$VIDEO_DIR" "$EVAL_DIR"

python -m jepa_robotics.train \
  --task "$TASK_NAME" \
  --output-root runs \
  --seed "${SEED:-41}" \
  --collect-steps "${COLLECT_STEPS:-400000}" \
  --collect-log-every 20000 \
  --scripted-fraction "$SCRIPTED_FRACTION" \
  --controller-gain "$CONTROLLER_GAIN" \
  --action-noise "$ACTION_NOISE" \
  --train-steps "${TRAIN_STEPS:-120000}" \
  --batch-size 512 \
  --horizons 1,2,4,8,16 \
  --latent-dim 128 \
  --hidden-dim 512 \
  --predictor-mode recurrent \
  --lr 3e-4 \
  --ema 0.996 \
  --lambda-pred-probe 0.5 \
  --lambda-pred-achieved 30.0 \
  --lambda-pred-goal 0.2 \
  --lambda-probe 0.2 \
  --lambda-achieved 5.0 \
  --lambda-goal 0.05 \
  --lambda-distance 0.05 \
  --eval-episodes 5 \
  --mpc-candidates 128 \
  --mpc-horizon 12 \
  --log-every 500 \
  --save-every 20000 \
  --device cuda \
  --model-path "$MODEL_PATH" \
  --save-path "$CKPT_PATH"

if [[ "$TASK_NAME" == "fetch_push" ]]; then
  # Stage 2 (push replacement): demos define desirable local futures; a
  # conditional flow proposes action chunks; JEPA dynamics selects the chunk that
  # best realizes the encoded future. No state->action BC policy is trained.
  SUBGOAL_PATH="$CKPT_DIR/${RUN_TAG}_latent_subgoals.pt"
  FLOW_PATH="$CKPT_DIR/${RUN_TAG}_flow_prior.pt"
  python scripts/train_fetch_pick_latent_subgoals.py \
    --task "$TASK_NAME" \
    --model-path "$MODEL_PATH" \
    --out "$SUBGOAL_PATH" \
    --collect-steps "${SUBGOAL_COLLECT_STEPS:-30000}" \
    --controller-gain "$CONTROLLER_GAIN" \
    --action-noise 0.05 \
    --device cuda
  python scripts/train_fetch_flow_prior.py \
    --task "$TASK_NAME" \
    --model-path "$MODEL_PATH" \
    --out "$FLOW_PATH" \
    --collect-steps "${FLOW_COLLECT_STEPS:-100000}" \
    --scripted-fraction 0.9 \
    --controller-gain "$CONTROLLER_GAIN" \
    --action-noise 0.12 \
    --chunk 8 \
    --future-horizons 4,8,12,16 \
    --concat-geometry \
    --train-steps "${FLOW_TRAIN_STEPS:-30000}" \
    --batch-size 512 \
    --flow-steps 32 \
    --device cuda
  python scripts/eval_fetch_flow_jepa.py \
    --task "$TASK_NAME" \
    --model-path "$MODEL_PATH" \
    --flow-path "$FLOW_PATH" \
    --subgoal-path "$SUBGOAL_PATH" \
    --goal-mode local \
    --episodes "${EVAL_EPISODES:-30}" \
    --seed "${EVAL_SEED:-123}" \
    --candidates "${FLOW_CANDIDATES:-32}" \
    --exec-k 1 \
    --flow-steps 16 \
    --target-horizon 8 \
    --latent-weight 1.0 \
    --state-weight 5.0 \
    --final-goal-weight 1.0 \
    --action-l2-weight 0.001 \
    --action-delta-weight 0.01 \
    --device cuda \
    --out "$EVAL_LOG"
  python scripts/record_expert.py --task "$TASK_NAME" --vary-goal --episodes 6 --gain "$CONTROLLER_GAIN"
  echo "DONE TASK=$TASK_NAME MODEL=$MODEL_PATH FLOW=$FLOW_PATH SUBGOALS=$SUBGOAL_PATH EVAL=$EVAL_LOG VIDEO_DIR=$VIDEO_DIR"
else
  # Stage 2 (pick/place replacement): train a self-supervised inverse chunk
  # prior, inverse(z_t, z_goal) -> a_{t:t+H-1}, from transition trials. The
  # policy is not a state->action BC clone; the JEPA model remains the latent
  # world model used to score candidate chunks at evaluation.
  INVERSE_PATH="$CKPT_DIR/${RUN_TAG}_inverse_prior.pt"
  python scripts/train_fetch_inverse_prior.py \
    --task "$TASK_NAME" \
    --model-path "$MODEL_PATH" \
    --out "$INVERSE_PATH" \
    --collect-steps "${INVERSE_COLLECT_STEPS:-80000}" \
    --train-steps "${INVERSE_TRAIN_STEPS:-25000}" \
    --scripted-fraction 1.0 \
    --controller-gain "$CONTROLLER_GAIN" \
    --action-noise 0.03 \
    --chunk 8 \
    --future-horizons 8 \
    --concat-geometry \
    --condition-on-goal-state \
    --batch-size 512 \
    --hidden 512 \
    --device cuda

  python scripts/eval_fetch_inverse_jepa.py \
    --task "$TASK_NAME" \
    --model-path "$MODEL_PATH" \
    --inverse-path "$INVERSE_PATH" \
    --goal-mode final \
    --episodes "${EVAL_EPISODES:-30}" \
    --seed "${EVAL_SEED:-123}" \
    --candidates "${INVERSE_CANDIDATES:-64}" \
    --noise-std 0.05 \
    --exec-k 1 \
    --target-horizon 8 \
    --latent-weight 1.0 \
    --state-weight 5.0 \
    --final-goal-weight 1.0 \
    --action-delta-weight 0.01 \
    --device cuda \
    --out "$EVAL_LOG"

  python scripts/record_expert.py --task "$TASK_NAME" --vary-goal --episodes 6 --gain "$CONTROLLER_GAIN"

  echo "DONE TASK=$TASK_NAME MODEL=$MODEL_PATH INVERSE=$INVERSE_PATH EVAL=$EVAL_LOG VIDEO_DIR=$VIDEO_DIR"
fi
