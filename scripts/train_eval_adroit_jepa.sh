#!/usr/bin/env bash
# Full JEPA latent-control pipeline for one flat Adroit task.
#
#   TASK=adroit_hammer bash scripts/train_eval_adroit_jepa.sh
#
# Stages, in order. Every training stage is skipped when its artifact already
# exists, so an interrupted run resumes at the first missing artifact:
#   1. exploration trials -- colored-noise action episodes for the world model
#   2. world model        -- dense multi-step rollout objective
#                            (algos/world_models/latent_jepa)
#   3. high level         -- learned z_t -> (z_{t+h}, state, progress) subgoal
#                            net (algos/latent_subgoal)
#   4. actor              -- future-conditioned chunk prior on
#                            (z_t, z_goal, horizon) (algos/priors.InversePrior)
#   5. controller         -- predictor-free baseline, the ranked controller,
#                            and the ablations that test which component is
#                            load-bearing
#
# Nothing here supplies task structure by hand: no phase count, no schedule, no
# switch threshold, no emphasis dims. The only task-specific inputs are the
# demonstrations and the environment id.
set -euo pipefail

TASK=${TASK:?set TASK, e.g. adroit_hammer}
SLUG=${TASK#adroit_}
ROOT=${ROOT:-runs/${TASK}}
DATA_ALL=${DATA_ALL:-${ROOT}/data/${SLUG}_all.npz}
DATA_OK=${DATA_OK:-${ROOT}/data/${SLUG}_success.npz}
TRIALS=${TRIALS:-${ROOT}/data/${SLUG}_colored_trials.npz}
WM=${WM:-${ROOT}/checkpoints/${SLUG}_jepa_planner_v2.pt}
SG=${SG:-${ROOT}/checkpoints/${SLUG}_latent_subgoal_v2.pt}
ACTOR=${ACTOR:-${ROOT}/checkpoints/${SLUG}_latent_actor_v2.pt}
EVAL=${EVAL:-${ROOT}/eval_results/jepa_latent_control.jsonl}
LOGS=${LOGS:-${ROOT}/logs}

WM_STEPS=${WM_STEPS:-20000}
SG_STEPS=${SG_STEPS:-12000}
ACTOR_STEPS=${ACTOR_STEPS:-15000}
HORIZON=${HORIZON:-4}
EXEC_K=${EXEC_K:-2}
RANK_HORIZONS=${RANK_HORIZONS:-2,4,8}
EPISODES=${EPISODES:-30}
SEED=${SEED:-64000}
TRIAL_EPISODES=${TRIAL_EPISODES:-400}
DEVICE=${DEVICE:-auto}

mkdir -p "${LOGS}" "$(dirname "${WM}")" "$(dirname "${EVAL}")"

echo "== 1/5 exploration trials =="
if [ ! -f "${TRIALS}" ]; then
  python scripts/data/collect_latent_trials.py --task "${TASK}" --policy colored \
    --episodes "${TRIAL_EPISODES}" --out "${TRIALS}" --device cpu | tail -1
fi

echo "== 2/5 world model =="
if [ ! -f "${WM}" ]; then
  python scripts/train/train_latent_jepa_wm.py --task "${TASK}" \
    --episodes-npz "${DATA_ALL}" "${TRIALS}" \
    --horizon 16 --latent-dim 128 --hidden-dim 512 \
    --transition-depth 2 --ensemble-heads 4 --inverse-horizon "${HORIZON}" \
    --train-steps "${WM_STEPS}" --batch-size 512 --log-every 2000 \
    --model-path "${WM}" --device "${DEVICE}" 2>&1 | tee "${LOGS}/wm_v2.log" | grep '"event"' | tail -3
fi

echo "== 3/5 learned high level =="
if [ ! -f "${SG}" ]; then
  python scripts/train/train_latent_subgoal.py --model-path "${WM}" \
    --episodes-npz "${DATA_OK}" --out "${SG}" \
    --horizons 2,4,8,16 --train-steps "${SG_STEPS}" --batch-size 1024 \
    --log-every 4000 --device "${DEVICE}" 2>&1 | tee "${LOGS}/subgoal_v2.log" | grep '"event"' | tail -2
fi

echo "== 4/5 learned actor =="
if [ ! -f "${ACTOR}" ]; then
  python scripts/train/train_latent_actor.py --model-path "${WM}" \
    --episodes-npz "${DATA_OK}" --out "${ACTOR}" \
    --chunk "${HORIZON}" --horizons "${RANK_HORIZONS}" \
    --train-steps "${ACTOR_STEPS}" --batch-size 1024 \
    --device "${DEVICE}" 2>&1 | tee "${LOGS}/actor_v2.log" | grep '"event"' | tail -2
fi

echo "== 5/5 controller + ablations =="
# Predictor-free baseline: the actor's own proposal executed directly.
python scripts/eval/eval_latent_jepa_mpc.py --task "${TASK}" \
  --model-path "${WM}" --subgoal-path "${SG}" --actor-path "${ACTOR}" \
  --proposal inverse-only --horizon "${HORIZON}" --exec-k "${EXEC_K}" \
  --episodes "${EPISODES}" --seed "${SEED}" \
  --tag "actor_only_h${HORIZON}k${EXEC_K}" \
  --out "${EVAL}" --device "${DEVICE}" | tail -1
# Ranked controller, then re-randomize one component at a time.
for ABL in none predictor encoder subgoal actor; do
  python scripts/eval/eval_latent_jepa_mpc.py --task "${TASK}" \
    --model-path "${WM}" --subgoal-path "${SG}" --actor-path "${ACTOR}" \
    --proposal inverse --rank-horizons "${RANK_HORIZONS}" \
    --horizon "${HORIZON}" --exec-k "${EXEC_K}" \
    --episodes "${EPISODES}" --seed "${SEED}" --ablate "${ABL}" \
    --out "${EVAL}" --device "${DEVICE}" | tail -1
done
