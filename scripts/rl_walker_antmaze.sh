#!/bin/bash
# One RL round to lift the AntMaze walker past its SSL imitation ceiling, then
# re-eval under the (unchanged) flow-macro HWM high level. RUN IN THE GPU SESSION.
#
# Why RL here: chunk-16 (0.0-0.1), hard fast-filter (0.125) and speed-weighted
# (0.34) imitation walkers all sit at/under the directed walker's 0.39 on Medium
# -- imitation cannot exceed the wandering demos' top speed. Online TD3+HER lets
# the ant experience falling/recovery and discover a faster gait (behavior not in
# the demos). The high level stays 100% SSL; RL only refines the motor primitive.
#
# Usage:  bash scripts/rl_walker_antmaze.sh antmaze_medium
#         bash scripts/rl_walker_antmaze.sh antmaze_large
set -e
M=${1:-antmaze_medium}
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate /u5/w223zhan/.conda/envs/myenv
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
P=runs/$M; J=$P/checkpoints/${M}_jepa_model.pt; D=$P/data/${M}_demos.npz
mkdir -p $P/logs

# Stage 1 - offline TD3+BC on the CURRENT jepa (GPU-bound; ~0.27 walker warm start)
python scripts/train_offline_td3bc.py --task $M --model-path $J --episodes-npz $D \
  --out $P/checkpoints/${M}_td3bc_v2.pt --steps 200000 --batch-size 2048 --device cuda

# Stage 2 - online TD3+HER (falling/recovery beyond the imitation ceiling)
python scripts/train_online_td3_her.py --task $M --model-path $J --episodes-npz $D \
  --init-actor $P/checkpoints/${M}_td3bc_v2.pt --out $P/checkpoints/${M}_td3_online_v2.pt \
  --env-steps 300000 --batch-size 2048 --updates-per-step 4 --device cuda

# Eval the RL walker as the low level under the unchanged flow-macro HWM high level
STRIDE=40; [ "$M" = "antmaze_umaze" ] && STRIDE=40
for seed in 30000 31000 32000 33000; do
  echo -n "seed=$seed: "
  python scripts/eval_hjepa_hwm.py --task $M \
    --hwm $P/checkpoints/${M}_hwm_s${STRIDE}.pt \
    --macro-flow $P/checkpoints/${M}_hwm_macroflow.pt --macro-flow-samples 16 --macro-flow-horizon 1 \
    --bc-policy $P/checkpoints/${M}_td3_online_v2.pt --low-type bc \
    --jepa-model $J --episodes 20 --seed $seed --reach-radius 1.0 --low-timeout 90 \
    --skip-diagnostics --device cuda 2>&1 | grep success_rate
done
