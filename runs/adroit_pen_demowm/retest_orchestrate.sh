#!/bin/bash
cd /u5/w223zhan/jepa-mini
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate myenv
export PYTHONNOUSERSITE=1 PYTHONPATH=/u5/w223zhan/jepa-mini MUJOCO_GL=egl
export MINARI_DATASETS_PATH=/u5/w223zhan/jepa-mini/.cache/minari
WM=runs/adroit_pen_demowm/checkpoints/adroit_pen_demowm_jepa_model.pt
BC=runs/adroit_pen_demowm/checkpoints/pen_bc_on_demowm.pt
RH=runs/adroit_pen_demowm/checkpoints/pen_reward_head_demowm.pt
LOG=runs/adroit_pen_demowm/retest_result.log
# 1) wait for WM
for i in $(seq 1 120); do
  pgrep -f "task adroit_pen --episodes-npz" >/dev/null 2>&1 || { [ -f "$WM" ] && break; }
  sleep 30
done
sleep 5
echo "=== WM present: $([ -f "$WM" ] && echo yes || echo NO) ===" | tee "$LOG"
[ -f "$WM" ] || { echo "WM missing, abort" | tee -a "$LOG"; exit 1; }
# 2) WM rollout accuracy on demo-WM (is it more precise?)
python scripts/eval_wm_rollout.py --task adroit_pen --model-path "$WM" --episodes 20 --max-horizon 16 --device cuda 2>&1 | grep -E "verdict|^\s+16 " | tee -a "$LOG"
# 3) re-BC on demo-WM latent
python -u -m jepa_robotics.train_policy --task adroit_pen --model-path "$WM" \
  --episodes-npz runs/adroit_pen/data/pen_expert_demos.npz --train-steps 30000 --batch-size 512 --hidden-dim 512 --device cuda --out "$BC" 2>&1 | grep '"event": "policy_saved"' | tee -a "$LOG"
# 4) reward head on demo-WM latent
python scripts/train_adroit_reward_head.py --dataset D4RL/pen/expert-v2 --model-path "$WM" --out "$RH" --max-episodes 1000 --epochs 8 --device cuda 2>&1 | grep reward_head_saved | tee -a "$LOG"
# 5) MPC vs BC on demo-WM
python scripts/eval_adroit_mpc.py --task adroit_pen --model-path "$WM" --policy-path "$BC" --reward-head "$RH" \
  --episodes 20 --seed 20000 --horizon 5 --candidates 200 --cem-iters 2 --std 0.3 --also-bc --device cuda 2>&1 | grep '"policy"' | tee -a "$LOG"
python scripts/eval_adroit_mpc.py --task adroit_pen --model-path "$WM" --policy-path "$BC" --reward-head "$RH" \
  --episodes 20 --seed 20000 --horizon 5 --candidates 200 --cem-iters 2 --std 0.15 --device cuda 2>&1 | grep '"policy"' | tee -a "$LOG"
echo "RETEST_DONE" | tee -a "$LOG"
