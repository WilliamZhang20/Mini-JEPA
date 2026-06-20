#!/bin/bash
cd /u5/w223zhan/jepa-mini
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate myenv
export PYTHONNOUSERSITE=1 PYTHONPATH=/u5/w223zhan/jepa-mini MUJOCO_GL=egl MINARI_DATASETS_PATH=/u5/w223zhan/jepa-mini/.cache/minari
WM=runs/antmaze_umaze/checkpoints/antmaze_umaze_jepa_model.pt
BC=runs/antmaze_umaze/checkpoints/antmaze_umaze_bc_low.pt
DEMOS=runs/antmaze_umaze/data/antmaze_umaze_demos.npz
LOG=runs/antmaze_umaze/hjepa_result.log; : > "$LOG"
# 1) wait for WM
for i in $(seq 1 120); do
  pgrep -f "task antmaze_umaze --episodes-npz" >/dev/null 2>&1 || { [ -f "$WM" ] && break; }
  sleep 30
done
sleep 5
echo "WM present: $([ -f "$WM" ] && echo yes || echo NO)" | tee -a "$LOG"
[ -f "$WM" ] || { echo abort | tee -a "$LOG"; exit 1; }
# 2) BC goal-conditioned low-level on antmaze demos
python -u -m jepa_robotics.train_policy --task antmaze_umaze --model-path "$WM" \
  --episodes-npz "$DEMOS" --train-steps 40000 --batch-size 512 --hidden-dim 512 --device cuda --out "$BC" 2>&1 | grep '"event": "policy_saved"' | tee -a "$LOG"
# 3) H-JEPA flat-vs-hierarchical (graph from demo trajectories; scaled for ant maze)
python scripts/eval_hjepa_maze.py --task antmaze_umaze --low-type bc --bc-policy "$BC" --jepa-model "$WM" \
  --graph-npz "$DEMOS" --episodes 30 --seed 20000 --landmarks 50 --k-reach 30 \
  --reach-radius 1.5 --subgoal-timeout 80 --device cuda 2>&1 | grep -E '"event"|"policy"' | grep -ivE "warn|deprecat" | tee -a "$LOG"
echo "ANTMAZE_HJEPA_DONE" | tee -a "$LOG"
