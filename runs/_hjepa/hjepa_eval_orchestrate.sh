#!/bin/bash
cd /u5/w223zhan/jepa-mini
source /opt/anaconda3/etc/profile.d/conda.sh; conda activate myenv
export PYTHONNOUSERSITE=1 PYTHONPATH=/u5/w223zhan/jepa-mini MUJOCO_GL=egl
LOG=runs/_hjepa/hjepa_results.log; : > "$LOG"
run_one(){
  local task=$1 jid=$2 lm=$3 gsteps=$4 rr=$5
  local WM=runs/$task/checkpoints/${task}_jepa_model.pt
  local LOW=runs/$task/checkpoints/${task}_pm_${jid}_tqc_best/best_model.zip
  for i in $(seq 1 240); do
    squeue -j $jid -h -o '%t' 2>/dev/null | grep -q . || { [ -f "$LOW" ] && break; }
    [ -f "$LOW" ] && [ -f "$WM" ] && break
    sleep 60
  done
  [ -f "$LOW" ] || LOW=runs/$task/checkpoints/${task}_pm_${jid}_tqc.zip
  echo "=== $task (job $jid) ===" | tee -a "$LOG"
  python scripts/eval_hjepa_maze.py --task $task --low-policy "$LOW" --jepa-model "$WM" \
    --episodes 30 --seed 20000 --graph-steps $gsteps --landmarks $lm --k-reach 25 \
    --reach-radius $rr --subgoal-timeout 60 --device cpu 2>&1 | grep -E '"event"|"policy"' | grep -ivE "warn|deprecat" | tee -a "$LOG"
}
run_one point_medium 1455903 90 200000 0.6
run_one point_large  1455904 130 300000 0.6
echo "HJEPA_ALL_DONE" | tee -a "$LOG"
