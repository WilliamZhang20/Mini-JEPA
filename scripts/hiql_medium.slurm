#!/bin/bash
# Tuned HIQL push to SOTA on AntMaze-Medium-Diverse, on a dedicated GPU.
# Stability fixes over the first run (which peaked ~0.48 but oscillated): AWR
# temperature beta=1.0 (was 3.0), cosine LR decay + grad clip (built into
# train_hiql.py), and best-checkpoint selection on a 50-episode metric. Then
# validate the best checkpoint at 100 episodes and record a video.
#SBATCH --job-name=hiql-med
#SBATCH --partition=YV2PANT
#SBATCH --exclude=watgpu208,watgpu408
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-myenv}"
set -u
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MUJOCO_GL=egl
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MINARI_DATASETS_PATH="$REPO_ROOT/.cache/minari"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"; mkdir -p "$TMPDIR"

python - <<'PY'
import torch, sys
print(f'{{"event":"gpu","name":"{torch.cuda.get_device_name(0)}"}}' if torch.cuda.is_available()
      else '{"event":"fatal","reason":"no_cuda"}', flush=True)
sys.exit(0 if torch.cuda.is_available() else 17)
PY

S=antmaze_medium; CK=runs/$S/checkpoints
WM=$CK/${S}_jepa_ctrl_model.pt
DEMOS=runs/$S/data/${S}_demos.npz

python -u scripts/train_hiql.py --task $S --model-path "$WM" --episodes-npz "$DEMOS" \
  --out $CK/${S}_hiql_tuned.pt \
  --steps "${STEPS:-1000000}" --expectile 0.7 --beta-low 1.0 --beta-high 1.0 --subgoal-k 25 \
  --eval-every 100000 --eval-episodes 50 --device cuda

echo '{"event":"stage","name":"validate_best_100ep"}'
python scripts/eval_hiql.py --task $S --model-path "$WM" \
  --hiql $CK/${S}_hiql_tuned_best.pt --episodes 100 --seed 20000 \
  --record-out runs/$S/videos/antmaze_medium_hiql_tuned.mp4 --device cuda

echo '{"event":"done","task":"'"$S"'"}'
