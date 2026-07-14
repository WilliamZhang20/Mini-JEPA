# Handoff: AntMaze Flip-Recovery SOTA + HandManipulate Data-Source Round

Date: 2026-07-12

## Scope

Two threads from the "push AntMaze + HandManipulate to SOTA under the JEPA/SSL
framework" session:

1. **AntMaze Medium/Large (COMPLETE):** diagnosed the real bottleneck (ant
   flipping, not gait speed), fixed it with a self-trial + RL-distilled
   recovery flow, and brought the fully-SSL runtime controller to
   historical-HIQL parity on all three mazes.
2. **HandManipulate Block/Egg (IN PROGRESS):** launched the goal-directed
   data-source RL runs the previous handoff recommended. Both plateaued at
   ~0.01 success within a ~2M-step / 8h budget and were stopped. Checkpoints
   are resumable.

## AntMaze: Final State (nothing left to do for these numbers)

| Maze | Old canonical | New | Historical reference |
| --- | --- | --- | --- |
| UMaze | 0.867/60 | **0.95/60** (1.00/0.90/0.95, seeds 30000-32000) | ~0.93 H-JEPA+BC |
| Medium | 0.39/80 | **0.7375/80** (0.75/0.85/0.75/0.60, seeds 30000-33000) | ~0.77 HIQL |
| Large | 0.18/60 | **0.483/60** (0.50/0.40/0.55, seeds 30000-32000) | ~0.54 HIQL |

Canonical eval command (Medium; swap task/paths for umaze/large — UMaze uses
`_flow_directed.pt` since it has no progcond walker and doesn't need one):

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_hjepa_hwm.py --task antmaze_medium \
  --hwm runs/antmaze_medium/checkpoints/antmaze_medium_hwm_s40.pt \
  --macro-flow runs/antmaze_medium/checkpoints/antmaze_medium_hwm_macroflow.pt \
  --bc-policy runs/antmaze_medium/checkpoints/antmaze_medium_flow_progcond.pt \
  --jepa-model runs/antmaze_medium/checkpoints/antmaze_medium_jepa_model.pt \
  --walker-recovery runs/antmaze_medium/checkpoints/antmaze_medium_recovery_flow_v3d.pt \
  --walker-replan 2 --walker-target-progress 1.0 \
  --low-type flow --macro-flow-horizon 1 --reach-radius 1.0 --low-timeout 60 \
  --episodes 20 --seed 30000 --device cuda
```

Story in one paragraph: `--walk-diagnostics` (new) showed the walker spent
38-57% of eval steps flipped on its back — demos contain no flipped states, so
a flipped walker flails off-manifold until the 1000-step timeout. Every speed
lever was negative until recovery existed. Natural recoveries were too rare to
mine (~1.2%/8-step window), so a dense-uprightness SAC specialist
(`train_recovery_rl.py`, flipped-state bank resets, 0.993 self-right success in
~45 steps) was trained as a DATA SOURCE ONLY and its 794 successful recoveries
distilled into an SSL recovery flow (`train_recovery_walker.py` →
`*_recovery_flow_v3d.pt`). The same distill npz
(`runs/antmaze_medium/data/recovery_rl_distill.npz`) re-encoded per-maze
transferred to UMaze/Large with zero maze-specific collection. With the flip
tax bounded, progress-conditioning (`train_flow_walker.py --progress-cond`,
eval `--walker-target-progress 1.0`) inverted from harmful to helpful.
Negative results (all in the ledger): best-of-N progress-greedy chunk
selection, quantile selection, progress-cond without recovery, flip-risk chunk
veto (flip outcome is state-determined before chunk choice), pooling
natural/OU recovery chunks into the distilled flow (never blur the expert
manifold — recovery included).

Artifacts: videos `runs/antmaze_{medium,large}/videos/antmaze_*_ssl_sota.mp4`;
new scripts `train_walker_scorer.py`, `train_flip_risk_scorer.py`,
`train_recovery_walker.py`, `collect_recovery_trials.py`,
`train_recovery_rl.py`; new eval flags in `eval_hjepa_hwm.py`
(`--walker-recovery/--walker-replan/--walker-target-progress/--walker-scorer/
--walker-samples/--walker-select-quantile/--walk-diagnostics/--rollout-out`);
`LowLevelFlow` recovery/selection logic in `jepa_robotics/algos/maze_low_level.py`.
HIQL code/checkpoints retained untouched as the historical comparison.
`PROJECT_STATUS.md` and `EXPERIMENT_LEDGER.md` are updated. All session
changes are uncommitted — see `git status`.

## HandManipulate: Where It Stands

Confirmed (2026-07-12): **no Shadow Hand datasets exist on Minari** (D4RL
"Pen" is Adroit). OU exploration contains zero reorientation successes, so the
SSL prior has nothing to extract — the missing piece is a data source, exactly
as the previous handoff said.

Raw-observation sparse TQC+HER (`train_sb3_reference.py`, the FetchSlide
recipe, deliberately NOT the frozen-OU-JEPA-latent setup that handicapped the
earlier probe) was run on both variants with 8h wall-clock caps:

- **Block:** stopped at ~2.05M steps, success_rate plateau ~0.01 (first
  nonzero at ~1.6M). Latest checkpoint
  `runs/handmanipulate_block/checkpoints/handmanipulate_block_tqc_reference_2050000_steps.zip`.
- **Egg:** stopped at ~1.95M steps, plateau ~0.01 (first nonzero at ~295k, did
  not compound). Latest
  `runs/handmanipulate_egg/checkpoints/handmanipulate_egg_tqc_reference_1950000_steps.zip`.

Both runs were killed by a session restart after their planned analysis, so
`<name>_tqc_reference.zip` (the final `model.save`) does NOT exist — resume
from the numbered checkpoint zips. Note `--resume` in the script expects the
save-model path; either rename the latest numbered checkpoint or point
`--save-model` at it.

Interpretation: consistent with the literature — original HER needed tens of
millions of (parallel) steps for these tasks; ~2M single-worker steps produced
the first flickers of learning, not a usable expert. Not evidence the recipe
fails; evidence the budget was ~10x short.

## Recommended Next Steps (HandManipulate)

1. **Resume Egg first** (it learned earliest) with a much larger budget:
   overnight `--train-seconds 86400`, and consider `learning_starts` already
   satisfied. Watch `success_rate` in the log; a distillable expert needs
   >=0.2-0.3.
2. **Stepping-stone variants:** `HandManipulateBlockRotateZ` /
   rotation-only tasks are far easier and would validate the full
   RL-data → DexterousJEPA → future-conditioned-prior distillation pipeline
   end-to-end before spending big compute on full-pose Block. (Task presets
   would need adding in `jepa_robotics/tasks.py`.)
3. **Distillation once an expert exists** (the Adroit/AntMaze-recovery
   pattern): collect ~1-2k successful episodes from the TQC expert, train
   DexterousJEPA on them (not on OU data), train a future-conditioned
   flow/inverse prior, evaluate SSL-only on 20+ fresh episodes, then apply the
   repo's replacement rules to the RL artifact.
4. The AntMaze session's transferable law: if a needed skill never occurs in
   the data, no amount of prior/scorer/conditioning work will create it —
   manufacture the data first, keep the runtime SSL.

## Loose Ends

- All of this session's code/docs changes are uncommitted (see `git status`);
  the five new scripts + `maze_low_level.py`/`eval_hjepa_hwm.py`/
  `train_flow_walker.py` diffs + two docs + two videos are one coherent commit.
- `runs/antmaze_*/data/selftrials_*.npz`, `recovery_trials_*.npz`,
  `recovery_rl_distill.npz` and the SAC zip are experiment artifacts; the
  distill npz is the one to keep (it regenerates every recovery flow).
- The TQC replay-buffer snapshots from the OLD aborted probe
  (`*_raw_jepa_tqc_*`) in `runs/handmanipulate_block/checkpoints/` predate
  this session and per the previous handoff should not be resumed.
