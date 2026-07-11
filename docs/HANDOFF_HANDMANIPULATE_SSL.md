# Handoff: Shadow Hand HandManipulate SSL Control

Date: 2026-07-11

## Scope

Initial investigation of the Gymnasium-Robotics `HandManipulateBlock-v1`,
`HandManipulateEgg-v1`, and `HandManipulatePen-v1` suite. These are 20-action,
100-step, sparse-success, in-hand object-pose tasks. The intended direction was
self-supervised DexterousJEPA dynamics plus goal-conditioned MPC.

## Available Data and Artifacts

Each variant has 200k steps / 2,000 episodes of OU-correlated exploration:

- `runs/handmanipulate_block/data/handmanipulate_block_explore.npz`
- `runs/handmanipulate_egg/data/handmanipulate_egg_explore.npz`
- `runs/handmanipulate_pen/data/handmanipulate_pen_explore.npz`

Block world-model checkpoints retained as experiment artifacts:

- `handmanipulate_block_dexterous_jepa.pt`: tokenized 7.16M-parameter
  DexterousJEPA trained for 30k steps on the Block exploration set.
- `handmanipulate_block_dexterous_jepa_rollout.pt`: an experimental second
  training run with direct decoded-rollout state/object-pose supervision.

The experimental code additions made during this round were reverted on user
request. The baseline HandManipulate task presets, collector, model, trainer,
and MPC evaluator from the preceding session remain in the repository.

## Checked Result

The baseline Block controller was evaluated with goal-conditioned latent MPC:

```bash
python scripts/eval_handmanipulate_mpc.py \
  --task handmanipulate_block \
  --model-path runs/handmanipulate_block/checkpoints/handmanipulate_block_dexterous_jepa.pt \
  --episodes 20 --seed 40000 --horizon 8 --candidates 512 --iters 4 \
  --exec-k 1 --disagree-weight 0.05 --device cuda
```

Result: **0.00/20 success**. Do not claim a HandManipulate solve, and do not
run Egg/Pen with the same random-exploration + flat-MPC recipe expecting a
different outcome.

## What Failed and Why

1. **OU exploration is not a dexterous skill dataset.** It provides local
   motion and some contact variation but does not contain sustained,
   goal-directed reorientation trajectories. A future-conditioned action
   planner cannot infer a rare multi-finger contact skill from states it never
   observes reaching the goal.
2. **Flat MPC is not enough to discover a contact sequence.** Sampling action
   chunks from a model trained on exploratory actions did not create a reliable
   grasp/regrasp/rotate primitive.
3. **The first online TQC+HER probe was not evidence of improvement.** The
   sparse-reward run stopped at roughly 2k steps because a generic collapse
   guard was too strict; the resumed run reached 52k steps but remained 0%
   success with mean sparse return -100. It was stopped. These artifacts are
   not a baseline and should not be resumed without a deliberate new objective.

## Recommendation

Do **not** prioritize a hierarchical world model yet. The task already supplies
a single object-pose goal; the missing capability is a low-level contact skill,
not a high-level subgoal router. Hierarchy is worthwhile only after there is a
goal-directed grasp/reorientation controller to execute its intermediate goals.

The next serious attempt should first build a contact-preserving, goal-directed
data source (for example dense physical-pose RL with robust gait/contact
regularization, planner-generated but consistency-filtered trajectories, or
teleoperated/reference trajectories). Then train an object-centric chunk prior
or low-level policy on that data. Evaluate Block first on fresh 20+ episode
slices before spending compute on Egg or Pen.

## Related AntMaze Note

The 300k-step online TD3+HER AntMaze Medium walker was evaluated under the
unchanged flow-macro HWM on 80 episodes: seed blocks
30000/31000/32000/33000 scored **0.25/0.05/0.10/0.15**, or **0.138/80**. It is
worse than the canonical SSL directed-flow walker at 0.39/80. The failed RL
driver was reverted; the HWM/SSL controller remains the canonical AntMaze path.
