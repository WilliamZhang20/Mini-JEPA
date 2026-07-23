# Handoff: FrankaKitchen Subtask-Specialist SSL Controller

Date: 2026-07-10

## Result

First **reproducible** full-4 success in the repo, SSL (no runtime BC):
**0.813 full-4 over 150 episodes** (6 seeds × 25 eps), with the wider light
demo bank (`kitchen_subtask_light_arm_wide.pt`).

| seed | 40000* | 50000 | 60000 | 70000 | 80000 | 90000 |
| --- | --- | --- | --- | --- | --- | --- |
| full-4 | 0.80 | 0.84 | 0.76 | 0.76 | 0.92 | 0.80 |
| mean tasks | 3.60 | 3.68 | 3.52 | 3.52 | 3.84 | 3.64 |

`*` seed 40000 was the dev/tuning seed. Held-out 5-seed (50000-90000):
**0.816 / 125 eps**, mean ~3.6/4. Baseline (reproducible raw flow): 0.00 full-4,
2.05 mean. (The earlier 400-segment light specialist scored 0.773/150.)

**Task order is not physically free, but it no longer has to be hardcoded.**
The original greedy scheduler (`--scheduler greedy`) scores only 0.25-0.40
because it ignores downstream handoffs. The newer `--scheduler graph` learns
directed terminal-to-start arm-pose costs from specialist demos and recovers the
viable microwave→kettle→light→slide route even when the requested list is
scrambled. With the all-task learned completion probe that reordered check
scores 0.87/100.

## Controller (canonical)

Four segment-pure per-subtask inverse specialists + firm ground-truth
completion scheduler + per-subtask demo-locked futures + feature-matched
emphasis + arm-pose demo matching with re-lock.

```bash
CKPT=runs/franka_kitchen/checkpoints
python scripts/eval_kitchen_subtask_inverse.py \
  --model-path $CKPT/franka_kitchen_jepa_model.pt \
  --specialist-paths \
    $CKPT/kitchen_subtask_mw.pt \
    $CKPT/kitchen_subtask_kettle.pt \
    $CKPT/kitchen_subtask_light_arm_wide.pt \
    $CKPT/kitchen_subtask_slide.pt \
  --episodes 25 --seed 50000 --exec-k 2 --target-horizon 8 \
  --match-dims 0,9 --geom-weight 3.0 --relock-margin 0.3 \
  --scheduler fixed --order 0,1,2,3 --torch-seed 0 --device auto
```

Specialists (`scripts/train_kitchen_subtask_inverse.py`, all
`--chunk 8 --future-horizons 2,4,8,16 --max-episodes 1200 --train-steps 30000
--batch-size 512 --hidden 768 --n-blocks 5 --concat-raw --emphasis-repeat 8`):

- `kitchen_subtask_mw.pt`     `--subtask 0` (object emphasis, obs 31:32)
- `kitchen_subtask_kettle.pt` `--subtask 1` (object emphasis, obs 32:39)
- `kitchen_subtask_slide.pt`  `--subtask 3` (object emphasis, obs 28:29)
- `kitchen_subtask_light_arm_wide.pt` `--subtask 2 --emphasis-dims 0,9
  --emphasis-repeat 12 --max-segments 800 --seg-pad 4` (**arm-joint emphasis** —
  light switch is approach-dominated; its object dim is near-constant until
  toggled so it carries no approach signal — plus a **wider demo bank** covering
  more post-kettle handoff poses). The earlier `kitchen_subtask_light_arm.pt`
  (400 segments, repeat 8) scored 0.773/150.

## Why it works (the four Relocate laws + two Kitchen levers)

1. **No blurring -> segment-pure specialists.** Each subtask is a distinct
   action manifold; a global policy averages them and stalls mid-sequence.
2. **Future coherence -> per-subtask demo-locked index** tracked h8 ahead.
3. **Firm predicate switch -> ground-truth completion scheduler.** Advance only
   on `info['step_task_completions']`; each subtask is a fresh short problem.
4. **Feature-matched emphasis (law 3 refined).** Emphasize the live feature
   whose error the chunk must null: arm joints for approach (light switch),
   object dims for manipulation (mw/kettle/slide). all-arm and all-object are
   both worse than the mix.
5. **Arm-pose demo matching + re-lock (decisive, 0.65 -> 0.80).** Match demos on
   the arm joints (reachability), not the near-constant object dim, and re-lock
   to a better demo when the arm deviates (`--relock-margin 0.3`). Recovers the
   bad kettle->light handoff locks that were the dominant failure.

## Failure anatomy (the remaining ~0.23)

All residual failures are **light-switch approach misses**: from a minority of
post-kettle arm poses the arm never reaches the switch (light-switch obs stays
at its ~0.68-from-goal rest value; slide/mw/kettle all complete). Completion
threshold is 0.3; successes reach ~0.009. Extra episode budget does not help
(identical at 280 and 400 steps) — it is a wrong-attractor approach failure,
not slowness.

## Recommended next directions (to close toward the old irreproducible 0.90)

1. **Flow light specialist with multimodal candidates.** Train a light-switch
   *flow* specialist (arm emphasis) and sample K approach candidates, selected
   by JEPA rollout or arm-geom score, to find the switch from the hard poses
   where the deterministic inverse commits to one wrong attractor. This is also
   the cleanest place to demonstrate the de-blurred flow prior
   (`train_flat_future_flow.py` now supports `--require-possession` /
   `--emphasis-dims`).
2. **Wider/denser light demo bank** or a nearest-demo re-lock keyed on the
   post-kettle arm pose, to cover the outlier handoff poses.
3. **Contact/approach-consistency check** on the light chunk before committing.

## Do not claim solved beyond this

The numbers above are with `--torch-seed 0` and describe the original
environment-switched reference. The runtime oracle has since been replaced by
`scripts/train_kitchen_completion_probe.py` plus
`eval_kitchen_subtask_inverse.py --completion-mode learned`. The aligned-replay
probe (0.99 threshold, three-frame debounce) reaches 0.784/125 full-4 and
3.58/4 mean on held-out seeds, versus 0.816/125 for the reference. Environment
completion events are scoring-only in that variant.
