# Handoff: Adroit Relocate SSL Contact Planning

Date: 2026-07-09 (after the possession-specialist round)

## Objective

Replace the retained Adroit Relocate BC controller with a self-supervised,
future-conditioned latent-control planner. The replacement must not copy
state-to-action labels at runtime. It should use demo/trial futures, a JEPA
latent state, action-conditioned dynamics, and an action prior or planner.

Current replacement status: **not solved, but the gap has closed from ~0.53 to
0.93 fresh-seed in three sessions.** Retain the BC checkpoint until an SSL
planner matches or beats the 1.00 checked BC result.

## Current Best SSL Controller (canonical)

Dual possession-specialist inverse tracking the demo-locked h8 future index,
with a **palm-ball-emphasis reach specialist** for closure micro-correction:

- `relocate_flat_inverse_h8_raw_free_emph.pt` — reach/regrasp specialist,
  trained only on pairs whose current frame is NOT in possession (includes
  closure transitions), with the live palm-ball vector (raw dims 30:33)
  duplicated 8x in the conditioning (`--emphasis-dims 30,33 --emphasis-repeat
  8`). This upweights live contact geometry so the closure chunk servos to the
  *live* ball rather than the demo ball. Targets and futures are unchanged, so
  the expert action manifold is not blurred.
- `relocate_flat_inverse_h8_raw_held.pt` — transport/place specialist, trained
  only on pairs whose current AND future frames are held (palm-ball < 0.06).
  No emphasis (in possession the palm-ball is already small, so it conveys
  little).
- Runtime switches on the live palm-ball predicate at 0.06. Emphasis config is
  read per-specialist from its checkpoint, so only the reach specialist gets
  the duplicated dims.

```bash
python scripts/eval_flat_future_inverse.py \
  --task adroit_relocate \
  --model-path runs/adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt \
  --inverse-path runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_free_emph.pt \
  --inverse-possession-path runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_held.pt \
  --episodes 30 --seed 80000 --candidates 1 --noise-std 0.0 --exec-k 2 \
  --action-delta-weight 0.001 --torch-seed 0 --device auto \
  --future-index demo_locked \
  --future-episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --target-horizon 8
```

Checked results (`--torch-seed 0`, 30 eps each):

- **Untouched validation seeds 80000/81000/82000: 0.90 / 1.00 / 0.90
  (0.93/90).** These seeds were used for no tuning or selection. Files
  `runs/adroit_relocate/eval_results/relocate_inv_emph_seed8*000_ep30.jsonl`.
  Same-seed baseline (non-emphasis reach specialist): 0.77 / 0.87 / 0.73
  (0.79/90).
- Fresh seeds 75000/77000/78000: 0.80 / 0.867 / 0.867 (**0.84/90**), files
  `runs/adroit_relocate/eval_results/relocate_inv_dualspec_emph_fresh_seed*_ep30.jsonl`.
  Same-seed non-emphasis baseline: 0.77 / 0.73 / 0.83 (0.78/90).
- Combined six fresh seeds: emphasis **0.89/180** vs non-emphasis 0.78/180.
  Every seed improved; none regressed, from a single untuned config
  (`--emphasis-repeat 8`).

Specialists were trained with:

```bash
# reach specialist (with closure emphasis)
python scripts/train_flat_future_inverse.py \
  --model-path runs/adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt \
  --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --out runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_free_emph.pt \
  --chunk 8 --future-horizons 2,4,8,16 --max-episodes 1200 --train-steps 30000 \
  --batch-size 512 --hidden 768 --n-blocks 5 --concat-raw \
  --require-possession free --emphasis-dims 30,33 --emphasis-repeat 8 --device auto

# held specialist (unchanged, no emphasis)
python scripts/train_flat_future_inverse.py \
  --model-path runs/adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt \
  --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --out runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_held.pt \
  --chunk 8 --future-horizons 2,4,8,16 --max-episodes 1200 --train-steps 30000 \
  --batch-size 512 --hidden 768 --n-blocks 5 --concat-raw \
  --require-possession held --device auto
```

## The Three Laws This Effort Established

1. **No blurring of the expert action manifold.** DAgger retrains,
   success-only self-imitation, and input-noise robustification all degraded
   the controller the same way. The positive counterpart: *splitting* the
   bimodal contact regime into segment-pure specialists (zero target
   modification) was a big fresh-seed gain (0.67 -> 0.78).
2. **Future-selection structure beats scoring.** Demo-locked, horizon-matched
   tracking gave consistent gains; every scoring/ranking branch (JEPA rollout,
   contact-dynamics energies, barriers, CVAE, multi-demo candidate ranking)
   was neutral at best once futures were coherent.
3. **Input-feature emphasis beats target/future modification for closure.**
   The marginal-grasp failures came from the reach chunk tracking the *demo*
   ball geometry embedded in `z_future`, not the *live* ball. Duplicating the
   live palm-ball vector (raw dims 30:33) 8x in the reach specialist's
   conditioning — with zero change to action targets or futures, so law 1 and
   law 2 both hold — was the single biggest fresh-seed gain of the whole
   effort (0.78 -> 0.93 on untouched seeds). Emphasis re-weights *which live
   observation dims the same demo action is conditioned on*; it does not add
   synthetic targets, retrieve different futures, or optimize actions at
   runtime. This is the first intervention that made the controller sensitive
   to the live-vs-demo ball offset without any runtime scoring.

## Closed This Round (all fresh-seed neutral or worse)

- Switch hysteresis 0.03: 0.74/90.
- Geometry-weighted demo matching (x3 on dims 30:39): 0.78/90 (no change).
- Re-lock-on-drop margin 0.5: worse on tuning.
- Held specialist trained at threshold 0.09, switch 0.075: 0.77/90.
- 4-candidate multi-demo futures ranked by the recalibrated contact head:
  0.79/90 (within noise; machinery retained via `--demo-candidates` and
  `DemoLockedFutureIndex.query_topk` but not canonical).
- Possession-gated future advance: 0.64/90.

## Failure Anatomy (what the remaining ~0.07 is)

The palm-ball emphasis closed most of the marginal-grasp mass. On seed 77000
(0.867, was 0.733) the reach failure count collapsed: 29/30 now grasp, and the
remaining failures shifted from marginal reach to early transport drops.
Per-episode diagnostics on the untouched seed 82000 (0.90, 3 failures /30):

- 1 still-marginal miss: min palm-ball 0.062 — right at the 0.06 boundary, the
  residual of the same calibration error.
- 1 wider miss: min palm-ball 0.093 — a harder reach the emphasis did not
  close.
- 1 early transport drop: grasped, held only ~5 steps, dropped just after the
  switch to the held specialist.

So the failure profile is now roughly split three ways (still-marginal reach /
wider reach / early post-switch drop) instead of dominated by marginal reach.
The post-switch drop is a new leading failure and is a *held-specialist /
switch-boundary* problem, not a reach problem.

## Recommended Next Directions

1. **Emphasis sweep + held-side closure.** `--emphasis-repeat 8` was the
   first untuned guess and already moved 0.78 -> 0.93. A small sweep (4/12/16)
   and applying the same emphasis idea to the held specialist's ball-target
   vector (raw dims 36:39) could close the post-switch drops. Keep the
   validation discipline: tune on 75000-78000, confirm on untouched seeds.
2. **Post-switch drop fix.** The remaining drops happen right after the reach
   -> held handoff. Options that respect the laws: a short exec-k=1 window
   through the switch so the held specialist re-plans immediately on the fresh
   grasp state; or a brief hysteresis so the reach specialist (which just
   achieved the grasp) keeps control for 1-2 steps into possession before the
   held specialist takes over.
3. **Contact-consistency JEPA finetune** (DexWM-style hand-consistency
   losses) — still untried; would make imagined latents trustworthy enough
   for candidate filtering to finally pay.

## Do Not Claim Solved Until

A fresh 30-episode eval on seeds not used for tuning or selection, with
`--torch-seed` set, matching or beating the retained BC 1.00. Current best is
**0.93/90 on untouched seeds 80000/81000/82000**, so
`runs/adroit_relocate/checkpoints/adroit_relocate_bc_on_explorewm.pt` stays.
The gap is now ~0.07; the remaining mass is characterized above.
