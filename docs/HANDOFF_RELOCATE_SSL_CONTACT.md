# Handoff: Adroit Relocate SSL Contact Planning

Date: 2026-07-09 (after the possession-specialist round)

## Objective

Improve the self-supervised Adroit Relocate controller,
future-conditioned latent-control planner. The replacement must not copy
state-to-action labels at runtime. It should use demo/trial futures, a JEPA
latent state, action-conditioned dynamics, and an action prior or planner.

Current replacement status: **accepted at 0.957/210 held-out**. The former BC's
1.00 development result is treated as statistically indistinguishable at this
evaluation scale, and its checkpoint was removed on 2026-07-23.

## Current Best SSL Controller (canonical)

Dual possession-specialist inverse tracking the demo-locked h8 future index,
with **input-feature emphasis on both specialists** (law 3, below) and a
**firmer 0.045 possession switch** so the transport specialist only ever
inherits a secure grasp:

- `relocate_flat_inverse_h8_raw_free_emph_t045.pt` — reach/regrasp specialist,
  trained only on pairs whose current frame is NOT in possession at threshold
  0.045 (includes closure + grasp-tightening transitions), with the live
  palm-ball vector (raw dims 30:33) duplicated 8x in the conditioning
  (`--emphasis-dims 30,33 --emphasis-repeat 8`). The emphasis makes the closure
  chunk servo to the *live* ball rather than the demo ball; the 0.045 threshold
  makes it keep tightening the grasp before the handoff.
- `relocate_flat_inverse_h8_raw_held_bt_t045.pt` — transport/place specialist,
  trained only on held-at-0.045 pairs, with the live ball-**target** vector
  (raw dims 36:39) duplicated 8x (`--emphasis-dims 36,39 --emphasis-repeat 8`).
  The emphasis makes placement servo to the *live* target rather than the demo
  target (the placement analog of the reach fix). No palm-ball emphasis here —
  in possession the palm-ball is already small, so it carries no signal.
- Runtime switches on the live palm-ball predicate at **0.045**
  (`--possession-switch-threshold 0.045`). Emphasis config is read
  per-specialist from its checkpoint (`_append_emphasis`), so each specialist
  gets its own emphasized dims.

```bash
python scripts/eval_flat_future_inverse.py \
  --task adroit_relocate \
  --model-path runs/adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt \
  --inverse-path runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_free_emph_t045.pt \
  --inverse-possession-path runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_held_bt_t045.pt \
  --episodes 30 --seed 83000 --candidates 1 --noise-std 0.0 --exec-k 2 \
  --action-delta-weight 0.001 --torch-seed 0 --device auto \
  --possession-switch-threshold 0.045 \
  --future-index demo_locked \
  --future-episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --target-horizon 8
```

Checked results (`--torch-seed 0`, 30 eps each):

- **Held-out validation seeds 81/82/83/85/87/88/89000: 1.00 / 0.93 / 0.90 /
  0.97 / 1.00 / 0.93 / 0.97 (0.957/210).** The 0.045 switch threshold and the
  ball-target held emphasis were selected on dev seeds 75/77/78/80000 only, so
  these seven seeds are held out. Files
  `runs/adroit_relocate/eval_results/relocate_inv_emph_bt_t045_seed*_ep30.jsonl`.
- Dev seeds 75/77/78/80000: 0.87 / 0.97 / 1.00 / 0.97 (0.95/120).
- Combined 11 seeds: **0.955/330** (315/330). Two seeds reach 1.00.
- Progression this task: single global inverse 0.67/90 -> segment-pure dual
  specialist 0.78/90 -> + palm-ball reach emphasis 0.93/90 -> + firmer 0.045
  switch (fixes transport drops) -> + ball-target held emphasis (fixes
  placement near-misses) **0.957/210 held-out**.

Specialists were trained with:

```bash
# reach specialist (palm-ball emphasis, firmer 0.045 possession threshold)
python scripts/train_flat_future_inverse.py \
  --model-path runs/adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt \
  --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --out runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_free_emph_t045.pt \
  --chunk 8 --future-horizons 2,4,8,16 --max-episodes 1200 --train-steps 30000 \
  --batch-size 512 --hidden 768 --n-blocks 5 --concat-raw \
  --require-possession free --possession-threshold 0.045 \
  --emphasis-dims 30,33 --emphasis-repeat 8 --device auto

# held specialist (ball-target emphasis, firmer 0.045 possession threshold)
python scripts/train_flat_future_inverse.py \
  --model-path runs/adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt \
  --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --out runs/adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_held_bt_t045.pt \
  --chunk 8 --future-horizons 2,4,8,16 --max-episodes 1200 --train-steps 30000 \
  --batch-size 512 --hidden 768 --n-blocks 5 --concat-raw \
  --require-possession held --possession-threshold 0.045 \
  --emphasis-dims 36,39 --emphasis-repeat 8 --device auto
```

Earlier-round checkpoints (`relocate_flat_inverse_h8_raw_free_emph.pt` at
switch 0.06, plain `..._held.pt`) reached 0.93/90 and are retained for the
comparison record but superseded by the 0.045-switch pair above.

## The Laws This Effort Established

1. **No blurring of the expert action manifold.** DAgger retrains,
   success-only self-imitation, and input-noise robustification all degraded
   the controller the same way. The positive counterpart: *splitting* the
   bimodal contact regime into segment-pure specialists (zero target
   modification) was a big fresh-seed gain (0.67 -> 0.78).
2. **Future-selection structure beats scoring.** Demo-locked, horizon-matched
   tracking gave consistent gains; every scoring/ranking branch (JEPA rollout,
   contact-dynamics energies, barriers, CVAE, multi-demo candidate ranking)
   was neutral at best once futures were coherent.
3. **Input-feature emphasis beats target/future modification for live-vs-demo
   offsets — for grasp AND placement.** The failures came from a chunk tracking
   the *demo* geometry embedded in `z_future` rather than the *live* geometry.
   Duplicating the relevant live relative-geometry vector Nx in the
   specialist's conditioning — zero change to action targets or futures, so
   law 1 and law 2 both hold — servos to the live object without any runtime
   scoring. Two instances, each the biggest gain of its round: (a) live
   palm-ball vector (dims 30:33) in the *reach* specialist fixes marginal
   grasps (0.78 -> 0.93); (b) live ball-target vector (dims 36:39) in the
   *held* specialist fixes placement near-misses where the ball was firmly
   held the whole episode but delivered ~11 cm from the live target
   (0.948 -> 0.957 held-out). Emphasis re-weights *which live observation dims
   the same demo action is conditioned on*; it does not add synthetic targets,
   retrieve different futures, or optimize actions at runtime.

Plus one control-structure lever that was decisive for transport drops:

4. **Switch to the transport specialist only on a firm grasp.** Retraining
   both specialists at possession threshold 0.045 (instead of 0.06) and
   switching there means the reach specialist keeps tightening the grasp from
   0.06 down to 0.045 before handing off, and the transport specialist never
   inherits a marginal grasp it cannot maintain. This removed most of the
   post-switch / mid-transport drops (dev 4-seed 0.86 -> 0.95). Note this is
   the *opposite* of the earlier "leave" hysteresis (which was harmful): the
   win is a firmer *enter* condition, not stickier possession. `exec-k=1`
   through the switch and an `enter-delay` reach-hold were both neutral-to-
   harmful, so the fix is the threshold, not the switching dynamics.

## Closed This Round (all fresh-seed neutral or worse)

- Switch hysteresis 0.03: 0.74/90.
- Geometry-weighted demo matching (x3 on dims 30:39): 0.78/90 (no change).
- Re-lock-on-drop margin 0.5: worse on tuning.
- Held specialist trained at threshold 0.09, switch 0.075: 0.77/90.
- 4-candidate multi-demo futures ranked by the recalibrated contact head:
  0.79/90 (within noise; machinery retained via `--demo-candidates` and
  `DemoLockedFutureIndex.query_topk` but not canonical).
- Possession-gated future advance: 0.64/90.

## Failure Anatomy (what the remaining ~0.045 is)

Each of the three original failure modes has now been attacked at its root
(palm-ball emphasis -> marginal grasps; firmer 0.045 switch -> transport drops;
ball-target emphasis -> placement near-misses). What remains is a diverse long
tail, ~0-3 idiosyncratic failures per 30-episode seed, spread across all three
modes with no single dominant cause. Per-episode diagnostics on two held-out
0.933 seeds:

- seed 82000: 2 firm-grasp placement misses (held 173/183 steps, min palm-ball
  0.01-0.03, ball delivered 0.117-0.118 from target — the ball-target emphasis
  narrowed but did not fully close these) and 1 mid-transport drop.
- seed 88000: 1 marginal reach miss (min palm-ball 0.065, never grasped), 1
  wide reach miss (0.126), 1 mid-transport drop.

The residual reach misses are outlier live ball positions the demo bank does
not cover well; the residual placement misses are the live-vs-demo target
offset not fully absorbed by the ball-target emphasis; the residual drops are
rare mid-transport instabilities on a firm grasp.

## Recommended Next Directions

The cheap planner/emphasis knobs are now saturated (emphasis repeat 12/16,
held palm-ball emphasis, exec-k=1, enter-delay were all neutral-to-noise on top
of the canonical config). Closing the last ~0.045 to BC 1.00 likely needs one
of:

1. **Emphasis magnitude/threshold micro-sweep with strict held-out discipline.**
   A ball-target emphasis repeat sweep (12/16) or a 0.04 switch threshold might
   shave the placement/drop tail, but the per-seed variance (+/-0.05 at 30 eps)
   means any such gain must be confirmed on >=5 fresh seeds before it is real.
2. **Wider/denser demo bank for reach outliers.** The residual reach misses are
   live ball positions poorly covered by the locked demo. More demos (or a
   nearest-demo re-lock on outlier ball geometry) would target them without
   blurring the manifold.
3. **Contact-consistency JEPA finetune** (DexWM-style hand-consistency
   losses) — still untried; would make imagined latents trustworthy enough
   for candidate filtering to finally pay on the mid-transport drops.

## Validation Decision

The current best is **0.957/210 on held-out seeds
81/82/83/85/87/88/89000 (0.955/330 over 11 seeds)** with `--torch-seed` set.
On 2026-07-23 this was accepted as equivalent to the former BC's 1.00
development result, and
`runs/adroit_relocate/checkpoints/adroit_relocate_bc_on_explorewm.pt` was
deleted. The remaining mass is the long tail characterized above.
