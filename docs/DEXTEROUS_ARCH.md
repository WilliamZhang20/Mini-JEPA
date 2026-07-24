# Dexterous-Hand Architecture (Shadow Hand)

A model architecture sized for the Adroit **Shadow Dexterous Hand** tasks
(24–30 DoF actions, 39–46 D obs; contact-rich, multimodal). The shallow MLP
`ActionConditionedJEPA` (6.9M) and MLP flow prior under-fit this regime — an MLP
mixes all joint/object/contact dims densely with no relational bias, and the MLP
flow prior mode-averages multimodal finger actions. Two transformer upgrades
live in `jepa_robotics/models/dexterous.py`:

## `DexterousJEPA` — tokenized transformer world model (~16.4M)

- **Per-DoF tokenization.** Every state scalar is its own token (shared value
  projection + a learned per-dim channel embedding) plus a `[LATENT]` summary
  token. Self-attention therefore models **joint↔joint↔object↔contact**
  interactions explicitly — the structure in-hand manipulation needs — instead
  of an MLP's dense mash.
- **Anatomical tactile tokenization.** Touch-enabled Shadow Hand observations
  append 92 taxels. `token_groups` maps them into 17 palm/phalanx patches while
  leaving proprioception, object pose, and goals scalar-tokenized. The
  Block-touch encoder therefore uses 93 tokens rather than 168 without dropping
  a sensor.
- **Causal transformer latent dynamics.** Sequence `[z, a_0..a_{H-1}]` with a
  causal mask; the output at step *i* predicts the latent after `a_0..a_i`
  (verified: perturbing `a_{H-1}` leaves step-0's rollout latent unchanged,
  moves step-(H-1)'s). Longer-range contact credit than a GRU cell.
- **Ensemble dynamics heads** → an epistemic **disagreement** signal for
  contact-uncertain regions (MPC cost / exploration), same interface as the MLP
  model's `disagreement`.
- **Contact-consistency head** (DexWM-style): decodes a configurable
  relative-geometry slice (e.g. Relocate palm-ball+ball-target `30:39`) so the
  latent retains the fingertip/object detail contact control depends on, on top
  of the full `state_probe`.
- **Drop-in**: exposes `encode / encode_target / update_target / reset_target /
  predict_rollout / predict / disagreement / state_probe`; loads via
  `load_jepa_artifact` when `config["arch"]="dexterous"`, so every existing
  planner (`eval_flat_future_inverse`, flow priors, HWM) runs on it unchanged.

## `DexterousFlowPrior` — DiT action-chunk prior (~16.8M)

- Each action step is a token; conditioning `(z_t, z_future)` + flow time
  modulate every block via **AdaLN-Zero** (DiT). Models the *distribution* over
  dexterous action chunks rather than averaging it (the failure mode of the MLP
  flow prior on Relocate). Matches the `EpsNet(x, t, cond) → velocity` signature,
  so it slots straight into the shared `sample_action_chunks` sampler and the
  existing future-conditioned flow trainers.

## Training

`scripts/train_dexterous_jepa.py` (world model): JEPA next-latent prediction +
VICReg + state-probe + contact-consistency, EMA target, multi-horizon. Saves the
standard artifact with `arch="dexterous"`. Example (GPU):

```bash
python scripts/train_dexterous_jepa.py --task adroit_relocate \
  --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
  --out runs/adroit_relocate/checkpoints/relocate_dexterous_jepa.pt \
  --horizons 1,2,4,8,16 --latent-dim 192 --d-model 256 --enc-depth 4 \
  --dyn-depth 4 --heads 8 --ensemble-heads 3 --contact-dims 30,39 \
  --steps 60000 --batch-size 256 --device cuda
```

The `DexterousFlowPrior` is a drop-in for `EpsNet` in the future-conditioned
flow trainers (`train_flat_future_flow.py`) via its matching forward signature;
pass `action_dim` and `chunk_dim=H*action_dim` at construction.

Both were verified end-to-end (shapes, causal mask, ensemble disagreement,
flow train step + sampler round-trip, and full
train→save→`load_jepa_artifact`→planner-ops).

## Smooth contact planning and tactile Block

The Shadow videos exposed discontinuities at iCEM replan boundaries. Colored
noise makes each sampled plan smooth internally, but does not constrain the
first action of the next plan relative to the command just executed. The shared
planning objective now projects every candidate onto a previous-action
conditioned actuator-rate constraint *before* JEPA rollout scoring. This is not
an output-only visual filter: the world model scores the exact feasible smooth
trajectory that the hand executes.

On three independent RotateZ seed blocks (3 episodes each), a 0.35
per-actuator step limit plus a small 0.02 delta cost retained the baseline
success count (1/9), reduced action-delta RMS from roughly 0.50 to 0.22, and
reduced command-jerk RMS from roughly 0.75 to 0.29. Median gap improved on the
hard 61000 block (118.8° to 61.3°) and on 62000 (24.9° to 13.4°).

`handmanipulate_block_touch` targets
`HandManipulateBlock_ContinuousTouchSensors-v1`. Its reward-free dataset has
60,000 smooth OU steps and 92 continuous touch channels. The anatomically
grouped 7.19M-parameter JEPA improved five-episode median terminal goal cost
from 2.357 (scalar-taxel JEPA) to 2.023, but did not solve full Block (0/5) and
still trailed the no-touch model's 1.867. This is a useful representation
improvement and an honest controller-bound negative, not a tactile success
claim.
