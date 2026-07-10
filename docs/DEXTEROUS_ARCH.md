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

Both were verified end-to-end on CPU (shapes, causal mask, ensemble
disagreement, flow train step + sampler round-trip, and full
train→save→`load_jepa_artifact`→planner-ops). GPU training/eval is pending.
