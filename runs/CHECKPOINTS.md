# Canonical checkpoints

This directory contains the minimal inference-time artifact set for the best
checked controller in each task. These are not resumable training snapshots:
none contains optimizer, scheduler, gradient, or replay-buffer state.
Superseded ablations and smoke checkpoints are intentionally omitted.

## Storage and downloading

The files are stored with Git LFS because of their **aggregate** size, not
because any individual checkpoint is unusually large:

| Measure | Value |
| --- | ---: |
| Checkpoint files | 56 |
| Total checked-out size | 1.156 GB |
| Median file | 21.9 MB |
| Largest file | 84.9 MB |
| Files at or below 10 MB | 20 |

Every file would fit in ordinary Git individually. Putting the complete set in
ordinary Git would, however, add the full 1.156 GB to the repository's object
history and to code-only clones. LFS keeps lightweight pointers in Git while
allowing either the complete set or one task to be fetched:

```bash
# Fetch every checkpoint after cloning.
git lfs pull

# Or clone source without checkpoint payloads, then fetch one task.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/WilliamZhang20/Mini-JEPA.git
cd Mini-JEPA
git lfs pull --include="runs/fetch_slide/checkpoints/*.pt"
```

A reader who downloads all checkpoints still transfers 1.156 GB; LFS only
avoids forcing that cost on a source-only clone.

## What the artifact types contain

All files are dictionaries written with `torch.save`. Load only checkpoints
from a trusted source because PyTorch checkpoint loading can execute pickled
Python objects.

| Artifact type | Stored payload | Runtime role |
| --- | --- | --- |
| JEPA world model | `model` weights; state `normalizer` mean/std; environment `spec`; architecture/training `config` | Encodes the current/target state, predicts action-conditioned future latents, and exposes state/distance/contact probes. Dexterous variants use the same envelope around transformer weights. |
| Inverse action prior | `state_dict`; conditioning/action dimensions; chunk horizon `H`; architecture metadata; JEPA `model_path`; sometimes future-state retrieval banks or task segments | Deterministically proposes an action chunk conditioned on current and desired-future representations. |
| Flow action prior | Online `state_dict`/`flow` and, where trained, preferred EMA weights; flow schedule and conditioning metadata; sometimes retrieval banks | Samples multiple coherent action chunks from a future-conditioned rectified-flow model. JEPA or task geometry selects among them. |
| Hierarchical world model (HWM) | `psi` high-level encoder; `macro` action-chunk GRU; `g` abstract macro dynamics; `dec` position-subgoal decoder; dimensions, stride, and macro statistics | Predicts at a longer time scale and emits a reachable position subgoal for the low-level controller. |
| HWM macro-flow prior | Online and EMA flow weights plus macro/HWM dimensions | Proposes feasible high-level macro transitions conditioned on abstract state and final goal. |
| Ballistic HWM | Equivariant model `state_dict`; JEPA dependency; feature normalization; ensemble/architecture settings; calibration metadata | Predicts FetchSlide's absorbing post-coast latent and puck endpoint from a pre-impact state and one strike macro. |
| Completion probe | Classifier `state_dict`; seven task names; learned thresholds; input/model metadata | Detects completed Kitchen subtasks from frozen JEPA latent plus live normalized state, without using the environment completion signal for control. |
| Task policy | `policy` weights and model dimensions/dependency | Direct latent-to-action policy used by the Fetch multi-task experiment. |

The `model_path` fields name dependencies, not duplicate model weights. Some
inverse/flow artifacts also embed the small state/future index or demonstration
segments needed to choose a reachable target at runtime. Those arrays explain
why similarly sized neural networks can produce differently sized files.

## Fetch

| File | Size | What it stores and why it is needed |
| --- | ---: | --- |
| `fetch_reach/checkpoints/fetch_reach_jepa_model.pt` | 8.1 MB | Reach JEPA encoder, EMA target encoder, rollout latent dynamics, state/distance probes, normalization, and `FetchReach-v4` dimensions. It is the model used by JEPA MPC. |
| `fetch_reach/checkpoints/reach_flow_prior_mh.pt` | 39.7 MB | Optional H=8 multi-horizon rectified-flow action proposer, including online and EMA weights, goal-geometry normalization, and its Reach JEPA dependency. |
| `fetch_push/checkpoints/push_v2_model.pt` | 8.1 MB | Push recurrent JEPA world model and preprocessing/spec metadata. |
| `fetch_push/checkpoints/push_v2_flow_prior_mh_geom.pt` | 39.8 MB | H=8 multi-horizon flow proposer conditioned on current/future JEPA latents plus normalized live goal geometry; JEPA ranks sampled chunks. |
| `fetch_pick_place/checkpoints/pickplace_v2_model.pt` | 8.1 MB | Pick-and-place recurrent JEPA world model and preprocessing/spec metadata. |
| `fetch_pick_place/checkpoints/pickplace_inverse_prior_h8_goalgeom.pt` | 3.8 MB | H=8 deterministic inverse action prior conditioned on current/future latent and live goal state geometry. It references the Pick-and-Place JEPA. |
| `fetch_multi/checkpoints/fetch_multi_model.pt` | 8.1 MB | Shared Fetch multi-task JEPA world model, normalizer, observation spec, and architecture config. |
| `fetch_multi/checkpoints/fetch_multi_policy.pt` | 1.3 MB | Goal-conditioned latent-to-action policy weights and the shared JEPA model dependency. This is the only retained direct task-policy bundle. |
| `fetch_slide/checkpoints/slide_jepa_beef_scratch_20260613_model.pt` | 25.4 MB | Deeper residual recurrent Slide JEPA (`H_max=24`, four transition blocks), normalizer, state/action spec, encoder/target encoder, and probes. |
| `fetch_slide/checkpoints/slide_equivariant_hwm_v9_calibrated.pt` | 35.7 MB | Goal-frame equivariant ballistic HWM ensemble, Fourier/contact feature statistics, strike-duration bounds and distance bins, validation/calibration metadata, and the Slide JEPA dependency. This is the endpoint predictor behind the 0.853/2000 result. |

## PointMaze

Each maze size uses the same four-stage deployment graph:

```text
JEPA state encoder -> HWM abstract state -> macro-flow reachable hop
                   -> decoded xy subgoal -> directed low-level flow walker
```

| File | Size | What it stores |
| --- | ---: | --- |
| `point_umaze/checkpoints/point_umaze_jepa_model.pt` | 3.3 MB | UMaze JEPA encoder, target encoder, primitive latent dynamics and probes, with state normalization/spec. |
| `point_umaze/checkpoints/point_umaze_hwm_s20.pt` | 1.0 MB | HWM weights (`psi`, action-chunk `macro` GRU, abstract dynamics `g`, xy decoder `dec`) with a 20-step macro stride. |
| `point_umaze/checkpoints/point_umaze_hwm_macroflow.pt` | 21.9 MB | EMA/online rectified-flow prior over feasible HWM macro-actions, conditioned on abstract state and final xy goal. |
| `point_umaze/checkpoints/point_umaze_flow_directed.pt` | 2.7 MB | Directed low-level flow walker: goal-conditioned primitive action chunks plus conditioning/dimension metadata. |
| `point_medium/checkpoints/point_medium_jepa_model.pt` | 3.3 MB | Medium JEPA encoder, target encoder, primitive latent dynamics and probes, with state normalization/spec. |
| `point_medium/checkpoints/point_medium_hwm_s30.pt` | 1.0 MB | Medium HWM with the same four modules and a 30-step macro stride. |
| `point_medium/checkpoints/point_medium_hwm_macroflow.pt` | 21.9 MB | Medium EMA/online macro-flow prior conditioned on abstract state and final xy goal. |
| `point_medium/checkpoints/point_medium_flow_directed.pt` | 2.7 MB | Medium directed low-level flow walker and conditioning metadata. |
| `point_large/checkpoints/point_large_jepa_model.pt` | 3.3 MB | Large JEPA encoder, target encoder, primitive latent dynamics and probes, with state normalization/spec. |
| `point_large/checkpoints/point_large_hwm_s40.pt` | 1.0 MB | Large HWM with the same four modules and a 40-step macro stride. |
| `point_large/checkpoints/point_large_hwm_macroflow.pt` | 21.9 MB | Large EMA/online macro-flow prior conditioned on abstract state and final xy goal. |
| `point_large/checkpoints/point_large_flow_directed.pt` | 2.7 MB | Large directed low-level flow walker and conditioning metadata. |

## AntMaze

AntMaze uses the same HWM deployment graph as PointMaze. Its low-level flow is
larger because it is a condition-modulated unified ant controller trained on
both directed locomotion and auxiliary self-righting transitions.

| File | Size | What it stores |
| --- | ---: | --- |
| `antmaze_umaze/checkpoints/antmaze_umaze_jepa_model.pt` | 27.5 MB | UMaze JEPA world model, normalization, environment spec, and architecture config. |
| `antmaze_umaze/checkpoints/antmaze_umaze_hwm_s40.pt` | 1.1 MB | 40-step HWM: `psi`, macro-action GRU, abstract macro predictor, xy subgoal decoder, and macro normalization statistics. |
| `antmaze_umaze/checkpoints/antmaze_umaze_hwm_macroflow.pt` | 21.9 MB | EMA/online macro-flow prior that keeps high-level proposals on demonstrated feasible transitions. |
| `antmaze_umaze/checkpoints/antmaze_umaze_flow_unified.pt` | 41.7 MB | Unified FiLM/residual low-level flow walker, progress statistics, and auxiliary-behavior mixture metadata. |
| `antmaze_umaze/checkpoints/antmaze_umaze_discrete_topology_router.pt` | 79 KB | Seven-region next-waypoint classifier distilled from official UMaze map shortest paths; 72/100 official fixed-pair evaluation, no map query at inference. |
| `antmaze_medium/checkpoints/antmaze_medium_jepa_model.pt` | 11.4 MB | Medium JEPA world model, normalization, environment spec, and architecture config. |
| `antmaze_medium/checkpoints/antmaze_medium_hwm_s40.pt` | 1.1 MB | Medium 40-step HWM and macro normalization statistics. |
| `antmaze_medium/checkpoints/antmaze_medium_hwm_macroflow.pt` | 21.9 MB | Medium EMA/online feasible macro-flow prior. |
| `antmaze_medium/checkpoints/antmaze_medium_flow_unified.pt` | 41.7 MB | Medium unified FiLM/residual low-level flow walker, progress statistics, and auxiliary-behavior mixture metadata. |
| `antmaze_large/checkpoints/antmaze_large_jepa_model.pt` | 11.4 MB | Large JEPA world model, normalization, environment spec, and architecture config. |
| `antmaze_large/checkpoints/antmaze_large_hwm_s40.pt` | 1.1 MB | Large 40-step HWM and macro normalization statistics. |
| `antmaze_large/checkpoints/antmaze_large_hwm_macroflow.pt` | 21.9 MB | Large EMA/online feasible macro-flow prior. |
| `antmaze_large/checkpoints/antmaze_large_flow_unified.pt` | 41.7 MB | Large unified FiLM/residual low-level flow walker, progress statistics, and auxiliary-behavior mixture metadata. |

The three unified walkers learn when auxiliary recovery behavior is useful from
the state itself. There is no hand-written runtime recovery mode or RL value
function.

**UMaze official-eval update (2026-07-26).** The continuous HWM remains the
random-pair controller and scores 0/100 on Minari's official fixed pair. The
discrete topology router above replaces only that high-level component and
scores **72/100** with the unchanged unified walker. It is learned and frozen at
inference, but its shortest-route supervision is generated from the official
maze map; this provenance is part of the checkpoint contract.

## Adroit

| File | Size | What it stores and why it is needed |
| --- | ---: | --- |
| `adroit_door/checkpoints/adroit_door_jepa_model.pt` | 27.6 MB | Door JEPA world model, state normalization/spec, and architecture metadata. |
| `adroit_door/checkpoints/door_phase_inverse_h8_p4.pt` | 28.3 MB | H=8 four-phase inverse prior plus 50k-entry current/future state, phase, and progress retrieval bank. The bank supplies phase-consistent desirable futures; the network maps them to action chunks. |
| `adroit_hammer/checkpoints/adroit_hammer_jepa_model.pt` | 6.8 MB | Hammer JEPA world model and preprocessing/spec metadata. |
| `adroit_hammer/checkpoints/hammer_phase_inverse_h8_p4.pt` | 31.8 MB | H=8 four-phase inverse prior and its 50k-entry future/progress retrieval bank. |
| `adroit_pen/checkpoints/adroit_pen_jepa_model.pt` | 27.7 MB | Pen JEPA world model and preprocessing/spec metadata. |
| `adroit_pen/checkpoints/pen_flat_flow_h8_raw.pt` | 68.1 MB | H=8 rectified-flow action prior with online/EMA weights, current/future retrieval bank, and raw-state-plus-latent conditioning metadata. |
| `adroit_relocate/checkpoints/adroit_relocate_jepa_model.pt` | 27.6 MB | Relocate JEPA world model and preprocessing/spec metadata. |
| `adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_free_emph_t045.pt` | 34.4 MB | H=8 reach/grasp inverse specialist, retrieval bank, raw-state conditioning, palm-to-ball feature emphasis, and `free` possession regime metadata. |
| `adroit_relocate/checkpoints/relocate_flat_inverse_h8_raw_held_bt_t045.pt` | 34.4 MB | H=8 held/transport inverse specialist, retrieval bank, raw-state conditioning, ball-to-target emphasis, and `held` possession regime metadata. |

The two Relocate specialists are both required: the live 0.045 possession
predicate chooses which one proposes the next chunk. Neither file is a BC or RL
policy.

## FrankaKitchen

| File | Size | What it stores and why it is needed |
| --- | ---: | --- |
| `franka_kitchen/checkpoints/franka_kitchen_jepa_model.pt` | 27.7 MB | Three-head recurrent Kitchen JEPA world model, normalizer, 59-D state/9-D action spec, encoder/target encoder, dynamics, and probes. |
| `franka_kitchen/checkpoints/kitchen_completion_probe_all7.pt` | 1.0 MB | Seven-label completion classifier, task-specific thresholds, validation metadata, and the Kitchen JEPA dependency. |
| `franka_kitchen/checkpoints/kitchen_subtask_mw.pt` | 19.2 MB | Microwave H=8 inverse specialist plus its segment-pure demonstration bank and object-feature emphasis metadata. |
| `franka_kitchen/checkpoints/kitchen_subtask_kettle.pt` | 18.3 MB | Kettle H=8 inverse specialist plus its segment-pure demonstration bank and object-feature emphasis metadata. |
| `franka_kitchen/checkpoints/kitchen_subtask_light_arm_wide.pt` | 39.8 MB | Light-switch H=8 inverse specialist with the wider 800-segment reachability bank and repeated arm-pose features. |
| `franka_kitchen/checkpoints/kitchen_subtask_slide.pt` | 24.1 MB | Slide-cabinet H=8 inverse specialist plus its segment-pure demonstration bank and object-feature emphasis. |
| `franka_kitchen/checkpoints/kitchen_subtask_top_burner.pt` | 16.9 MB | Top-burner H=8 inverse specialist, all-seven task metadata, and its demonstration segment bank. |
| `franka_kitchen/checkpoints/kitchen_subtask_bottom_burner.pt` | 23.5 MB | Bottom-burner H=8 inverse specialist, all-seven task metadata, and its demonstration segment bank. |
| `franka_kitchen/checkpoints/kitchen_subtask_hinge_cabinet_v2_strict.pt` | 18.0 MB | Hinge-cabinet H=8 strict segment-pure inverse specialist; cross-segment training is disabled in its saved metadata. |

The completion probe chooses when a specialist is done. The learned
demonstration-handoff graph chooses a feasible order for a requested subset.
All seven specialist files are needed for arbitrary subsets; the four-task
headline evaluation uses microwave, kettle, light switch, and slide cabinet.

## Shadow Dexterous Hand

| File | Size | What it stores and current claim |
| --- | ---: | --- |
| `handmanipulate_block/checkpoints/handmanipulate_block_dexterous_jepa_rollout.pt` | 28.7 MB | Transformer-tokenized Block JEPA: online/EMA encoders, causal action-chunk dynamics, state/contact probes, normalizer, and task/architecture config. It is a world model, not a solved full-pose controller. |
| `handmanipulate_block_rotate_z/checkpoints/handmanipulate_block_rotate_z_wm_onpolicy_best.pt` | 84.9 MB | Larger RotateZ DexterousJEPA ensemble with object-geodesic supervision and on-policy self-goal calibration. Smooth iCEM uses this model directly; no separate policy checkpoint is bundled. |
| `handmanipulate_block_touch/checkpoints/handmanipulate_block_touch_grouped_jepa.pt` | 28.9 MB | Continuous-touch Block DexterousJEPA with all 92 taxels grouped into 17 anatomical palm/phalanx tokens, plus normalizer/spec/config. This is the best tactile representation checkpoint but remains controller-bound. |
| `handmanipulate_egg/checkpoints/handmanipulate_egg_dexterous_jepa.pt` | 28.5 MB | Exploratory Egg DexterousJEPA world model. It is retained as a representation/adaptation starting point, not as a solved Egg controller. |

Shadow Pen has no canonical checkpoint; its training was intentionally deferred
to focus on Block.

## Integrity and loading

At publication, all 56 files passed `torch.load`; every embedded `.pt`
dependency existed; and every model/prior/HWM/probe/policy was reconstructed
from its saved metadata and ran a forward pass. The artifact loader for JEPA
world models is `jepa_robotics.evaluate.load_jepa_artifact`; specialized
evaluation scripts reconstruct their controller bundle from the fields
documented above.

The checked headline metrics and exact evaluation protocols live in
`docs/PROJECT_STATUS.md` and `docs/EXPERIMENT_LEDGER.md`. Listing a world model
here is not a claim that an open Shadow task is solved.
