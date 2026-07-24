# Canonical checkpoints

PyTorch artifacts under `runs/*/checkpoints/` are stored with Git LFS. After
cloning, run `git lfs pull`. This tree intentionally retains only the best
checked version of each required component; optimizer snapshots, smoke models,
and superseded ablations are not shipped.

Some controllers require more than one artifact. A JEPA world model cannot
replace its future-conditioned action prior or high-level world model, so those
minimal component sets are listed together below.

| Task | Canonical deploy artifacts |
| --- | --- |
| FetchReach | `fetch_reach_jepa_model.pt`, `reach_flow_prior_mh.pt` |
| FetchPush | `push_v2_model.pt`, `push_v2_flow_prior_mh_geom.pt` |
| FetchPickAndPlace | `pickplace_v2_model.pt`, `pickplace_inverse_prior_h8_goalgeom.pt` |
| Fetch multi-task | `fetch_multi_model.pt`, `fetch_multi_policy.pt` |
| FetchSlide | `slide_jepa_beef_scratch_20260613_model.pt`, `slide_equivariant_hwm_v9_calibrated.pt` |
| PointMaze UMaze | `point_umaze_jepa_model.pt`, `point_umaze_hwm_s20.pt`, `point_umaze_hwm_macroflow.pt`, `point_umaze_flow_directed.pt` |
| PointMaze Medium | `point_medium_jepa_model.pt`, `point_medium_hwm_s30.pt`, `point_medium_hwm_macroflow.pt`, `point_medium_flow_directed.pt` |
| PointMaze Large | `point_large_jepa_model.pt`, `point_large_hwm_s40.pt`, `point_large_hwm_macroflow.pt`, `point_large_flow_directed.pt` |
| AntMaze UMaze | `antmaze_umaze_jepa_model.pt`, `antmaze_umaze_hwm_s40.pt`, `antmaze_umaze_hwm_macroflow.pt`, `antmaze_umaze_flow_unified.pt` |
| AntMaze Medium | `antmaze_medium_jepa_model.pt`, `antmaze_medium_hwm_s40.pt`, `antmaze_medium_hwm_macroflow.pt`, `antmaze_medium_flow_unified.pt` |
| AntMaze Large | `antmaze_large_jepa_model.pt`, `antmaze_large_hwm_s40.pt`, `antmaze_large_hwm_macroflow.pt`, `antmaze_large_flow_unified.pt` |
| Adroit Door | `adroit_door_jepa_model.pt`, `door_phase_inverse_h8_p4.pt` |
| Adroit Hammer | `adroit_hammer_jepa_model.pt`, `hammer_phase_inverse_h8_p4.pt` |
| Adroit Pen | `adroit_pen_jepa_model.pt`, `pen_flat_flow_h8_raw.pt` |
| Adroit Relocate | `adroit_relocate_jepa_model.pt`, `relocate_flat_inverse_h8_raw_free_emph_t045.pt`, `relocate_flat_inverse_h8_raw_held_bt_t045.pt` |
| FrankaKitchen | `franka_kitchen_jepa_model.pt`, `kitchen_completion_probe_all7.pt`, and the seven `kitchen_subtask_*.pt` specialists |
| Shadow Block | `handmanipulate_block_dexterous_jepa_rollout.pt` |
| Shadow Block RotateZ | `handmanipulate_block_rotate_z_wm_onpolicy_best.pt` |
| Shadow Block + touch | `handmanipulate_block_touch_grouped_jepa.pt` |
| Shadow Egg | `handmanipulate_egg_dexterous_jepa.pt` (exploratory world model; no solved controller claim) |
| Shadow Pen | No canonical checkpoint yet; training was intentionally deferred to focus on Block. |

The checked headline metrics and exact evaluation protocol live in
`docs/PROJECT_STATUS.md`. A listed world model is not itself a claim that an
open Shadow task is solved.
