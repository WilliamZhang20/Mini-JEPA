# FetchSlide Video Manifest

Clean top-level videos are named by algorithm and measured success rate.
Original filenames and nested recorder outputs are preserved in `_archive_original_names_20260613/`.

- `reference_tqc_rl_success087_ep30.mp4`: TQC+HER reference, 0.867 success over 30 episodes.
- `jepa_teacher_bc_clean_policy_success077_ep30.mp4`: clean TQC-teacher distillation into JEPA latent policy, 0.767 success over 30 episodes.
- `jepa_latent_rl_tqc_policywarm25k_best_success080_ep30.mp4`: JEPA-latent TQC policy-warmup continuation best checkpoint, 0.800 success over 30 confirmation episodes.
- `jepa_latent_rl_tqc_policywarm25k_740k_success090_ep10.mp4`: JEPA-latent TQC policy-warmup 740k checkpoint, 0.900 success over 10 searched/confirmed episodes.
- `jepa_teacher_bc_policy_success070_ep20.mp4`: earlier TQC-teacher distillation into JEPA latent policy, 0.700 success over 20 episodes.
- `jepa_latent_rl_tqc_best_success070_ep10.mp4`: pre-collapse JEPA-latent TQC EvalCallback best checkpoint, 0.700 success over 10 triage episodes.
- `jepa_vicreg_resume_policy_success067_ep6.mp4`: proper regularized JEPA resume policy, 0.667 success over 6 recorded episodes.
- `jepa_latent_rl_tqc_625k_success033_ep6.mp4`: JEPA-latent TQC RL checkpoint at 625k steps, 0.333 success over 6 episodes.
- `jepa_vicreg_cont_policy_success035_ep20.mp4`: locally recovered continued-regularized JEPA policy, 0.350 success over 20 episodes.
- `jepa_vicreg_cont_mpc_success017_ep6.mp4`: continued-regularized JEPA model-only CEM/MPC, 0.167 success over 6 episodes.
- `jepa_latent_rl_tqc_collapsed_success017_ep30.mp4`: earlier collapsed/conservative JEPA-latent TQC run, 0.167 success over 30 episodes.
- `jepa_latent_rl_tqc_local_iter_success000_ep_unknown.mp4`: short local iteration video; deterministic eval was 0.000.
