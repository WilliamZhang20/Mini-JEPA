# CLAUDE.md — jepa-mini project guide

Action-conditioned **JEPA world model** + learned **goal-conditioned policy** + **model-predictive
control (MPC)** for Gymnasium-Robotics manipulation. The world model is a *predictor*, not a
controller: the BC policy proposes actions and the world-model MPC refines them.

## Repo map
- `jepa_robotics/models/` — package: `world_model.py` (`ActionConditionedJEPA`: encoder + EMA target +
  `direct`/`rollout`/`recurrent` predictor + **K-head ensemble** + `state_probe`/`distance_probe`),
  `policy.py` (`GoalConditionedPolicy`), `mlp.py`, `regularizers.py` (VICReg variance+covariance, normalized-MSE).
- `jepa_robotics/scoring/` — per-task MPC score mixins (`manip`/`strike`/`goal`/`common`) composed into `JEPAMPCPolicy`.
- `jepa_robotics/train.py` — WM training (multi-horizon JEPA loss + probes + VICReg; `--ensemble-heads`,
  `--episodes-npz` for offline data, non-goal-env support).
- `jepa_robotics/train_policy.py` — BC of the policy on the **frozen** JEPA latent (`--episodes-npz`, `--her-relabel-frac`).
- `jepa_robotics/evaluate.py` — `JEPAMPCPolicy` (random/CEM/grad; `latent`/`state`/`combined`/`manip`/`strike`,
  `--open-loop`/`--replan-window`) + Random/Scripted/SB3/LearnedPolicyOnly baselines.
- `jepa_robotics/data.py` — episode collection, scripted experts (`reach`/`push`/`pick_place`/`slide`/`maze`),
  `load_episodes_npz`, `Normalizer`.
- `jepa_robotics/envs.py` — `make_env` (Maze success→is_success alias, AntMaze continuing-task off), `ObsSpec`, `flatten_obs`.
- `jepa_robotics/sb3_jepa.py` — `JEPALatentExtractor` (HER on latent), `JEPAEncoderExtractor`/`JEPAConcatExtractor`
  (trainable encoder), `JEPALatentObsWrapper`.
- `jepa_robotics/tasks.py` — `TASKS` dict (Fetch / slide / PointMaze / AntMaze / Adroit).
- Roadmap-B scripts (unified Fetch controller): `collect_fetch_multi.py` (canonical union →npz),
  `eval_fetch_multi.py` (per-task success for one model+policy), `train_eval_multi.sh` (full pipeline).
- Key scripts: `train_jepa_sb3_policy.py` (TQC+HER on latent; `--demo-npz` offline seeding),
  `train_adroit_*.py` (teacher / reward-head / controller), `minari_to_npz.py` (D4RL→Episode npz),
  `eval_hjepa_maze.py` (Hierarchical-JEPA subgoal graph), `eval_wm_rollout.py` (WM accuracy probe),
  `train_eval_antmaze_hjepa.slurm` / `uncap_antmaze_hjepa.slurm`.

## Environment / run notes
- Conda env `myenv`; `MUJOCO_GL=egl`; H200 GPU (`--device cuda`). `gymnasium-robotics>=1.2`, action space is
  4-D `(dx,dy,dz,gripper)` for **all** Fetch tasks.
- Per-task obs widths differ: **FetchReach** `observation`=10 → `state_dim`=16; **Push/PickPlace/Slide**
  `observation`=25 → `state_dim`=31. This mismatch is the central obstacle for a unified model (see below).

## Status (best success rate per task)
| Tier | Task | Best agent | Success |
|------|------|-----------|---------|
| base | FetchReach-v4 | JEPA+MPC (grad, state) | 95–100% |
| base | FetchPush-v4 | JEPA policy + CEM | 100% |
| base | FetchPickAndPlace-v4 | JEPA policy + CEM (manip) | 100% |
| base | **fetch_multi (one model+policy: reach+push+pick)** | unified JEPA policy+MPC (canonical adapter) | **1.00 / 0.97 / 1.00** (mean 0.99) |
| 1 | FetchSlide-v4 | JEPA-latent TQC+HER | 0.83 |
| 2 | PointMaze U/Med/Large | **H-JEPA** (subgoal graph) | 1.00 / 0.90 / 1.00 |
| 2 | AntMaze UMaze | H-JEPA (BC low-level) | 0.93 |
| 2 | AntMaze Medium | H-JEPA (control-aware TD3+BC low) | 0.27 (0.00→0.27; walker-capped) |
| 2 | AntMaze Large | H-JEPA | low (walker-capped) |
| 3 | Adroit Door/Hammer/Pen/Relocate | JEPA-latent BC on offline demos | 0.96 / 1.00 / 0.77 / 1.00 |
| 4 | FrankaKitchen-v1 | **control-aware-JEPA skill-hierarchy + online self-imitation** | **0.90 full-4 success** (3.88/4 sub-tasks) |

Tiers 1–4 essentially cleared. The three recurring lessons (see roadmaps + README):
**(a)** JEPA's *encoder/representation* is what carries control (BC/RL/diffusion act in the latent); its
*predictor* only pays off for planning on **smooth** dynamics (Fetch reach/push, H-JEPA high level),
not contact-rich (slide/Adroit/kitchen → model exploitation, predictor worse-than-no-op). **(b)** Long-horizon
mazes/kitchen need **hierarchy** (H-JEPA subgoal graph; subtask skill-hierarchy), not flat MPC/HER.
**(c)** Contact-rich long-horizon manipulation (kitchen) needs: an *inverse-dynamics control-aware* encoder
(plain VICReg/predictive latent sheds contact precision — trails raw obs), an *action-chunked flow/diffusion*
policy (per-step BC compounds error + averages multimodal demos to mush), and *self-imitation/DAgger* to
bootstrap scarce full-sequence data.

---

# Difficulty ranking: what to tackle next (next-hardest → ultimate-hardest)

We have solved the Fetch **reach / push / pick-and-place** triad (4-D action, single rigid object, 3-D goal,
dense scriptable experts). The list below ranks the remaining Gymnasium-Robotics envs by how much *new*
machinery they demand on top of what we have. Each tier states **why it is harder** and **what our stack needs**.

### Tier 1 — FetchSlide-v4  *(smallest step up; do this first)*
Same Fetch arm, same 4-D action, same obs layout as Push. **Harder because** the puck is on a low-friction
surface and the goal is *out of reach*: the arm must impart the right momentum in a single strike, then has
**no corrective authority** after contact (open-loop ballistics). Receding-horizon MPC that re-plans every
step is poorly matched to a "commit then watch" task.
*Needs:* a longer planning horizon, a friction/contact-accurate world model (the current GRU dynamics
underestimate post-contact coasting), and a scripted "strike" expert for data. Almost no new infra — reuses
the entire pipeline. **Best ROI as the immediate next task.**

### Tier 2 — PointMaze (UMaze → Medium → Large), then AntMaze
Navigation with **sparse reward** and **long horizons** (hundreds of steps), where the straight-line-to-goal
heuristic our experts rely on fails (walls). **Harder because** success requires *sub-goal planning* — the
world model must support multi-step lookahead/graph search, not one-shot MPC.
*Needs:* hindsight relabeling (HER) for data, a sub-goal proposer (latent-space planning / RRT over the world
model), and a maze-aware reward/score. **AntMaze** adds an 8-DoF quadruped locomotion controller underneath
the navigation — a second control problem stacked on the first, so rank it after PointMaze.

### Tier 3 — Adroit hand suite: Door → Hammer → Pen → Relocate
24–28-D action, anthropomorphic hand. **Harder because** of high-DoF contact-rich control and the death of the
hand-written expert: there is no simple geometric controller for finger coordination, so our "scripted teacher
+ BC" data engine breaks. Internal ordering: **Door** (articulated, fixed pivot) < **Hammer** (dynamic
strike + tool use) < **Pen** (in-hand reorientation) < **Relocate** (pick + transport + place of a ball with a
hand). `AdroitHandDoor-v1` is already stubbed in `tasks.py` (`controller="none"`) and unsolved.
*Needs:* a learned (not scripted) data source — offline RL datasets or an RL teacher — plus a stochastic
world model to handle contact noise.

### Tier 4 — FrankaKitchen-v1  ✅ **0.90 full-4 success (3.88/4 sub-tasks), fully JEPA-based**
9-DoF arm, **compositional sequential sub-tasks** (microwave/kettle/light switch/slide cabinet — the standard
D4RL complete-v2 set). The hardest tier we've cleared, and it needed the full upgraded stack. Every *flat*
controller (BC/TD3+BC/IQL/CEM-MPC/latent-Dreamer) scored **0** — chaining, not single-step control, is the wall;
the JEPA *predictor* was even worse-than-no-op here (contact-rich → model exploitation). **The winning recipe:**
(1) **control-aware JEPA encoder** — add an *inverse-dynamics head* (`--inverse-dynamics`, predict aₜ from zₜ,zₜ₊₁)
so the latent keeps contact-relevant detail the plain VICReg/predictive encoder smooths away (lifts JEPA from
trailing-raw 1.37 → tying-raw 1.86); (2) **action-chunked flow-matching policy** on `[raw ⊕ JEPA latent]`
(`scripts/train_diffusion_policy.py --objective flow --concat-raw`) — chunking kills per-step compounding error,
flow/denoising models the multimodal demos instead of averaging to mush (flow ≈ diffusion at ~10× cheaper
sampling); (3) **subtask skill-hierarchy** — label demos by subtask via env-replay
(`scripts/label_kitchen_subtasks.py`), condition the flow skill on a target one-hot (`--subtask-cond`), drive it
with a trivial next-incomplete-subtask scheduler in `eval_diffusion_policy.py` → each subtask is a fresh
short-horizon problem (0→2.56/4, full-4 0.28); (4) **self-imitation / DAgger** — harvest the policy's own full-4
successes (`eval_diffusion_policy.py --collect-out`), augment the scarce 19 expert demos to 500+, retrain
(full-4 0.28→0.57→0.68); (5) **online self-imitation fine-tuning** (warm-start + collect-successes + fine-tune,
iterated; `--init-from`) → full-4 **0.68→0.81→0.87→0.90**. *Best policy*
`runs/franka_kitchen/checkpoints/kitchen_flow_skill_ft3.pt`; *videos* `kitchen_jepa_rl_tuned.mp4` (online-tuned 0.90),
`kitchen_jepa_skill_hierarchy.mp4` (offline 0.68). SLURM: `scripts/kitchen_hierarchy.slurm`.
**Behaviour (`scripts/analyze_kitchen_behavior.py`):** the bottleneck is the *3rd task, light switch* (0.33→0.88
across fine-tuning; microwave/kettle always ~1.0; fixed microwave→kettle→light-switch→slide-cabinet order).
**Negative on RL:** an *actual* advantage-weighted-regression objective (`scripts/finetune_skill_rl.py`) matched
self-imitation at peak (0.93) but then *collapsed* (on-policy instability) — the bottleneck is skill-reliability,
not exploration/credit-assignment, so filtered self-imitation (accumulating-success buffer) is the better, more
robust tool here.

### Tier 5 — Shadow Dexterous Hand in-hand manipulation  *(ultimate)*
`HandReach` (warm-up, 20-DoF reach) → `HandManipulateBlock` → `HandManipulateEgg` → `HandManipulatePen`,
with **full position+rotation** goals as the apex. **Hardest because** in-hand reorientation is the canonical
hard manipulation benchmark: 20-DoF, extremely contact-rich, near-chaotic dynamics, frequent drops, and a
**rotation goal on SO(3)** (quaternion geodesic distance, not Euclidean). Compounding world-model error over a
long contact-rich rollout is catastrophic here. **Block < Egg < Pen**; full-rotation Pen is the ceiling.
*Needs:* the whole upgraded stack — stochastic/ensemble world model, SO(3)-aware goal metric, learned data,
uncertainty-penalized long-horizon planning. This is the end-state target, not a near-term task.

**Summary order:** FetchSlide → PointMaze → AntMaze → Adroit(Door→Hammer→Pen→Relocate) → FrankaKitchen →
ShadowHand(Reach→Block→Egg→Pen).

---

# Roadmap A — Beefing up the world model

Motivated by PickAndPlace variance (40%↔100%) and the harder tiers above. In rough priority order:

1. **Grasp/contact auxiliary head.** Add a learned binary head (predicting "object grasped" / finger-object
   contact) to `ActionConditionedJEPA` alongside `state_probe`/`distance_probe`. The current `manip` score
   *heuristically* rewards closing fingers near the object; a learned grasp predictor gives MPC a crisp,
   differentiable grasp signal — directly attacks the pick bottleneck. Cheap, high ROI.
2. **Ensemble dynamics + disagreement penalty.** Replace the single transition/GRU with K=3–5 transition
   heads (shared encoder). Use mean for prediction and **inter-head disagreement** as an MPC cost term, so the
   planner avoids regions where the model is uncertain (a known fix for model-exploitation in MPC). Also
   doubles as an exploration signal for data collection.
3. **Stochastic latent (RSSM-lite).** Split the latent into deterministic + stochastic parts with a small
   posterior/prior head and a KL term (Dreamer/RSSM style). Contact dynamics (push/slide/in-hand) are
   genuinely multi-modal; a deterministic predictor blurs them. Enables the harder Adroit/ShadowHand tiers.
4. **Reduce compounding rollout error.** The recurrent predictor already trains on horizons `1,2,4,8,16`; add
   **scheduled sampling** (mix ground-truth and predicted latents during training) and an explicit open-loop
   rollout-consistency loss so 16-step predictions stop drifting. Consider a short Transformer dynamics core
   over a context window instead of a single GRU cell for the long-horizon tiers.
5. **Better representation regularization.** Add a **covariance (VICReg-style)** term to the existing variance
   regularizer to decorrelate latent dims and further guard against collapse.
6. **Better data.** (a) **HER** relabeling so the WM and policy see successful goal-reaching transitions;
   (b) increase the share / quality of grasp demos for pick; (c) capacity bump (`latent-dim` 128→192,
   `hidden-dim` 512) once the above are in.

Implementation touch-points: heads & ensemble in `models.py`; loss terms & flags in `train.py`; new MPC cost
(disagreement, learned-grasp) in `evaluate.py::JEPAMPCPolicy._manip_scores` / `_score_action_tensor`.

---

# Roadmap B — One controller + world model for reach + push + pick-and-place

Goal: a **single** JEPA world model and a **single** policy that solve all three Fetch tasks, instead of three
separate per-task checkpoints. Feasible because all three share the **4-D action space** and goal-conditioned
structure; the blocker is the **obs-width mismatch** (reach=16, push/pick=31) and skill disambiguation.

**Design:**
1. **Canonical state adapter** (in `envs.py`): map every Fetch env into one fixed-width vector =
   superset layout `[gripper pose+vel, fingers, object pose/rot/vel, achieved_goal, desired_goal]` +
   an **`object_present` flag** + a **task one-hot** (`reach`/`push`/`pick`). Reach zero-fills the object
   fields and sets `object_present=0`. This gives a single `state_dim` across tasks.
2. **Mixed multi-task data:** sample a task per episode in `collect_episodes`, run the matching scripted
   expert, tag with the task one-hot. Union dataset feeds both WM and BC.
3. **Skill conditioning:** the task one-hot + `object_present` flag (already in the state) let one encoder
   serve all skills; the policy reads it implicitly via `z`. Push vs pick is further disambiguated by goal
   height (table vs air), which the model can learn.
4. **Unified MPC score:** generalize `manip` scoring so reach/align/grasp terms are **gated by
   `object_present`** (reach collapses to pure gripper→goal distance; push/pick keep the grasp/align terms).
   `achieved_goal` already equals gripper (reach) or object (push/pick), so the terminal goal-distance term is
   already task-correct.
5. **Single new task `fetch_multi`** in `tasks.py` + a `scripts/train_eval_multi.sh` orchestrating the union
   train/eval. Per-task success is reported by filtering eval episodes by task one-hot.

**Why this should work:** the per-task models already share architecture and hyperparameters; the only genuine
differences are obs width (solved by the canonical adapter) and the expert used at collection time (solved by
per-episode task sampling). Expect a small per-task regression vs the specialists initially, recovered by the
Roadmap-A world-model upgrades (esp. the grasp head for pick).

---

# Roadmap C — Hierarchical JEPA for long-horizon tasks

The single-level world model rolls latents one primitive step at a time and drifts past ~16 steps. Tasks from
Tier 2 onward (mazes, kitchen, in-hand) run for hundreds of steps, where flat MPC is hopeless. **H-JEPA** stacks
two JEPA levels at different time scales:

- **Low level (fast)** — the *existing* `ActionConditionedJEPA` + `GoalConditionedPolicy`, conditioned on
  primitive 4-D actions, predicting per-step latents. Already built.
- **High level (slow)** — a second encoder/predictor over **subsampled** latents (every *k* steps), conditioned
  on **subgoals** rather than raw actions, predicting the abstract latent several macro-steps ahead. A 200-step
  task becomes ~25 high-level steps.

**Planning = two nested loops:** the outer loop plans a short sequence of subgoals over the slow model to
minimize distance-to-goal; the inner loop is the current `JEPAMPCPolicy` with the chosen subgoal swapped in as
`desired_goal`. Re-plan the high level every *k* steps or on subgoal achievement.

**Why it fits this codebase:**
1. The low level is *already* a goal reacher — `GoalConditionedPolicy(z)` reaches the goal encoded in `z`, so
   the high level only needs to emit subgoals; the existing policy+MPC executes them. No new low-level controller.
2. **Put subgoals in `achieved_goal` space** (interpretable, groundable) rather than abstract latent space, so
   `goal_reach_distance` and the `manip` terminal term work unchanged on subgoals.
3. It directly attacks compounding error: the horizon-16 head in `train.py` is already a crude jump predictor;
   H-JEPA formalizes that as its own level over far fewer, far more reliable steps.

**Build sketch:** add `encoder_hi`/`predictor_hi` on stride-*k* latents in `models.py`; train the high level
self-supervised with **HER** (the state actually reached *k* steps later *is* the subgoal achieved — free
labels); add a `HierarchicalMPCPolicy` in `evaluate.py` wrapping the existing planner.

**Caveats:** overkill for the 50–100-step Fetch triad (flat MPC already works) — it is the unlock for **Tier 2+**.
Watch for the high level proposing subgoals the low level cannot reach (mitigate with a reachability critic or by
restricting subgoals to the demonstrated manifold). Train the low level to convergence and freeze it before
training the high level.

---

# Realized roadmap progress

- **Roadmap A:** ✅ ensemble dynamics + disagreement (`--ensemble-heads`), ✅ VICReg covariance, ✅ better
  data (HER for slide/maze, **offline D4RL demos** for Adroit/AntMaze via `minari_to_npz.py`).
  Not pursued: grasp head (pick already solved), stochastic/RSSM latent (would be the Tier-5 prereq),
  scheduled sampling.
- **Roadmap C (Hierarchical JEPA):** ✅ demonstrated — `eval_hjepa_maze.py` beats flat on every maze
  (PointMaze 1.0/0.9/1.0, AntMaze UMaze 0.93). High level = data-driven **subgoal graph** (landmarks +
  empirical k-step reachability → routes around walls) + Dijkstra; low level = goal-conditioned HER/BC policy.
- **Roadmap B (unified Fetch controller):** ✅ done — ONE world model + ONE policy
  solve reach+push+pick at **1.00 / 0.97 / 1.00** (mean 0.99), matching the per-task
  specialists with no regression. Built exactly per the design: a **canonical state
  adapter** (`envs.py::CanonicalFetchWrapper`, `make_env(canonical_task=...)`) maps every
  Fetch env into one 35-D superset state `[25-D superset obs (object fields zeroed for
  reach) ⊕ object_present ⊕ task one-hot(3) ⊕ achieved ⊕ desired]`; a **union collector**
  (`data.py::collect_fetch_multi_episodes`, `scripts/collect_fetch_multi.py`) samples a
  sub-task per episode and saves the canonical ObsSpec into the npz; **`train.py` /
  `train_policy.py` read that spec via `load_spec_npz`** (`--episodes-npz`, backward-compatible);
  the **manip score gates grasp/reach/align by `object_present`** (`JEPAMPCPolicy.object_present_idx`,
  `CANONICAL_OBJECT_PRESENT_IDX`) so reach ignores the absent object and per-sub-task reach
  weight differs (push 0.0, pick 0.1). `scripts/eval_fetch_multi.py` reports per-task success;
  `scripts/train_eval_multi.sh` orchestrates collect→WM→policy→eval. Artifacts:
  `runs/fetch_multi/fetch_multi_{model,policy}.pt` + `fetch_multi_{reach,push,pick}_*.mp4`.

# Active work — Tier 5 (ShadowHand) is the remaining frontier

Tiers 1–4 are cleared (Kitchen Tier-4 solved at SOTA — see above). The Kitchen recipe (control-aware encoder
+ action-chunked flow policy + subtask skill-hierarchy + DAgger self-imitation) is the template for the
remaining contact-rich tiers. **Tier 5 (Shadow Dexterous Hand in-hand manipulation)** is the ultimate target:
20-DoF, near-chaotic contact, SO(3) rotation goals — needs the upgraded stack plus an SO(3)-aware goal metric
and likely a stochastic/RSSM latent.

**AntMaze (Medium/Large) — diagnosed + improved, but walker-capped (not SOTA).** Prior "0.25–0.30" was
optimistic; real Medium was ~0 (weak BC low-levels + a PointMaze-scale `reach_radius` bug in eval). Fixed
the reach scale, and the bottleneck is now pinned: the **navigation/hierarchy is fine** (the agent follows
11/12 subgoal waypoints around the maze) but the **offline-learned ant gait falls/stalls ~half the time**,
which caps chained success at ~0.27 (Medium). Tried 8 low-level recipes: latent BC/BC+HER (~0.06), latent
TD3+BC (0.25), **control-aware (inverse-dynamics) latent TD3+BC (0.27, best)**, latent IQL (0.10, critic
collapsed→AWR≈BC), raw-obs IQL (0.04–0.08, AWR over-peaked), raw-obs TD3+BC (0.00 — the JEPA latent's
features actually *help* the small MLP policy). New code: `scripts/train_gcrl_raw.py` (raw goal-conditioned
IQL), `--raw` mode in `train_offline_td3bc.py`, and **`scripts/eval_hjepa2.py` — a PROPER two-tier H-JEPA**
(learned high-level JEPA-2 feasibility model over the abstract latent + directed A* search/pruning, replacing
the empirical-reachability Dijkstra graph; matches the graph at 0.23, confirming the planner is not the
ceiling). Video: `runs/antmaze_medium/videos/antmaze_medium_hjepa.mp4`. UMaze (0.93, graph low-level) is the working
showcase. **Two further attempts (both negative, instructive):** (1) the *true* HWM planner —
`eval_hjepa2.py --planner cem` does CEM optimization over a CONTINUOUS sequence of subgoal offsets in latent
space (not discrete A*-over-landmarks), but scores 0.00: the latent-proxy feasibility can't reason about
walls at *future* positions along a multi-hop rollout (A*'s empirical edges encode that structure, which is
why discrete search is the tractable approximation). (2) **action-chunked locomotion** (`train_chunked_walker.py`,
`eval_hjepa_maze.py --low-type chunk`) — predict an 8-step action chunk for gait stability — also 0.00:
**BC averages the multimodal HER chunks to mush** (the exact kitchen failure mode; BC loss floors ~0.145).
The genuine SOTA path is therefore the full kitchen stack adapted to a goal-conditioned walker: action-chunked
**flow** (models multimodality instead of averaging) + HER + self-imitation/DAgger — a multi-hour build,
documented as the next effort. Net: AntMaze navigation/architecture are solved (proper two-tier H-JEPA +
both A* and CEM planners); the robust offline *walker* is the unbroken wall.

**Proper HWM high level (literature two-tier JEPA, `train_hjepa_hwm.py` +
`eval_hjepa_hwm.py`).** Strictly on top of the frozen low level: a high encoder
psi (192->16 abstract latent), a GRU macro-action encoder, a macro-step predictor
g(z_high, macro)->z_high at +N trained with the JEPA recipe (stop-grad target +
VICReg + normalized-MSE), and a decoder (abstract->achieved_goal position).
Planning is **CEM over continuous macro-actions through g** (one level up), not
Dijkstra. The three experiments (all confirmed): **(1) g generalizes** — held-out
macro-prediction err 0.0028, **100% landmark-pair coverage vs the empirical table's
12.2%** (the real generalization win); **(2) at K=1 (single macro-hop) HWM-CEM
matches Dijkstra (0.27=0.27)** — the learned continuous planner is competitive,
both walker-capped, so the *planner is solved, not the bottleneck*; **(3) compounding
error** (true GT chain RMSE 0.89/1.49/2.12/2.75 at depth 1/2/3/4) kills deep
rollouts — K=1/2/4 -> 0.27/0.17/0.07, the low-level predictor's failure mode one
level up. Bonus: `train_online_td3_her.py` (offline->online TD3+HER warm-started
from the 0.27 walker) attempts to break the walker ceiling.

# Infra note
GPU control node `watgpu208` has a broken SLURM GPU cgroup (`/dev/nvidia-uvm` PermissionError → `cuInit`
fails though `nvidia-smi` works); **route GPU work through `sbatch`**, not the interactive node. Offline
demos need `pip install --no-deps minari h5py portion` (keeps gymnasium 1.2.3 / numpy 1.26.4);
`MINARI_DATASETS_PATH=<repo>/.cache/minari`.

---
*Conventions: run from repo root inside conda `myenv` with `PYTHONNOUSERSITE=1 MUJOCO_GL=egl`; checkpoints/logs/eval land under `runs/<task>/`.*
