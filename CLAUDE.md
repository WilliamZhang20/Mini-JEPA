# CLAUDE.md — jepa-mini project guide

Action-conditioned **JEPA world model** + learned **goal-conditioned policy** + **model-predictive
control (MPC)** for Gymnasium-Robotics manipulation. The world model is a *predictor*, not a
controller: the BC policy proposes actions and the world-model MPC refines them.

## Repo map
- `jepa_robotics/models.py` — `ActionConditionedJEPA` (encoder + EMA target + `direct`/`rollout`/`recurrent`
  predictor + `state_probe` + `distance_probe`), `GoalConditionedPolicy`.
- `jepa_robotics/train.py` — world-model training (multi-horizon normalized-MSE JEPA loss + probes + regularizers).
- `jepa_robotics/train_policy.py` — behaviour cloning of the policy on the **frozen** JEPA latent.
- `jepa_robotics/evaluate.py` — `JEPAMPCPolicy` (random/CEM/grad planning; `latent`/`state`/`combined`/`manip` scoring),
  plus Random/Scripted/SB3/LearnedPolicyOnly baselines.
- `jepa_robotics/data.py` — episode collection, scripted experts (`reach`/`push`/`pick_place`), `Normalizer`.
- `jepa_robotics/envs.py` — `make_env`, `ObsSpec`, `flatten_obs` = concat[observation, achieved_goal, desired_goal].
- `jepa_robotics/tasks.py` — `TASKS` dict.
- `scripts/train_eval_object_v2.sh` — end-to-end: train WM → train policy → eval.

## Environment / run notes
- Conda env `myenv`; `MUJOCO_GL=egl`; H200 GPU (`--device cuda`). `gymnasium-robotics>=1.2`, action space is
  4-D `(dx,dy,dz,gripper)` for **all** Fetch tasks.
- Per-task obs widths differ: **FetchReach** `observation`=10 → `state_dim`=16; **Push/PickPlace/Slide**
  `observation`=25 → `state_dim`=31. This mismatch is the central obstacle for a unified model (see below).

## Status (success rate, mean final distance)
| Task | Best agent | Success | Dist |
|------|-----------|---------|------|
| FetchReach-v4 | JEPA+MPC (grad, state) | 95–100% | 0.02 |
| FetchPush-v4 | JEPA policy + CEM (policy60) | 100% | 0.014 |
| FetchPickAndPlace-v4 | JEPA policy + CEM (manip) | **40–100%** | 0.01–0.24 |

**PickAndPlace is the weak spot** (grasp is the bottleneck) and drives the world-model roadmap below.

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

### Tier 4 — FrankaKitchen-v1
9-DoF arm in a kitchen with **compositional, sequential sub-tasks** (open microwave, move kettle, flip switch,
slide cabinet…). **Harder because** it is *long-horizon and multi-task at once*: success = completing an
ordered set of sub-goals, demanding task decomposition and a skill scheduler, not a single goal-reach.
*Needs:* hierarchical control (a high-level sub-goal/skill selector over our flat goal-conditioned policy) and
a world model that stays accurate across very long rollouts.

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

# Active work — FetchSlide (Tier 1)

In progress: first task beyond the reach/push/pick triad. FetchSlide locks the gripper (`block_gripper=True`)
and places the goal **out of reach** via `target_offset=[0.4,0,0]` — a ballistic *strike* task with no
post-contact correction. Architecture beef-up for it: deeper recurrent dynamics (stacked residual transition
blocks, `--transition-depth`) + extended horizons to capture coasting + a scripted `slide` striking expert.
Task entry `fetch_slide` in `tasks.py`; video resolution is now configurable via `--width/--height` on
`record_jepa.py` (env default is 480×480).

---
*Conventions: run from repo root inside conda `myenv`; checkpoints/logs/eval land under `runs/<task>/`.*
