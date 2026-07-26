# Official UMaze-diverse architecture attack — 2026-07-26

Target: Minari `D4RL/antmaze/umaze-diverse-v1`, `eval_env=True`, 700 steps,
terminate on first success. This is the fixed `r`/`g` map, not the earlier
random-pair environment.

## Outcome

The initial continuous/retrieval sweep did not solve the official pair. A final
discrete topology replacement succeeds at **72/100** and is promoted as the
official UMaze high level.

The elevated render showed that the fixed reset and goal are separated by the
center wall. The original HWM decoded wall-crossing endpoints, while subsequent
low-level variants either stalled at the wall or lost gait stability.

| Variant | Official result | Diagnostic |
| --- | ---: | --- |
| Original HWM + unified flow walker | 0/100 | Infeasible wall-crossing subgoals. |
| Topology scorer + original/full-horizon macro flow | 0/10, 0/20 | No completed turn. |
| Direct waypoint flow, including final-goal conditioning | 0/3 | Invalid or short targets. |
| Waypoint retrieval + unified walker | 0/3 and 0/10 | Walker flips/stalls. |
| Route-specialized continued flow | 0/10 | More upright, but slow. |
| Progress-conditioned route flow, corrected auxiliary labels | 0/10 | 93.8% stalled. |
| Route-only progress flow | 0/10 | 94.1% stalled. |
| Deterministic chunk BC | 0/10 and 0/20 | Not rollout-stable. |
| Final-goal successful-prefix flow | 0/20 | 95.8% stalled. |
| 25-route trajectory ensemble | 0/125 trials | No sequence transferred. |

## Decisive map-router control

After inspecting the elevated failure video, a diagnostic shortest-path router
over the environment's exposed maze cells was added. It emits the actual
U-shaped sequence:

`(0,4) -> (-4,4) -> (-4,0) -> (-4,-4) -> (0,-4) -> goal`.

With the **unchanged unified flow walker**, this oracle route scored **8/10**
on the official fixed pair. The first rollout reached the goal in roughly 323
steps; aggregate flip fraction was 2.8%. This proves UMaze locomotion is not the
problem: the learned HWM high level is selecting wall-directed endpoints.

This 8/10 diagnostic motivated a discrete learned replacement.

## Learned discrete topology replacement

`DiscreteTopologyRouter` is trained to classify the next free region from
`(current_xy, goal_xy, delta_xy)`. Supervision consists of shortest cell routes
computed from the official evaluation maze, with coordinate jitter. At
inference it uses only its frozen weights and stored seven-region codebook; it
does not query the live maze map.

- Supervised route accuracy: 100%
- Official smoke test: 15/20
- Official publication evaluation: **72/100**
- 95% Wilson interval: **62.51–79.86%**
- Whole-maze video check: 3/3

The use of map-derived route labels at training time is privileged structural
supervision and must be disclosed in every literature comparison.

## Simulator finding

Minari declares MuJoCo `>=3.1.1,<=3.1.6`; the shared environment had 3.9.0.
A project-local MuJoCo 3.1.6 install was used for final checks via:

```bash
PYTHONPATH="$PWD/.cache/mujoco316:$PWD" ...
```

An exact successful trajectory failed under 3.9.0 but reproduced in 67 steps
from its recorded state under 3.1.6. Matching dynamics did not make learned
policies or route-sequence transfer succeed from official resets.

## Artifacts

- Training/evaluation logs in this directory
- Diagnostic video:
  `../../videos/antmaze_umaze_fixed_eval_latest_arch_diagnostic_multi.mp4`
- Correct-route oracle video:
  `../../videos/antmaze_umaze_fixed_map_oracle_overview_multi.mp4`
- Learned-router video:
  `../../videos/antmaze_umaze_discrete_topology_router_overview_multi.mp4`

All videos use an elevated camera framing the complete maze. The learned-router
video contains three successes in three episodes.
