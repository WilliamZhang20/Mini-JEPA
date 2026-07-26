# AntMaze UMaze-diverse: official fixed-pair diagnosis

## Result boundary

Two evaluation protocols had been conflated:

| Protocol | Learned controller |
| --- | ---: |
| Random start/goal pairs used during development | 1.00/60 |
| Minari fixed pair, continuous HWM baseline | 0/100 |
| Minari fixed pair, discrete learned router | **72/100** |

The official fixed map places reset and goal on opposite sides of the center
wall. The learned flow-macro HWM selects an endpoint toward the goal and the ant
runs into that wall.

## Decisive control

A diagnostic breadth-first router over the environment's exposed free maze
cells produces:

```text
(0, 4) -> (-4, 4) -> (-4, 0) -> (-4, -4) -> (0, -4) -> goal
```

With the canonical unified flow walker unchanged, this route scores **8/10**.
The first checked rollout succeeds in roughly 323 steps; aggregate flip
fraction is 2.8%. Therefore:

```text
walker can execute UMaze
            +
correct U-shaped waypoints
            =
official fixed-pair success
```

This isolated the remaining failure to high-level learned topology. The 8/10
control reads the maze map, so it is an oracle rather than the learned-method
publication number.

## Negative architecture sweep

The following stayed at zero on the official pair:

- temporal-distance topology scorers;
- longer-horizon macro flow;
- direct and final-goal waypoint flows;
- demonstrated-waypoint and action-chunk retrieval;
- route-specialized and progress-conditioned flow walkers;
- deterministic action-chunk BC;
- a 25-trajectory route repertoire.

These continuous endpoint/action variants either point through the wall,
oscillate near it, stall, or lose gait stability. Detailed counts and logs are
in
[`runs/antmaze_umaze/experiments/architecture_attack_20260726/`](../runs/antmaze_umaze/experiments/architecture_attack_20260726/).

## Architecture replacement

A `DiscreteTopologyRouter` now replaces continuous endpoint selection for
official UMaze:

1. Enumerate the seven free UMaze regions during training.
2. Compute shortest cell-route labels for every current/goal region pair.
3. Train a classifier on jittered `(current_xy, goal_xy, delta_xy)` inputs to
   predict the next region.
4. Decode the selected class through the stored region-center codebook.
5. Execute that waypoint with the unchanged unified walker.

The frozen classifier reaches 100% supervised route accuracy and **72/100** over
the official evaluation. The 95% Wilson interval is 62.51–79.86%. A three-run
video check scores 3/3.

Inference does not query `env.unwrapped.maze`; it uses only current/goal
coordinates, learned weights, and the checkpoint's region-center codebook.
Training does use the official maze map to construct shortest-path labels. This
is map-distilled supervision and must be disclosed when comparing against
offline RL methods that learn topology solely from trajectories.

## Reproducibility and media

- Minari dataset: `D4RL/antmaze/umaze-diverse-v1`
- Evaluation: `eval_env=True`, 700 steps, terminate on first success
- MuJoCo: 3.1.6 (Minari declares `>=3.1.1,<=3.1.6`)
- Learned-method raw result:
  `runs/publication_benchmark/raw/antmaze_umaze_fixed_eval_100.log`
- Learned router checkpoint:
  `runs/antmaze_umaze/checkpoints/antmaze_umaze_discrete_topology_router.pt`
- Learned router training log:
  `runs/antmaze_umaze/logs/train_discrete_topology_router.log`
- Oracle raw result:
  `runs/antmaze_umaze/experiments/architecture_attack_20260726/map_oracle_fixed_eval_10.log`
- Whole-maze oracle video:
  `runs/antmaze_umaze/videos/antmaze_umaze_fixed_map_oracle_overview_multi.mp4`
- Whole-maze learned-router video:
  `runs/antmaze_umaze/videos/antmaze_umaze_discrete_topology_router_overview_multi.mp4`
- Whole-maze wall-collision diagnostic:
  `runs/antmaze_umaze/videos/antmaze_umaze_fixed_eval_latest_arch_diagnostic_multi.mp4`
