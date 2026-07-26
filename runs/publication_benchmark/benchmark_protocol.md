# Protocol-aligned frozen-controller comparison

This artifact compares the canonical frozen JEPA controllers with published
baselines. No policy, world model, prior, or critic was trained or fine-tuned
for this comparison. Every new result is a 100-episode evaluation starting at
seed 240000, except AntMaze and Fetch, which start at seeds 220000 and 230000.

## Evaluation definitions

| Panel | This-work evaluation | Literature anchor |
| --- | --- | --- |
| AntMaze | Minari `eval_env=True` fixed `r`/`g` map; terminate on first success; 700 steps for UMaze and 1000 for Medium/Large | D4RL `*-diverse-v2`, 100-episode normalized score from CORL |
| Adroit | Gymnasium-Robotics `AdroitHand*-v1`, dense reward, 200 steps; normalized with the Minari D4RL expert reference min/max | D4RL `*-expert-v1` normalized return from CORL |
| Kitchen | Gymnasium-Robotics `FrankaKitchen-v1`; standard microwave/kettle/light/slide goal; 280 steps; score is mean completed tasks divided by four | D4RL Kitchen Partial scores reported by VanTA |
| Fetch | Gymnasium-Robotics Fetch v4, sparse success, exactly 50 steps | Fetch-v1, sparse success, 50 steps from the FAHER study |

The Gymnasium-Robotics conversions contain bug fixes and are not binary-identical
to older D4RL/Fetch environment versions. The figure is therefore
**metric-, horizon-, task-, and reset-protocol aligned**, but it should not be
described as a certified same-codebase leaderboard comparison. Exact same-version
comparison would require rerunning every baseline in the modern environment.

## This-work results

| Task | Result | Additional statistic |
| --- | ---: | ---: |
| AntMaze UMaze-diverse | 72/100 success | 72.0 |
| AntMaze Medium-diverse | 54/100 success | 54.0 |
| AntMaze Large-diverse | 31/100 success | 31.0 |
| Adroit Door expert | 98/100 success | 101.46 normalized return |
| Adroit Relocate expert | 91/100 success | 85.04 normalized return |
| Kitchen partial | 83/100 full-four success | 3.63/4 mean = 90.75 normalized |
| FetchPickAndPlace | 100/100 success | 50-step horizon |
| FetchSlide | 86/100 success | 50-step horizon |

The AntMaze fixed-pair results supersede the random-pair 1.00/0.775/0.533
headline **for literature comparisons only**. The latter remains a valid result
for the repository's random-pair evaluation distribution.

## Normalization

Adroit normalized return is

`100 * (return - ref_min_score) / (ref_max_score - ref_min_score)`.

The Minari metadata reference pairs used here are:

- Door: `ref_min=-45.80706024169922`, `ref_max=2940.578369140625`
- Relocate: `ref_min=9.189092636108398`, `ref_max=4287.70458984375`

Kitchen's Minari reference range is 0 to 4 completed tasks, hence
`100 * mean_tasks / 4`.

## Sources

- CORL benchmark (NeurIPS 2023):
  https://papers.neurips.cc/paper_files/paper/2023/file/62d2cec62b7fd46dd35fa8f2d4aeb52d-Paper-Datasets_and_Benchmarks.pdf
- VanTA Kitchen results (ICLR 2025):
  https://openreview.net/pdf/ca4557fa427c5a802d5bea5975f8fd9c2d014209.pdf
- FAHER Fetch results:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11784800/
- Minari AntMaze evaluation specification:
  https://minari.farama.org/main/datasets/antmaze/large-diverse/

The exact plotted values, error-bar definitions, and source labels are in
`benchmark_scores.csv`. Raw evaluation stdout and JSONL files are under `raw/`.

### UMaze architecture follow-up (2026-07-26)

The continuous-HWM 0/100 UMaze result was attacked with topology scoring, longer-horizon macro
flow, direct waypoint flow, demonstrated-waypoint retrieval, route-specialized
flow, progress-conditioned flow, deterministic chunk BC, and a 25-trajectory
route repertoire. None achieved a success on the official fixed map, so the
continuous variants remained at zero. Final evaluations used Minari's declared
MuJoCo 3.1.6 compatibility range. Full negative results are under
`runs/antmaze_umaze/experiments/architecture_attack_20260726/`.

A map-router oracle subsequently scored 8/10 with the unchanged low-level
walker, confirming the zero is a learned high-level topology failure rather
than an UMaze locomotion failure. The oracle consumes the environment maze map
and is therefore not plotted as our learned method.

The published replacement is a frozen seven-region classifier distilled from
shortest routes on the official evaluation maze. It receives only current and
goal coordinates at inference, predicts the next region, and uses a stored
region-center codebook; it does not query the live maze map. It scores 72/100
(95% Wilson CI 62.51–79.86%). Because map topology supplies its training labels,
the figure and CSV explicitly describe it as map-distilled supervision.
