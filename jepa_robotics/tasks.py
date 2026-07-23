from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    """Static description of a Fetch task: env id, controller, episode length, and training horizons."""

    name: str
    env_id: str
    slug: str
    controller: str
    max_episode_steps: int
    horizons: str


TASKS = {
    "fetch_reach": TaskConfig(
        name="fetch_reach",
        env_id="FetchReach-v4",
        slug="fetch_reach",
        controller="reach",
        max_episode_steps=50,
        horizons="1,2,4,8",
    ),
    "fetch_pick_place": TaskConfig(
        name="fetch_pick_place",
        env_id="FetchPickAndPlace-v4",
        slug="fetch_pick_place",
        controller="pick_place",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    "fetch_push": TaskConfig(
        name="fetch_push",
        env_id="FetchPush-v4",
        slug="fetch_push",
        controller="push",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    # Roadmap B — ONE world model + ONE policy for reach + push + pick-and-place.
    # Not a single env: data is the canonical union (collect_fetch_multi.py) and
    # eval runs each sub-task through the canonical adapter (eval_fetch_multi.py).
    # env_id is a representative Fetch env (only used as a slug/placeholder);
    # controller="multi" signals the unified setup.
    "fetch_multi": TaskConfig(
        name="fetch_multi",
        env_id="FetchPickAndPlace-v4",
        slug="fetch_multi",
        controller="multi",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    "fetch_slide": TaskConfig(
        name="fetch_slide",
        env_id="FetchSlide-v4",
        slug="fetch_slide",
        controller="slide",
        # Striking task: the puck coasts out of reach, so train on longer
        # horizons to capture the post-contact ballistic phase.
        max_episode_steps=80,
        horizons="1,2,4,8,16,24",
    ),
    # Tier 3 — Adroit hand suite. Flat (non-goal) obs, dense reward, 24-30-D
    # action; no scriptable expert, so data comes from offline demonstrations or
    # exploratory rollouts for the world model. controller="none" -> random
    # collection. Difficulty order: Door < Hammer < Pen < Relocate.
    "adroit_door": TaskConfig(
        name="adroit_door",
        env_id="AdroitHandDoor-v1",
        slug="adroit_door",
        controller="none",
        max_episode_steps=200,
        horizons="1,2,4,8,16",
    ),
    "adroit_hammer": TaskConfig(
        name="adroit_hammer",
        env_id="AdroitHandHammer-v1",
        slug="adroit_hammer",
        controller="none",
        max_episode_steps=200,
        horizons="1,2,4,8,16",
    ),
    "adroit_pen": TaskConfig(
        name="adroit_pen",
        env_id="AdroitHandPen-v1",
        slug="adroit_pen",
        controller="none",
        max_episode_steps=200,
        horizons="1,2,4,8,16",
    ),
    "adroit_relocate": TaskConfig(
        name="adroit_relocate",
        env_id="AdroitHandRelocate-v1",
        slug="adroit_relocate",
        controller="none",
        max_episode_steps=200,
        horizons="1,2,4,8,16",
    ),
    # Shadow Dexterous Hand in-hand reorientation (HandManipulate suite): 20-D
    # actuator action, obs=61 (24-DoF hand qpos/qvel + object pose/vel),
    # achieved/desired goal = object pose (pos+quat, 7-D). Sparse reward,
    # contact-rich, multimodal — the target regime for DexterousJEPA. No scripted
    # expert (controller="none"); data is exploration collected with the trained
    # world model / random actions, control is goal-conditioned latent planning
    # toward the desired object pose.
    "handmanipulate_block": TaskConfig(
        name="handmanipulate_block",
        env_id="HandManipulateBlock-v1",
        slug="handmanipulate_block",
        controller="none",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    "handmanipulate_egg": TaskConfig(
        name="handmanipulate_egg",
        env_id="HandManipulateEgg-v1",
        slug="handmanipulate_egg",
        controller="none",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    # Stepping-stone for the hierarchical-JEPA + dexterous-flow SSL bet: single-axis
    # (Z) rotation with position IGNORED, so "direction" is an unambiguous scalar
    # (CW/CCW) and success only needs the rotation threshold. Used to test whether a
    # small reorientation is direction-controllable at all (the primitive the
    # abstract SO(3) planner would compose) before building the full stack.
    "handmanipulate_block_rotate_z": TaskConfig(
        name="handmanipulate_block_rotate_z",
        env_id="HandManipulateBlockRotateZ-v1",
        slug="handmanipulate_block_rotate_z",
        controller="none",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    "handmanipulate_pen": TaskConfig(
        name="handmanipulate_pen",
        env_id="HandManipulatePen-v1",
        slug="handmanipulate_pen",
        controller="none",
        max_episode_steps=100,
        horizons="1,2,4,8,16",
    ),
    # Tier 2 — PointMaze navigation: 2-D force action, obs = [x, y, vx, vy],
    # achieved/desired goal = (x, y). Sparse reward, long horizon, walls break
    # the straight-line heuristic, so the expert is a BFS-waypoint controller
    # (controller="maze"). Longer training horizons capture the multi-step
    # glide of the point mass between waypoints.
    "point_umaze": TaskConfig(
        name="point_umaze",
        env_id="PointMaze_UMaze-v3",
        slug="point_umaze",
        controller="maze",
        max_episode_steps=300,
        horizons="1,2,4,8,16,32",
    ),
    "point_medium": TaskConfig(
        name="point_medium",
        env_id="PointMaze_Medium-v3",
        slug="point_medium",
        controller="maze",
        max_episode_steps=600,
        horizons="1,2,4,8,16,32",
    ),
    "point_large": TaskConfig(
        name="point_large",
        env_id="PointMaze_Large-v3",
        slug="point_large",
        controller="maze",
        max_episode_steps=800,
        horizons="1,2,4,8,16,32",
    ),
    # AntMaze (Tier-2 tail): 8-DoF quadruped locomotion UNDER navigation. Env
    # version -v4 matches the Minari D4RL offline datasets (27-D obs). No scripted
    # expert (controller="none"); the low-level is BC on offline data. Used for
    # the canonical Hierarchical-JEPA demonstration.
    "antmaze_umaze": TaskConfig(
        name="antmaze_umaze",
        env_id="AntMaze_UMaze-v4",
        slug="antmaze_umaze",
        controller="none",
        max_episode_steps=700,
        horizons="1,2,4,8,16,32",
    ),
    # env_id MUST match the env the Minari dataset was recorded with (same maze
    # layout), else the BC low-level's maze != the eval maze. The *-diverse-v1
    # datasets recover to the "*_Diverse_GR-v4" envs.
    "antmaze_medium": TaskConfig(
        name="antmaze_medium",
        env_id="AntMaze_Medium_Diverse_GR-v4",
        slug="antmaze_medium",
        controller="none",
        max_episode_steps=1000,
        horizons="1,2,4,8,16,32",
    ),
    "antmaze_large": TaskConfig(
        name="antmaze_large",
        env_id="AntMaze_Large_Diverse_GR-v4",
        slug="antmaze_large",
        controller="none",
        max_episode_steps=1000,
        horizons="1,2,4,8,16,32",
    ),
    # Tier 4 — FrankaKitchen: 9-DoF arm, compositional sequential subtasks
    # (microwave/kettle/burner/switch). Flat 59-D obs (goal is a fixed task set,
    # not a coordinate); no scripted expert -> offline D4RL demos + BC on latent.
    "franka_kitchen": TaskConfig(
        name="franka_kitchen",
        env_id="FrankaKitchen-v1",
        slug="franka_kitchen",
        controller="none",
        max_episode_steps=280,
        horizons="1,2,4,8,16",
    ),
}


def task_from_env(env_id: str) -> TaskConfig:
    env_lower = env_id.lower()
    if "pickandplace" in env_lower:
        return TASKS["fetch_pick_place"]
    if "fetchpush" in env_lower:
        return TASKS["fetch_push"]
    if "fetchslide" in env_lower:
        return TASKS["fetch_slide"]
    if "fetchreach" in env_lower:
        return TASKS["fetch_reach"]
    if "adroit" in env_lower:
        if "hammer" in env_lower:
            return TASKS["adroit_hammer"]
        if "pen" in env_lower:
            return TASKS["adroit_pen"]
        if "relocate" in env_lower:
            return TASKS["adroit_relocate"]
        if "door" in env_lower:
            return TASKS["adroit_door"]
    if "pointmaze" in env_lower:
        if "large" in env_lower:
            return TASKS["point_large"]
        if "medium" in env_lower:
            return TASKS["point_medium"]
        return TASKS["point_umaze"]
    slug = (
        env_id.lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "_")
    )
    return TaskConfig(
        name=slug,
        env_id=env_id,
        slug=slug,
        controller="none",
        max_episode_steps=50,
        horizons="1,2,4,8",
    )


def resolve_task(task: str | None, env_id: str | None) -> TaskConfig:
    if task:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Available: {', '.join(sorted(TASKS))}")
        base = TASKS[task]
        if env_id is None:
            return base
        return TaskConfig(
            name=base.name,
            env_id=env_id,
            slug=base.slug,
            controller=base.controller,
            max_episode_steps=base.max_episode_steps,
            horizons=base.horizons,
        )
    if env_id is None:
        return TASKS["fetch_reach"]
    return task_from_env(env_id)


def task_dir(root: Path, task: TaskConfig) -> Path:
    return root / task.slug
