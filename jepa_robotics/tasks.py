from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
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
    "adroit_door": TaskConfig(
        name="adroit_door",
        env_id="AdroitHandDoor-v1",
        slug="adroit_door",
        controller="none",
        max_episode_steps=200,
        horizons="1,2,4,8,16",
    ),
}


def task_from_env(env_id: str) -> TaskConfig:
    env_lower = env_id.lower()
    if "pickandplace" in env_lower:
        return TASKS["fetch_pick_place"]
    if "fetchreach" in env_lower:
        return TASKS["fetch_reach"]
    if "adroit" in env_lower and "door" in env_lower:
        return TASKS["adroit_door"]
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
