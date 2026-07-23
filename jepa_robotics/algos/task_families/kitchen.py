"""Shared FrankaKitchen task metadata used by training and runtime control."""
from __future__ import annotations

import numpy as np

KITCHEN_TASKS = ["microwave", "kettle", "light switch", "slide cabinet"]
ALL_KITCHEN_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

# Flat observation slices: [robot_obs(18), obj_qpos(21), obj_qvel(20)].
KITCHEN_OBJ_DIMS = {
    "bottom burner": (20, 22),
    "top burner": (24, 26),
    "microwave": (31, 32),
    "kettle": (32, 39),
    "light switch": (26, 28),
    "slide cabinet": (28, 29),
    "hinge cabinet": (29, 31),
}

KITCHEN_GOALS = {
    "bottom burner": np.array([-0.88, -0.01], np.float32),
    "top burner": np.array([-0.92, -0.01], np.float32),
    "microwave": np.array([-0.75], np.float32),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06], np.float32),
    "light switch": np.array([-0.69, -0.05], np.float32),
    "slide cabinet": np.array([0.37], np.float32),
    "hinge cabinet": np.array([0.0, 1.45], np.float32),
}


def parse_kitchen_tasks(value: str | None, *, default=KITCHEN_TASKS) -> list[str]:
    """Parse a comma-separated task set, accepting ``all`` for all seven."""
    if value is None:
        return list(default)
    tasks = list(ALL_KITCHEN_TASKS) if value.strip().lower() == "all" else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    unknown = [task for task in tasks if task not in ALL_KITCHEN_TASKS]
    if unknown:
        raise ValueError(f"Unknown Kitchen tasks: {unknown}; choices={ALL_KITCHEN_TASKS}")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Kitchen task list must be non-empty and contain no duplicates")
    return tasks


class DemonstrationTaskGraph:
    """Order tasks using demonstrated handoff compatibility.

    Each specialist supplies segment starts and terminal states. Edge ``a -> b``
    is the typical normalized arm-pose distance from an ``a`` terminal to the
    closest demonstrated ``b`` start. A small dynamic program selects the
    minimum-cost route through whatever task subset is requested at runtime.
    """

    def __init__(
        self,
        task_segments: dict[str, list[np.ndarray]],
        normalizer,
        *,
        match_dims: tuple[int, int] = (0, 9),
        robust_quantile: float = 0.25,
    ) -> None:
        self.tasks = list(task_segments)
        lo, hi = match_dims
        self.match_dims = (int(lo), int(hi))
        self.quantile = float(robust_quantile)
        self.starts: dict[str, np.ndarray] = {}
        self.ends: dict[str, np.ndarray] = {}
        for task, segments in task_segments.items():
            clean = [np.asarray(segment, np.float32) for segment in segments if len(segment)]
            if not clean:
                raise ValueError(f"Task {task!r} has no demonstration segments")
            self.starts[task] = normalizer.encode(np.stack([segment[0] for segment in clean]))[:, lo:hi]
            self.ends[task] = normalizer.encode(np.stack([segment[-1] for segment in clean]))[:, lo:hi]
        self.transition = {
            (source, target): self._bank_cost(self.ends[source], self.starts[target])
            for source in self.tasks
            for target in self.tasks
            if source != target
        }
        self.normalizer = normalizer

    def _bank_cost(self, sources: np.ndarray, targets: np.ndarray) -> float:
        distance = np.linalg.norm(sources[:, None, :] - targets[None, :, :], axis=-1)
        nearest = distance.min(axis=1)
        return float(np.quantile(nearest, self.quantile))

    def start_cost(self, state: np.ndarray, task: str) -> float:
        lo, hi = self.match_dims
        live = self.normalizer.encode(np.asarray(state, np.float32))[lo:hi]
        return float(np.linalg.norm(self.starts[task] - live[None], axis=1).min())

    def best_route(self, state: np.ndarray, remaining: list[str]) -> list[str]:
        """Return the minimum demonstrated-handoff-cost route through remaining tasks."""
        remaining = list(dict.fromkeys(remaining))
        if len(remaining) <= 1:
            return remaining
        n = len(remaining)
        initial = [self.start_cost(state, task) for task in remaining]
        # (visited mask, final index) -> (cost, path indices)
        dp = {(1 << i, i): (initial[i], [i]) for i in range(n)}
        for size in range(1, n):
            for (mask, last), (cost, path) in list(dp.items()):
                if mask.bit_count() != size:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    new_cost = cost + self.transition[(remaining[last], remaining[nxt])]
                    key = (new_mask, nxt)
                    if key not in dp or new_cost < dp[key][0]:
                        dp[key] = (new_cost, path + [nxt])
        full = (1 << n) - 1
        _, best_path = min((value for (mask, _last), value in dp.items() if mask == full),
                           key=lambda item: item[0])
        return [remaining[index] for index in best_path]


class KitchenScheduler:
    """Choose the next unfinished Kitchen task, with optional stall rotation."""

    def __init__(self, tasks=KITCHEN_TASKS, timeout: int = 0) -> None:
        self.tasks = list(tasks)
        self.timeout = int(timeout)
        self.cur = 0
        self.t_on = 0

    def update(self, done_tasks) -> int:
        done_idx = {i for i, task in enumerate(self.tasks) if task in done_tasks}
        incomplete = [i for i in range(len(self.tasks)) if i not in done_idx]
        if not incomplete:
            return self.cur
        if self.cur not in incomplete:
            self.cur, self.t_on = incomplete[0], 0
        elif self.timeout > 0 and self.t_on >= self.timeout:
            order = [i for i in incomplete if i > self.cur] + [i for i in incomplete if i < self.cur]
            self.cur, self.t_on = order[0], 0
        else:
            self.t_on += 1
        return self.cur
