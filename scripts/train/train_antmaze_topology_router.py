"""Distill maze-cell shortest routes into a discrete learned waypoint router."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.algos.hwm import DiscreteTopologyRouter
from jepa_robotics.envs import make_env
from jepa_robotics.tasks import resolve_task


def shortest_path(maze_map, start, goal):
    queue = deque([start])
    parent = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        row, col = cell
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            rr, cc = neighbor
            if (
                0 <= rr < len(maze_map)
                and 0 <= cc < len(maze_map[0])
                and maze_map[rr][cc] != 1
                and neighbor not in parent
            ):
                parent[neighbor] = cell
                queue.append(neighbor)
    if goal not in parent:
        raise RuntimeError(f"No path from {start} to {goal}")
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    return list(reversed(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--minari-dataset",
        default=None,
        help="Use this dataset's evaluation maze specification for supervision.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-pair", type=int, default=2048)
    parser.add_argument("--position-jitter", type=float, default=0.8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=261200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    if args.minari_dataset is not None:
        import minari

        dataset = minari.load_dataset(args.minari_dataset, download=False)
        env = dataset.recover_environment(eval_env=True)
    else:
        env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    maze = env.unwrapped.maze
    cells = [
        (row, col)
        for row in range(maze.map_length)
        for col in range(maze.map_width)
        if maze.maze_map[row][col] != 1
    ]
    centers = np.asarray(
        [maze.cell_rowcol_to_xy(cell) for cell in cells], dtype=np.float32
    )
    env.close()
    cell_index = {cell: index for index, cell in enumerate(cells)}

    current, goals, targets = [], [], []
    for start in cells:
        for goal in cells:
            path = shortest_path(maze.maze_map, start, goal)
            target = cell_index[path[1] if len(path) > 1 else path[0]]
            for _ in range(args.samples_per_pair):
                current.append(
                    centers[cell_index[start]]
                    + rng.uniform(-args.position_jitter, args.position_jitter, 2)
                )
                goals.append(
                    centers[cell_index[goal]]
                    + rng.uniform(-args.position_jitter, args.position_jitter, 2)
                )
                targets.append(target)
    current_t = torch.from_numpy(np.asarray(current, dtype=np.float32)).to(device)
    goals_t = torch.from_numpy(np.asarray(goals, dtype=np.float32)).to(device)
    targets_t = torch.from_numpy(np.asarray(targets, dtype=np.int64)).to(device)
    router = DiscreteTopologyRouter(len(cells), args.hidden).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(targets_t), (args.batch_size,), device=device)
        loss = nn.functional.cross_entropy(
            router(current_t[index], goals_t[index]), targets_t[index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 1000 == 0:
            with torch.no_grad():
                accuracy = (
                    router(current_t, goals_t).argmax(dim=-1) == targets_t
                ).float().mean()
            print(
                json.dumps(
                    {
                        "event": "topology_router_train",
                        "step": step,
                        "loss": float(loss.detach()),
                        "train_accuracy": float(accuracy),
                    }
                ),
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": router.state_dict(),
            "centers": centers,
            "config": {
                "architecture": "discrete_topology_router_v1",
                "task": task.name,
                "region_count": len(cells),
                "hidden": args.hidden,
                "samples_per_pair": args.samples_per_pair,
                "position_jitter": args.position_jitter,
                "training_supervision": "maze_map_shortest_paths",
                "minari_dataset": args.minari_dataset,
                "inference_uses_maze_map": False,
            },
        },
        args.out,
    )
    print(json.dumps({"event": "topology_router_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
