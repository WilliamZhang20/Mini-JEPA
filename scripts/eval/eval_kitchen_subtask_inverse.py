"""Evaluate the FrankaKitchen subtask-specialist SSL controller.

Sequential ports of the Relocate laws:

  1. No blurring: one segment-pure inverse specialist per subtask
     (train_kitchen_subtask_inverse.py).
  2. Future coherence: each specialist tracks a demo-locked future index built
     from that subtask's demo segments.
  3. Input-feature emphasis: each specialist duplicates its object qpos dims in
     the conditioning (applied automatically from checkpoint metadata).
  4. Firm predicate switch: advance only when a completion predicate fires.
     This can be either the legacy environment event or a JEPA-latent probe
     learned from demonstration progression (no runtime completion oracle).

The high level can follow an explicit order or infer a minimum-cost handoff
route from demonstrated specialist segment boundaries. Both the requested task
set and number of specialists are dynamic, including all seven Kitchen tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.control.completion import LatentCompletionProbe
from jepa_robotics.algos.futures import DemoLockedFutureIndex
from jepa_robotics.algos.task_families.kitchen import (
    DemonstrationTaskGraph,
    KITCHEN_GOALS,
    KITCHEN_OBJ_DIMS,
    parse_kitchen_tasks,
)
from jepa_robotics.algos.priors import InversePrior, append_emphasis


class LearnedCompletionDetector:
    """Runtime wrapper for a frozen demonstration-trained completion probe."""

    def __init__(self, path, wm, norm, dev, required_tasks, threshold_override=None):
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        self.task_names = [str(name) for name in ckpt.get("task_names", [])]
        missing = [task for task in required_tasks if task not in self.task_names]
        if missing:
            raise ValueError(
                f"Completion checkpoint lacks requested tasks {missing}; contains {self.task_names}"
            )
        self.task_index = {task: index for index, task in enumerate(self.task_names)}
        self.probe = LatentCompletionProbe(
            int(ckpt.get("input_dim", ckpt["latent_dim"])),
            int(ckpt["num_tasks"]), int(ckpt["hidden"])
        ).to(dev)
        self.probe.load_state_dict(ckpt["state_dict"])
        self.probe.eval()
        self.wm = wm
        self.norm = norm
        self.dev = dev
        self.concat_raw = bool(ckpt.get("concat_raw", False))
        self.thresholds = np.asarray(ckpt["thresholds"], dtype=np.float32)
        if threshold_override is not None:
            self.thresholds[:] = float(threshold_override)

    @torch.no_grad()
    def probabilities(self, raw) -> dict[str, float]:
        state = torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev)
        features = self.wm.encode(state)
        if self.concat_raw:
            features = torch.cat([features, state], dim=-1)
        values = torch.sigmoid(self.probe(features))[0].cpu().numpy()
        return {task: float(values[index]) for task, index in self.task_index.items()}

    def threshold(self, task: str) -> float:
        return float(self.thresholds[self.task_index[task]])


class Specialist:
    def __init__(self, path, wm, norm, dev, target_horizon, geom_weight,
                 match_dims=(0, 9), relock_margin=0.0):
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        self.ckpt = ckpt
        self.prior = InversePrior(int(ckpt["cond_dim"]), int(ckpt["chunk_dim"]),
                                  int(ckpt["hidden"]), int(ckpt["n_blocks"])).to(dev)
        self.prior.load_state_dict(ckpt["state_dict"])
        self.prior.eval()
        self.wm = wm
        self.norm = norm
        self.dev = dev
        self.name = ckpt.get("subtask_name", "?")
        horizons = list(ckpt.get("future_horizons", [int(ckpt["H"])]))
        self.target_h = int(target_horizon or max(horizons))
        self.horizons_max = max(horizons)
        segs = [np.asarray(s, dtype=np.float32) for s in ckpt["segments"]]
        self.segments = segs
        # Demo matching keys on the ROBOT ARM dims (0:9), not the object dims:
        # a subtask's object qpos is near-constant until manipulated, so weighting
        # it corrupts the lock, whereas the arm pose determines reachability of the
        # object from the current handoff state (relocate law 2, future coherence).
        self.future_index = DemoLockedFutureIndex(
            segs, norm, horizon=self.target_h,
            geom_dims=match_dims, geom_weight=geom_weight,
            relock_margin=relock_margin,
        )

    def reset(self):
        self.future_index.reset()

    def match_distance(self, raw):
        """How reachable this subtask is from the live state (min weighted
        distance to any of its demo segments); lower = more reachable."""
        return self.future_index.match_distance(raw, self.norm)

    @torch.no_grad()
    def plan(self, raw, env, action_scale):
        target_state = self.future_index.query(raw, self.norm)
        s = torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev)
        tgt = torch.from_numpy(self.norm.encode(target_state)).unsqueeze(0).to(self.dev)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(tgt)
        h_token = torch.tensor([[float(self.target_h) / float(self.horizons_max)]], dtype=z.dtype, device=self.dev)
        parts = [z, z_goal, h_token]
        if bool(self.ckpt.get("concat_raw", False)):
            parts.extend([s, tgt])
        append_emphasis(parts, s, self.ckpt)
        cond = torch.cat(parts, dim=-1)
        chunk = self.prior(cond).view(int(self.ckpt["H"]), int(self.ckpt["action_dim"])) * action_scale
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.dev)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.dev)
        return chunk.clamp(low, high).cpu().numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--specialist-paths", type=Path, nargs="+", required=True,
                   help="One checkpoint per requested task; checkpoint names determine task identity.")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated requested task set, or 'all'. Defaults to all supplied specialists.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=40000)
    p.add_argument("--exec-k", type=int, default=2)
    p.add_argument("--target-horizon", type=int, default=8)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--geom-weight", type=float, default=3.0,
                   help="Upweight the --match-dims slice in demo-locked matching so re-localization tracks the reachability-determining features.")
    p.add_argument("--match-dims", default="0,9",
                   help="Demo-matching slice (lo,hi) upweighted by --geom-weight. Default 0,9 = robot arm joints (approach reachability); the object qpos is near-constant until manipulated so weighting it corrupts the lock.")
    p.add_argument("--relock-margin", type=float, default=0.0,
                   help="Re-lock to a better-matching demo when its weighted match beats the locked demo by this margin (recovery from a bad handoff lock).")
    p.add_argument("--subtask-patience", type=int, default=0,
                   help="If the current task stalls, defer it and re-route (0 disables).")
    p.add_argument("--order", default=None,
                   help="[fixed scheduler] Comma-separated task names or supplied-specialist indices.")
    p.add_argument("--scheduler", default="graph", choices=["fixed", "greedy", "graph"],
                   help="'graph': plan the lowest-cost task route from demonstrated handoffs; "
                        "'greedy': nearest task start only; 'fixed': follow --order.")
    p.add_argument("--completion-mode", default="env", choices=["env", "learned"],
                   help="Specialist switch source. 'learned' uses only a frozen JEPA-latent probe trained from demo progression; env events remain scoring-only.")
    p.add_argument("--completion-path", type=Path, default=None,
                   help="Checkpoint from train_kitchen_completion_probe.py (required for --completion-mode learned).")
    p.add_argument("--completion-threshold", type=float, default=None,
                   help="Override per-task thresholds calibrated into the learned completion checkpoint.")
    p.add_argument("--completion-debounce", type=int, default=2,
                   help="Consecutive learned-positive observations required before switching.")
    p.add_argument("--max-episode-steps", type=int, default=280)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--log-episodes", action="store_true")
    p.add_argument("--video-out", type=Path, default=None,
                   help="Save an mp4 of the first episode completing every requested task.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    match_dims = tuple(int(x) for x in args.match_dims.split(","))
    supplied = [Specialist(pth, wm, norm, dev, args.target_horizon, args.geom_weight,
                           match_dims=match_dims, relock_margin=args.relock_margin)
                for pth in args.specialist_paths]
    specialist_by_name = {specialist.name: specialist for specialist in supplied}
    if len(specialist_by_name) != len(supplied):
        raise ValueError("Specialist checkpoints must have unique subtask_name values")
    tasks = parse_kitchen_tasks(args.tasks, default=list(specialist_by_name))
    missing_specialists = [name for name in tasks if name not in specialist_by_name]
    if missing_specialists:
        raise ValueError(f"Missing specialist checkpoints for requested tasks {missing_specialists}")
    specialists = {name: specialist_by_name[name] for name in tasks}
    num_tasks = len(tasks)
    if args.completion_mode == "learned" and args.completion_path is None:
        p.error("--completion-mode learned requires --completion-path")
    completion = (
        LearnedCompletionDetector(
            args.completion_path, wm, norm, dev, tasks, args.completion_threshold
        )
        if args.completion_mode == "learned" else None
    )
    task_graph = DemonstrationTaskGraph(
        {name: specialists[name].segments for name in tasks},
        norm,
        match_dims=match_dims,
    )

    render = args.video_out is not None
    env = make_env(
        task.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        render_mode="rgb_array" if render else None,
        width=args.width if render else None,
        height=args.height if render else None,
        kitchen_tasks=tasks,
    )
    if args.order is None:
        order = list(tasks)
    else:
        order = []
        for token in (part.strip() for part in args.order.split(",") if part.strip()):
            if token.lstrip("-").isdigit():
                order.append(tasks[int(token)])
            else:
                order.append(token)
        if set(order) != set(tasks) or len(order) != len(tasks):
            raise ValueError(f"--order must contain each requested task exactly once: {tasks}")
    video_saved = False

    successes, ntasks = [], []
    initial_route_counts: dict[str, int] = {}
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for sp in specialists.values():
            sp.reset()
        # ``done_tasks`` is controller state.  In learned mode it is updated
        # exclusively by the latent probe. ``actual_done_tasks`` is scoring and
        # diagnostics only and is never read by the scheduler.
        done_tasks: set = set()
        actual_done_tasks: set = set()
        completion_streak = {name: 0 for name in tasks}
        false_switches = 0
        cached: list[np.ndarray] = []
        term = trunc = False
        info = {}
        switches = []
        traj = [flatten_obs(obs).copy()]
        capture = render and not video_saved
        frames = []
        if capture:
            fr = env.render()
            if fr is not None:
                frames.append(fr)

        def next_uncompleted(after):
            """Next task in the configured attempt order that is not yet done."""
            pos = order.index(after)
            for step in range(1, num_tasks + 1):
                candidate = order[(pos + step) % num_tasks]
                if candidate not in done_tasks:
                    return candidate
            return after

        def pick_next(after, raw):
            """Choose the next task using fixed, local-greedy, or graph routing."""
            candidates = [name for name in tasks if name not in done_tasks and name != after]
            if not candidates:
                candidates = [name for name in tasks if name not in done_tasks]
            if not candidates:
                return after
            if args.scheduler == "greedy":
                return min(candidates, key=lambda name: specialists[name].match_distance(raw))
            if args.scheduler == "graph":
                route = task_graph.best_route(raw, candidates)
                if not route:
                    return after
                return route[0]
            return next_uncompleted(after)

        # Firm scheduler with patience rotation. Graph routing plans over all
        # requested tasks using demonstrated handoff costs; no fixed count or
        # canonical order is assumed.
        if args.scheduler == "greedy":
            cur = min(tasks, key=lambda name: specialists[name].match_distance(flatten_obs(obs)))
            specialists[cur].reset()
            initial_route = [cur]
        elif args.scheduler == "graph":
            initial_route = task_graph.best_route(flatten_obs(obs), tasks)
            cur = initial_route[0]
            specialists[cur].reset()
        else:
            cur = order[0]
            initial_route = list(order)
        route_key = " -> ".join(initial_route)
        initial_route_counts[route_key] = initial_route_counts.get(route_key, 0) + 1
        steps_on_cur = 0
        while not (term or trunc):
            if cur in done_tasks:
                cur = pick_next(cur, flatten_obs(obs))
                steps_on_cur = 0
                specialists[cur].reset()
                cached = []
            if not cached:
                raw = flatten_obs(obs)
                plan = specialists[cur].plan(raw, env, args.action_scale)
                k = max(1, min(args.exec_k, len(plan)))
                cached = [plan[i].copy() for i in range(k)]
            action = np.clip(cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
            traj.append(flatten_obs(obs).copy())
            if capture:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
            steps_on_cur += 1
            actual_newly = set(info.get("step_task_completions", []))
            actual_done_tasks |= actual_newly
            completion_prob = None
            if completion is None:
                newly = actual_newly
            else:
                completion_prob = completion.probabilities(flatten_obs(obs))
                for name in tasks:
                    completion_streak[name] = (
                        completion_streak[name] + 1
                        if completion_prob[name] >= completion.threshold(name) else 0
                    )
                # Check only the executing specialist. This preserves the
                # controller's causal order and prevents an unrelated high
                # probe score from skipping a task.
                newly = set()
                if completion_streak[cur] >= max(1, args.completion_debounce):
                    newly.add(cur)
            new_here = newly - done_tasks
            if new_here:
                if completion is not None and not new_here.issubset(actual_done_tasks):
                    false_switches += 1
                done_tasks |= newly
                switches.append((int(info.get("tasks_done", len(done_tasks))), sorted(new_here)))
                cached = []  # firm switch: drop the rest of the current chunk
                if cur in done_tasks and len(done_tasks) < num_tasks:
                    cur = pick_next(cur, flatten_obs(obs))
                    steps_on_cur = 0
                    specialists[cur].reset()
            elif (
                args.subtask_patience > 0
                and steps_on_cur >= args.subtask_patience
                and len(done_tasks) < num_tasks - 1
            ):
                # Stalled: defer this subtask, rotate to another uncompleted one.
                nxt = pick_next(cur, flatten_obs(obs))
                if nxt != cur:
                    switches.append((int(info.get("tasks_done", len(done_tasks))), [f"defer:{cur}->{nxt}"]))
                    cur = nxt
                    specialists[cur].reset()
                    cached = []
                steps_on_cur = 0
        success = float(info.get("is_success", 0.0))
        nt = int(info.get("tasks_done", len(done_tasks)))
        successes.append(success)
        ntasks.append(nt)
        if capture and success > 0.5 and frames:
            import imageio.v2 as imageio
            args.video_out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.video_out, frames, fps=args.fps, format="FFMPEG")
            video_saved = True
            print(json.dumps({"event": "video_saved", "path": str(args.video_out), "episode": ep}), flush=True)
        if args.log_episodes:
            tj = np.asarray(traj, dtype=np.float32)
            min_dist = {}
            for name in tasks:
                lo, hi = KITCHEN_OBJ_DIMS[name]
                d = np.linalg.norm(tj[:, lo:hi] - KITCHEN_GOALS[name][None], axis=-1)
                min_dist[name] = round(float(d.min()), 3)
            print(json.dumps({"event": "kitchen_episode", "episode": ep, "success": success,
                              "tasks_done": nt, "completed": sorted(done_tasks),
                              "actual_completed": sorted(actual_done_tasks),
                              "false_switches": false_switches,
                              "min_goal_dist": min_dist, "switches": switches}), flush=True)

    env.close()
    row = {
        "event": "kitchen_subtask_eval",
        "task": task.name,
        "requested_tasks": tasks,
        "scheduler": args.scheduler,
        "order": order,
        "initial_route_counts": initial_route_counts,
        "specialist_paths": [str(x) for x in args.specialist_paths],
        "episodes": int(args.episodes),
        "success_rate": float(np.mean(successes)),
        "mean_tasks": float(np.mean(ntasks)),
        "tasks_hist": {
            str(i): int(np.sum(np.asarray(ntasks) == i)) for i in range(num_tasks + 1)
        },
        "exec_k": int(args.exec_k),
        "target_horizon": int(args.target_horizon),
        "geom_weight": float(args.geom_weight),
        "action_scale": float(args.action_scale),
        "torch_seed": args.torch_seed,
        "seed": args.seed,
        "completion_mode": args.completion_mode,
        "completion_path": None if args.completion_path is None else str(args.completion_path),
        "completion_threshold": args.completion_threshold,
        "completion_debounce": int(args.completion_debounce),
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
