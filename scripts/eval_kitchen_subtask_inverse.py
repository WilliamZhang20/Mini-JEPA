"""Evaluate the FrankaKitchen subtask-specialist SSL controller.

Sequential ports of the Relocate laws:

  1. No blurring: one segment-pure inverse specialist per subtask
     (train_kitchen_subtask_inverse.py).
  2. Future coherence: each specialist tracks a demo-locked future index built
     from that subtask's demo segments.
  3. Input-feature emphasis: each specialist duplicates its object qpos dims in
     the conditioning (applied automatically via _append_emphasis from the ckpt).
  4. Firm predicate switch: advance to the next specialist ONLY when the env
     reports the current subtask complete (info['step_task_completions']) --
     the ground-truth analog of Relocate's firm possession switch.

The high level is a trivial fixed-order scheduler over the canonical D4RL
complete-v2 set; each subtask is a fresh short-horizon problem for its
specialist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.futures import DemoLockedFutureIndex
from jepa_robotics.algos.priors import InversePrior
from scripts.eval_flat_future_inverse import _append_emphasis
from scripts.train_kitchen_subtask_inverse import KITCHEN_TASKS, KITCHEN_OBJ_DIMS

# Per-subtask goal qpos values (gymnasium_robotics OBS_ELEMENT_GOALS) at the
# object dims, used only for diagnostics: min goal-distance reached per subtask.
KITCHEN_GOALS = {
    "microwave": np.array([-0.75], np.float32),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06], np.float32),
    "light switch": np.array([-0.69, -0.05], np.float32),
    "slide cabinet": np.array([0.37], np.float32),
}


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
        _append_emphasis(parts, s, self.ckpt)
        cond = torch.cat(parts, dim=-1)
        chunk = self.prior(cond).view(int(self.ckpt["H"]), int(self.ckpt["action_dim"])) * action_scale
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.dev)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.dev)
        return chunk.clamp(low, high).cpu().numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--specialist-paths", type=Path, nargs=4, required=True,
                   help="4 specialist checkpoints in canonical order: microwave kettle light-switch slide-cabinet")
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
                   help="If the current subtask does not complete within this many env steps, defer it and rotate to the next uncompleted subtask (0 disables; success needs all 4 in any order).")
    p.add_argument("--order", default="0,1,2,3",
                   help="Attempt order as subtask indices (0=microwave 1=kettle 2=light 3=slide). Success needs all 4 in any order; e.g. 0,1,3,2 does slide before light.")
    p.add_argument("--max-episode-steps", type=int, default=280)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--log-episodes", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    match_dims = tuple(int(x) for x in args.match_dims.split(","))
    specialists = [Specialist(pth, wm, norm, dev, args.target_horizon, args.geom_weight,
                              match_dims=match_dims, relock_margin=args.relock_margin)
                   for pth in args.specialist_paths]
    # Sanity: specialist order should match canonical task order.
    for i, sp in enumerate(specialists):
        if sp.name != KITCHEN_TASKS[i]:
            print(json.dumps({"event": "warn_order", "slot": i, "expected": KITCHEN_TASKS[i], "got": sp.name}), flush=True)

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    order = [int(x) for x in args.order.split(",")]

    successes, ntasks = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for sp in specialists:
            sp.reset()
        done_tasks: set = set()
        cached: list[np.ndarray] = []
        term = trunc = False
        info = {}
        switches = []
        traj = [flatten_obs(obs).copy()]

        def next_uncompleted(after):
            """Next subtask index in the configured attempt --order not yet done."""
            pos = order.index(after)
            for step in range(1, 5):
                j = order[(pos + step) % 4]
                if KITCHEN_TASKS[j] not in done_tasks:
                    return j
            return after

        # Firm scheduler with patience rotation: run the current subtask's
        # specialist; if it completes, advance to the next uncompleted subtask;
        # if it stalls for --subtask-patience env steps without completing, defer
        # it and rotate to the next uncompleted subtask (success needs all 4 in
        # ANY order, so an unreachable-from-here subtask should be retried later).
        cur = order[0]
        steps_on_cur = 0
        while not (term or trunc):
            if KITCHEN_TASKS[cur] in done_tasks:
                cur = next_uncompleted(cur)
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
            steps_on_cur += 1
            newly = set(info.get("step_task_completions", []))
            new_here = newly - done_tasks
            if new_here:
                done_tasks |= newly
                switches.append((int(info.get("tasks_done", len(done_tasks))), sorted(new_here)))
                cached = []  # firm switch: drop the rest of the current chunk
                if KITCHEN_TASKS[cur] in done_tasks and len(done_tasks) < 4:
                    cur = next_uncompleted(cur)
                    steps_on_cur = 0
                    specialists[cur].reset()
            elif args.subtask_patience > 0 and steps_on_cur >= args.subtask_patience and len(done_tasks) < 3:
                # Stalled: defer this subtask, rotate to the next uncompleted one.
                nxt = next_uncompleted(cur)
                if nxt != cur:
                    switches.append((int(info.get("tasks_done", len(done_tasks))), [f"defer:{KITCHEN_TASKS[cur]}->{KITCHEN_TASKS[nxt]}"]))
                    cur = nxt
                    specialists[cur].reset()
                    cached = []
                steps_on_cur = 0
        success = float(info.get("is_success", 0.0))
        nt = int(info.get("tasks_done", len(done_tasks)))
        successes.append(success)
        ntasks.append(nt)
        if args.log_episodes:
            tj = np.asarray(traj, dtype=np.float32)
            min_dist = {}
            for name in KITCHEN_TASKS:
                lo, hi = KITCHEN_OBJ_DIMS[name]
                d = np.linalg.norm(tj[:, lo:hi] - KITCHEN_GOALS[name][None], axis=-1)
                min_dist[name] = round(float(d.min()), 3)
            print(json.dumps({"event": "kitchen_episode", "episode": ep, "success": success,
                              "tasks_done": nt, "completed": sorted(done_tasks),
                              "min_goal_dist": min_dist, "switches": switches}), flush=True)

    env.close()
    row = {
        "event": "kitchen_subtask_eval",
        "task": task.name,
        "specialist_paths": [str(x) for x in args.specialist_paths],
        "episodes": int(args.episodes),
        "success_rate": float(np.mean(successes)),
        "mean_tasks": float(np.mean(ntasks)),
        "tasks_hist": {str(i): int(np.sum(np.asarray(ntasks) == i)) for i in range(5)},
        "exec_k": int(args.exec_k),
        "target_horizon": int(args.target_horizon),
        "geom_weight": float(args.geom_weight),
        "action_scale": float(args.action_scale),
        "torch_seed": args.torch_seed,
        "seed": args.seed,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
