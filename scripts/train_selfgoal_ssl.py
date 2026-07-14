"""Self-goaling SSL loop for dexterous in-hand reorientation (no reward, no demos).

Replaces the expert demos every contact-rich win in this repo has leaned on with
self-supervised frontier expansion. The exploration data (OU) is UNDIRECTED, so a
flow prior trained on it can rotate the object (it breaks the greedy regrasp
ceiling) but cannot steer to a *specific* goal. This loop manufactures DIRECTED
data with no reward and no demos:

    round r:
      1. self-goals: sample a target orientation a frontier-band angle away from a
         fresh start (curriculum: band widens as the controller succeeds).
      2. attempt: drive the current FLOW controller (ceiling-breaking gaits) toward
         each self-goal; log the whole trajectory regardless of success.
      3. hindsight: append trajectories to the buffer — the JEPA/flow objectives
         relabel achieved futures themselves; the actions are now GOAL-DIRECTED.
      4. retrain: WM (train_dexterous_jepa) then flow (train_flat_future_flow with
         object-pose emphasis) on the enlarged directed buffer. No planning/model
         logic is duplicated here — collection reuses scripts.eval_dexterous_flow's
         FlowController; training delegates to the entry-point scripts.
      5. frontier: widen the band when reach-success is high.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import Episode, load_episodes_npz, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.tasks import resolve_task
from scripts.eval_subgoal_controllability import quat_geodesic, z_subgoal_quat
from scripts.eval_dexterous_flow import build_flow_controller

REPO = Path(__file__).resolve().parent.parent


def collect_round(task, spec, ctrl, env, n_goals, control_steps, f_lo, f_hi, thr, rng):
    """Attempt n_goals frontier-band self-goals; return (episodes, reach_success)."""
    ag = spec.obs_dim
    episodes, reached = [], 0
    for _ in range(n_goals):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        s = flatten_obs(obs)
        angle = float(rng.uniform(f_lo, f_hi)) * (1.0 if rng.random() < 0.5 else -1.0)
        q_goal = z_subgoal_quat(s[ag + 3:ag + 7], angle)   # directed target a band-angle away
        ctrl.reset(q_goal)
        states, actions = [s.copy()], []
        best = quat_geodesic(s[ag + 3:ag + 7], q_goal)
        for _ in range(control_steps):
            a = ctrl.act(s)
            obs, _, term, trunc, _ = env.step(a)
            s = flatten_obs(obs)
            states.append(s.copy()); actions.append(a.copy())
            best = min(best, quat_geodesic(s[ag + 3:ag + 7], q_goal))
            if best < thr or term or trunc:
                break
        reached += int(best < thr)
        if len(actions) >= 2:
            episodes.append(Episode(states=np.asarray(states, np.float32),
                                    actions=np.asarray(actions, np.float32)))
    return episodes, reached / max(1, n_goals)


def retrain(task_name, buf_path, wm_out, flow_out, ag, goal_dim, wm_steps, flow_steps, dev):
    common = ["--task", task_name, "--episodes-npz", str(buf_path), "--device", dev]
    subprocess.run([sys.executable, "scripts/train_dexterous_jepa.py", *common, "--out", str(wm_out),
                    "--horizons", "1,2,4,8,16", "--object-dims", f"{ag},{ag + goal_dim}",
                    "--contact-dims", f"{ag},{ag + goal_dim}", "--lambda-object", "5.0",
                    "--ensemble-heads", "3", "--steps", str(wm_steps)], check=True, cwd=str(REPO))
    subprocess.run([sys.executable, "scripts/train_flat_future_flow.py", "--model-path", str(wm_out),
                    "--episodes-npz", str(buf_path), "--out", str(flow_out), "--chunk", "8",
                    "--future-horizons", "8,16", "--concat-raw", "--emphasis-dims", f"{ag},{ag + goal_dim}",
                    "--emphasis-repeat", "8", "--train-steps", str(flow_steps),
                    "--device", dev], check=True, cwd=str(REPO))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--seed-npz", type=Path, required=True)
    p.add_argument("--init-model", type=Path, required=True)
    p.add_argument("--init-flow", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--goals-per-round", type=int, default=160)
    p.add_argument("--control-steps", type=int, default=70)
    p.add_argument("--reach-threshold-deg", type=float, default=12.0)
    p.add_argument("--frontier-lo-deg", type=float, default=20.0)
    p.add_argument("--frontier-hi-deg", type=float, default=45.0)
    p.add_argument("--frontier-grow-deg", type=float, default=25.0)
    p.add_argument("--grow-at", type=float, default=0.4)
    p.add_argument("--keep-seed", action="store_true", help="keep OU seed data in the buffer across rounds")
    p.add_argument("--wm-steps", type=int, default=18000)
    p.add_argument("--flow-train-steps", type=int, default=25000)
    p.add_argument("--candidates", type=int, default=48)
    p.add_argument("--select", default="progress", choices=["progress", "trust"])
    p.add_argument("--step-deg", type=float, default=25.0)
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    args.workdir.mkdir(parents=True, exist_ok=True)
    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=0, max_episode_steps=args.max_episode_steps)
    spec = obs_spec_from_env(env)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    ag = spec.obs_dim
    thr = np.radians(args.reach_threshold_deg)
    f_lo, f_hi = np.radians(args.frontier_lo_deg), np.radians(args.frontier_hi_deg)
    rng = np.random.default_rng(0)

    seed_eps = load_episodes_npz(args.seed_npz)
    directed_eps = []                       # accumulated directed attempts
    model_path, flow_path = args.init_model, args.init_flow

    for r in range(args.rounds):
        ctrl, _, _, _ = build_flow_controller(
            model_path, flow_path, dev, lo, hi, candidates=args.candidates, select=args.select,
            flow_steps=16, step_deg=args.step_deg, inner_thr_deg=12.0, max_hold=16, exec_k=args.exec_k)
        new_eps, reach = collect_round(task, spec, ctrl, env, args.goals_per_round,
                                       args.control_steps, f_lo, f_hi, thr, rng)
        directed_eps += new_eps
        buffer = (seed_eps + directed_eps) if args.keep_seed else (directed_eps if directed_eps else seed_eps)
        buf_path = args.workdir / f"buffer_round{r}.npz"
        save_episodes_npz(buf_path, buffer, spec)
        grew = reach >= args.grow_at
        if grew:
            f_hi += np.radians(args.frontier_grow_deg)
        print(json.dumps({"event": "selfgoal_round", "round": r, "goals": args.goals_per_round,
                          "reach_success": round(reach, 3), "directed_eps": len(directed_eps),
                          "buffer_eps": len(buffer),
                          "frontier_band_deg": [round(np.degrees(f_lo), 1), round(np.degrees(f_hi), 1)],
                          "frontier_grew": bool(grew)}), flush=True)
        wm_out = args.workdir / f"wm_round{r}.pt"
        flow_out = args.workdir / f"flow_round{r}.pt"
        retrain(args.task, buf_path, wm_out, flow_out, ag, spec.goal_dim,
                args.wm_steps, args.flow_train_steps, dev.type)
        model_path, flow_path = wm_out, flow_out
        print(json.dumps({"event": "selfgoal_retrained", "round": r,
                          "wm": str(wm_out), "flow": str(flow_out)}), flush=True)

    env.close()
    print(json.dumps({"event": "selfgoal_done", "final_model": str(model_path),
                      "final_flow": str(flow_path),
                      "final_frontier_hi_deg": round(np.degrees(f_hi), 1)}), flush=True)


if __name__ == "__main__":
    main()
