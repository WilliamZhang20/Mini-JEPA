"""Crux test for the hierarchical-JEPA + dexterous-flow SSL bet.

The abstract SO(3) planner can only compose a long reorientation if the low-level
has a *direction-controllable* small-reorientation primitive. This probes exactly
that, with no reward and no demos: pick a start state, synthesize an orientation
SUBGOAL a fixed angle delta away around Z (either sign), and ask whether the
WM-based low-level can drive the object to it — and whether it beats random action
of the same magnitude.

Reports, per delta:
* controller subgoal-reach success (geodesic angle to subgoal < threshold),
* random-action subgoal-reach success (the null: does shaking hit it as often?),
* signed directedness: mean geodesic gap closed toward the commanded subgoal.

If the controller reliably reaches small +/- subgoals and clearly beats random,
the composable primitive exists and the hierarchical stack is worth building. If
+/- 15-30 deg is no better than random, the primitive is not in the data and the
lever is self-goaling exploration (train_selfgoal_ssl.py), not a bigger planner.
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

from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w,x,y,z) order."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def quat_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(2.0 * np.arccos(min(1.0, abs(float(a @ b)))))


def z_subgoal_quat(q_cur: np.ndarray, delta_rad: float) -> np.ndarray:
    """Compose a world-frame rotation of delta about Z onto the current object quat."""
    qz = np.array([np.cos(delta_rad / 2.0), 0.0, 0.0, np.sin(delta_rad / 2.0)], dtype=np.float32)
    return quat_mul(qz, q_cur)


class Controller:
    """Self-contained model-predictive low-level toward an arbitrary object-pose
    subgoal (geodesic-correct cost), independent of the env's own desired_goal.

    Two samplers over the shared WM rollout + cost:
    * ``cem``  — elite-truncation, no warm-start (each plan from scratch). Fine for
      a *fixed* subgoal held for many steps (the controllability probe).
    * ``mppi`` — softmax path-integral update with a *warm-started* nominal control
      sequence shifted one step each call. Temporally consistent, so it holds
      momentum under a moving subgoal (the hierarchical carrot) where a
      from-scratch CEM stalls.
    """

    def __init__(self, wm, norm, spec, dev, H, N, iters, elite_frac, init_std,
                 planner="cem", exec_k=None, temperature=1.0):
        self.wm, self.norm, self.spec, self.dev = wm, norm, spec, dev
        self.H, self.N, self.iters = H, N, iters
        self.elite = max(1, int(N * elite_frac))
        self.init_std, self.planner, self.temperature = init_std, planner, temperature
        self.exec_k = exec_k if exec_k is not None else H   # cem default: exec the whole plan
        self.ag_lo = spec.obs_dim
        self.ag_hi = spec.obs_dim + spec.goal_dim
        self._buf = []
        self._nominal = None  # [H,A] warm-start for mppi

    def reset(self):
        self._buf = []
        self._nominal = None

    def _cost(self, z, acts, tgt):
        roll = self.wm.predict_rollout(z.expand(acts.shape[0], -1), acts, self.H)
        pred = self.norm.decode_tensor(self.wm.state_probe(roll))
        pred_ag = pred[:, :, self.ag_lo:self.ag_hi]
        pos = torch.linalg.vector_norm(pred_ag[:, :, :3] - tgt[:3], dim=-1)
        qa = pred_ag[:, :, 3:]
        qa = qa / torch.linalg.vector_norm(qa, dim=-1, keepdim=True).clamp_min(1e-6)
        qb = (tgt[3:] / torch.linalg.vector_norm(tgt[3:]).clamp_min(1e-6)).view(1, 1, 4)
        dot = (qa * qb).sum(-1).abs().clamp(max=1.0)
        rot = 2.0 * torch.acos(dot)
        dist = 10.0 * pos + rot
        return dist[:, -1] + 0.25 * dist.mean(1)  # terminal + path

    @torch.no_grad()
    def _plan(self, raw, target_pose, lo, hi):
        A = self.spec.action_dim
        sn = torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev)
        z = self.wm.encode(sn)
        tgt = torch.as_tensor(target_pose, dtype=torch.float32, device=self.dev)
        if self.planner == "mppi":
            mean = self._nominal if self._nominal is not None else torch.zeros(self.H, A, device=self.dev)
            for _ in range(self.iters):
                eps = torch.randn(self.N, self.H, A, device=self.dev)
                acts = (mean.unsqueeze(0) + self.init_std * eps).clamp(lo, hi)
                cost = self._cost(z, acts, tgt)
                # scale by cost spread so temperature is dimensionless: temp too
                # large -> uniform weights -> weighted mean of zero-mean noise ~ 0
                # (the hand freezes). Normalizing keeps the update well-conditioned.
                w = torch.softmax(-(cost - cost.min()) / (self.temperature * cost.std().clamp_min(1e-6)), dim=0)
                mean = (w.view(-1, 1, 1) * acts).sum(0)
            self._nominal = torch.cat([mean[1:], mean[-1:]], dim=0)  # shift-1 warm-start
            return mean.cpu().numpy()
        # cem
        mean = torch.zeros(self.H, A, device=self.dev)
        std = torch.full((self.H, A), self.init_std, device=self.dev)
        best_first = None
        for _ in range(self.iters):
            eps = torch.randn(self.N, self.H, A, device=self.dev)
            acts = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(lo, hi)
            cost = self._cost(z, acts, tgt)
            order = torch.argsort(cost)
            elite = acts[order[: self.elite]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.02)
            best_first = acts[order[0]]
        return best_first.cpu().numpy()

    def act(self, obs, target_pose, lo, hi):
        if not self._buf:
            plan = self._plan(flatten_obs(obs), target_pose, lo, hi)
            self._buf = [plan[i].copy() for i in range(max(1, min(self.exec_k, len(plan))))]
        return np.clip(self._buf.pop(0), lo.cpu().numpy(), hi.cpu().numpy()).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--deltas-deg", default="15,30,45,90")
    p.add_argument("--trials", type=int, default=40, help="start states per delta (each tried +delta and -delta)")
    p.add_argument("--control-steps", type=int, default=50)
    p.add_argument("--reach-threshold-deg", type=float, default=10.0)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--candidates", type=int, default=512)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--seed", type=int, default=50000)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=1000)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    ctrl = Controller(wm, norm, spec, dev, args.horizon, args.candidates, args.iters, 0.1, 0.5)
    ag = spec.obs_dim
    thr = np.radians(args.reach_threshold_deg)
    deltas = [np.radians(float(x)) for x in args.deltas_deg.split(",")]
    rng = np.random.default_rng(args.seed)

    def drive(obs, target_pose, mode):
        """Run control-steps toward target_pose; return best geodesic gap reached."""
        ctrl.reset()
        s = flatten_obs(obs)
        best = quat_geodesic(s[ag + 3:ag + 7], target_pose[3:])
        a = np.zeros(spec.action_dim, dtype=np.float32)
        for _ in range(args.control_steps):
            if mode == "controller":
                act = ctrl.act(obs, target_pose, lo, hi)
            else:  # random OU of the same magnitude as data collection
                a = a + 0.15 * (-a) + 0.3 * rng.standard_normal(spec.action_dim).astype(np.float32)
                act = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, _, term, trunc, _ = env.step(act)
            s = flatten_obs(obs)
            best = min(best, quat_geodesic(s[ag + 3:ag + 7], target_pose[3:]))
            if term or trunc:
                break
        return best

    rows = []
    for d in deltas:
        c_succ = r_succ = 0
        c_closed = r_closed = 0.0
        n = 0
        for t in range(args.trials):
            for sign in (+1.0, -1.0):
                obs, _ = env.reset(seed=args.seed + t)
                s0 = flatten_obs(obs)
                q_cur = s0[ag + 3:ag + 7].copy()
                pos_cur = s0[ag:ag + 3].copy()
                target_pose = np.concatenate([pos_cur, z_subgoal_quat(q_cur, sign * d)])
                start_gap = quat_geodesic(q_cur, target_pose[3:])
                # controller
                obs, _ = env.reset(seed=args.seed + t)
                bc = drive(obs, target_pose, "controller")
                # random (same start)
                obs, _ = env.reset(seed=args.seed + t)
                br = drive(obs, target_pose, "random")
                c_succ += int(bc < thr); r_succ += int(br < thr)
                c_closed += (start_gap - bc); r_closed += (start_gap - br)
                n += 1
        row = {"event": "subgoal_controllability", "task": task.name, "delta_deg": round(np.degrees(d), 1),
               "trials": n, "reach_thr_deg": args.reach_threshold_deg,
               "controller_success": round(c_succ / n, 3), "random_success": round(r_succ / n, 3),
               "controller_gap_closed_deg": round(np.degrees(c_closed / n), 1),
               "random_gap_closed_deg": round(np.degrees(r_closed / n), 1)}
        print(json.dumps(row), flush=True)
        rows.append(row)
    env.close()
    verdict = any(r["controller_success"] >= 0.5 and r["controller_success"] > r["random_success"] + 0.15 for r in rows)
    print(json.dumps({"event": "controllability_verdict",
                      "primitive_controllable": bool(verdict),
                      "note": "controllable if any small delta reaches >=0.5 success AND clearly beats random"}), flush=True)


if __name__ == "__main__":
    main()
