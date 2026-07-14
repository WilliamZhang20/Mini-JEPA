"""Hierarchical SSL reorientation with a future-conditioned FLOW low-level.

Why flow over CEM/MPPI: greedy planners stall at the regrasp ceiling (~25-30 deg)
because continuing a rotation needs a finger-gait that *temporarily worsens* the
objective, so a myopic cost-minimizer never proposes it. A future-conditioned flow
prior instead samples action chunks from the data distribution — including the
regrasp chunks that occur in the exploration data's large-rotation episodes — so it
can propose gaits that CEM would never sample.

Structure (pure SSL, no reward / no demos):
* abstract level: split the rotation to the goal into ~step-deg SO(3) subgoals.
* goal-directed retrieval: for the current subgoal orientation, fetch a REAL state
  from the flow's exploration bank whose object is at that orientation and whose
  hand pose is closest to now -> encode it as z_goal (the demo-free future).
* flow low-level: sample N chunks from flow(a | z_t, z_goal), select by signed
  forward-progress (keeps gaits, steers direction), execute, replan.

``FlowController`` is reused by the self-goaling loop (train_selfgoal_ssl.py) to
COLLECT directed attempts, so the planning logic lives here once.
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
from jepa_robotics.algos.priors import EpsNet, make_ddpm, sample_action_chunks
from jepa_robotics.models import DexterousFlowPrior
from scripts.eval_subgoal_controllability import quat_geodesic, Controller
from scripts.eval_hierarchical_reorient import step_toward


class GoalDirectedBank:
    """Retrieve a real exploration state whose object orientation matches a target
    (weighted by hand-pose proximity to now). This is the demo-free 'future'."""

    def __init__(self, bank_states, ag, proprio_hi, hand_weight):
        self.states = bank_states.astype(np.float32)
        self.ag = ag
        self.obj_q = self.states[:, ag + 3:ag + 7]
        self.obj_q = self.obj_q / (np.linalg.norm(self.obj_q, axis=-1, keepdims=True) + 1e-9)
        self.hand = self.states[:, :proprio_hi]
        self.hand_weight = hand_weight

    def query(self, cur_state, target_q):
        tq = target_q / (np.linalg.norm(target_q) + 1e-9)
        dot = np.abs(self.obj_q @ tq).clip(0, 1)
        obj_ang = 2.0 * np.arccos(dot)
        hand_d = np.linalg.norm(self.hand - cur_state[:self.hand.shape[1]], axis=-1)
        return self.states[int(np.argmin(obj_ang + self.hand_weight * hand_d))]


class FlowController:
    """Hierarchical SO(3) + flow low-level. Stateful over an episode: reset(q_goal)
    then act(state) each env step; carries the moving subgoal and action buffer."""

    def __init__(self, *, wm, norm, spec, ck, flow, ddpm, index, dev, lo, hi,
                 candidates, select, flow_steps, step_deg, inner_thr_deg, max_hold, exec_k,
                 fine_deg=0.0, fine_candidates=512, cfg_weight=1.0):
        self.wm, self.norm, self.spec, self.ck, self.flow, self.ddpm = wm, norm, spec, ck, flow, ddpm
        self.index, self.dev, self.lo, self.hi = index, dev, lo, hi
        self.candidates, self.select, self.flow_steps = candidates, select, flow_steps
        self.exec_k = exec_k
        self.cfg_weight = cfg_weight
        self.synthetic_goal = False
        self.step_rad = np.radians(step_deg)
        self.inner_thr = np.radians(inner_thr_deg)
        self.max_hold = max_hold
        # Fine-precision CEM primitive that takes over near the goal to hit the
        # tight success threshold the coarse gaiting flow overshoots.
        self.fine_rad = np.radians(fine_deg)
        self.fine = Controller(wm, norm, spec, dev, int(ck["H"]), fine_candidates, 5, 0.1, 0.5,
                               planner="cem", exec_k=int(ck["H"])) if fine_deg > 0 else None
        self.H, self.A = int(ck["H"]), int(ck["action_dim"])
        self.ag, self.dgo = spec.obs_dim, spec.obs_dim + spec.goal_dim
        horizons = list(ck.get("future_horizons", [self.H]))
        self.h_token_val = float(max(horizons)) / float(max(horizons))
        self.concat_raw = bool(ck.get("concat_raw", False))
        emph = ck.get("emphasis_dims", None)
        self.emph_lo, self.emph_hi = (int(x) for x in emph.split(",")) if emph else (None, None)
        self.emph_rep = int(ck.get("emphasis_repeat", 0))
        self._lo_np, self._hi_np = lo.cpu().numpy(), hi.cpu().numpy()

    def reset(self, q_goal):
        self.q_goal = q_goal.copy()
        self.subgoal_q = None
        self.buf = []
        self._fine_buf = []
        self.held = 0
        if self.fine is not None:
            self.fine.reset()

    def _cond_from(self, z, z_goal, s_raw_tensor, g_raw_tensor):
        parts = [z, z_goal, torch.full((z.shape[0], 1), self.h_token_val, dtype=z.dtype, device=self.dev)]
        if self.concat_raw:
            parts.extend([s_raw_tensor, g_raw_tensor])
        if self.emph_lo is not None and self.emph_rep > 0:
            parts.append(s_raw_tensor[:, self.emph_lo:self.emph_hi].repeat(1, self.emph_rep))
        return torch.cat(parts, dim=-1)

    def _sample(self, cond):
        ch = sample_action_chunks(self.flow, self.ddpm, cond, self.ck["chunk_dim"], self.dev,
                                  objective="flow", flow_steps=self.flow_steps, cfg_weight=self.cfg_weight)
        return ch.view(cond.shape[0], self.H, self.A).clamp(self.lo, self.hi)

    def _goal_state_np(self, raw, target_q):
        if self.synthetic_goal:
            gs = raw.copy(); gs[self.ag + 3:self.ag + 7] = target_q
            return gs
        return self.index.query(raw, target_q)

    @torch.no_grad()
    def _plan(self, raw, subgoal_pose):
        s = torch.from_numpy(self.norm.encode(raw).astype(np.float32)).unsqueeze(0).to(self.dev)
        g_np = self.norm.encode(self._goal_state_np(raw, subgoal_pose[3:])).astype(np.float32)
        g = torch.from_numpy(g_np).unsqueeze(0).to(self.dev)
        z, z_goal = self.wm.encode(s), self.wm.encode_target(g)
        N = self.candidates
        acts = self._sample(self._cond_from(z, z_goal, s, g).repeat(N, 1))
        if self.select == "trust":
            return acts[0].cpu().numpy()
        tq = torch.as_tensor(subgoal_pose[3:] / (np.linalg.norm(subgoal_pose[3:]) + 1e-9),
                             dtype=torch.float32, device=self.dev).view(1, 4)
        qcur = torch.as_tensor(raw[self.ag + 3:self.ag + 7] / (np.linalg.norm(raw[self.ag + 3:self.ag + 7]) + 1e-9),
                               dtype=torch.float32, device=self.dev).view(1, 4)
        start_gap = 2.0 * torch.acos((qcur * tq).sum(-1).abs().clamp(max=1.0))
        z_mid = self.wm.predict_rollout(z.expand(N, -1), acts, self.H)[:, -1]  # [N,latent]
        if self.select == "lookahead":
            # Imagine one more flow chunk from the mid latent (total 2*H steps, a
            # full regrasp cycle) and score the END orientation. A gait that goes
            # backward then forward nets forward here, so it is not rejected.
            mid_state = self.norm.decode_tensor(self.wm.state_probe(z_mid))            # [N,state] raw
            mid_norm = torch.from_numpy(self.norm.encode(mid_state.cpu().numpy()).astype(np.float32)).to(self.dev)
            cont = self._sample(self._cond_from(z_mid, z_goal.expand(N, -1), mid_norm, g.expand(N, -1)))
            z_end = self.wm.predict_rollout(z_mid, cont, self.H)[:, -1]
            pred_q = self.norm.decode_tensor(self.wm.state_probe(z_end.unsqueeze(1)))[:, 0, self.ag + 3:self.dgo]
        else:  # progress: single-chunk terminal
            pred_q = self.norm.decode_tensor(self.wm.state_probe(z_mid.unsqueeze(1)))[:, 0, self.ag + 3:self.dgo]
        pred_q = pred_q / torch.linalg.vector_norm(pred_q, dim=-1, keepdim=True).clamp_min(1e-6)
        end_gap = 2.0 * torch.acos((pred_q * tq).sum(-1).abs().clamp(max=1.0))
        return acts[int(torch.argmax(start_gap - end_gap))].cpu().numpy()

    def act(self, s):
        ag = self.ag
        q_cur = s[ag + 3:ag + 7]
        # Fine stage: within fine_rad of the true goal, servo it directly with the
        # precise CEM primitive instead of the coarse flow.
        if self.fine is not None and quat_geodesic(q_cur, self.q_goal) < self.fine_rad:
            goal_pose = np.concatenate([s[ag:ag + 3], self.q_goal]).astype(np.float32)
            if not getattr(self, "_fine_buf", None):
                plan = self.fine._plan(s, goal_pose, self.lo, self.hi)
                self._fine_buf = [plan[i].copy() for i in range(self.fine.exec_k)]
            return np.clip(self._fine_buf.pop(0), self._lo_np, self._hi_np).astype(np.float32)
        self._fine_buf = []
        if self.subgoal_q is None:
            self.subgoal_q = step_toward(q_cur, self.q_goal, self.step_rad)
        if (quat_geodesic(q_cur, self.subgoal_q) < self.inner_thr or self.held >= self.max_hold) \
                and quat_geodesic(q_cur, self.q_goal) > 1e-3:
            self.subgoal_q = step_toward(q_cur, self.q_goal, self.step_rad)
            self.held = 0
            self.buf = []
        if not self.buf:
            subgoal_pose = np.concatenate([s[ag:ag + 3], self.subgoal_q]).astype(np.float32)
            plan_chunk = self._plan(s, subgoal_pose)
            self.buf = [plan_chunk[i].copy() for i in range(max(1, min(self.exec_k, len(plan_chunk))))]
        self.held += 1
        return np.clip(self.buf.pop(0), self._lo_np, self._hi_np).astype(np.float32)


def build_flow_controller(model_path, flow_path, dev, lo, hi, *, candidates, select,
                          flow_steps, step_deg, inner_thr_deg, max_hold, exec_k,
                          hand_weight=0.02, bank_sub=12000, fine_deg=0.0, fine_candidates=512, cfg_weight=1.0):
    wm, norm, spec, cfg = load_jepa_artifact(model_path, dev)
    ck = torch.load(flow_path, map_location=dev, weights_only=False)
    if ck.get("prior_arch", "mlp") == "dit":
        flow = DexterousFlowPrior(ck["chunk_dim"], ck["cond_dim"], hidden=ck["hidden"],
                                  n_blocks=ck["n_blocks"], heads=ck.get("heads", 6),
                                  action_dim=int(ck["action_dim"])).to(dev)
    else:
        flow = EpsNet(ck["chunk_dim"], ck["cond_dim"], ck["hidden"], n_blocks=ck["n_blocks"]).to(dev)
    flow.load_state_dict(ck["ema"] if "ema" in ck else ck["state_dict"])
    flow.eval()
    ddpm = make_ddpm(int(ck["diffusion_steps"]), dev)
    bank = ck["bank_states"]
    if len(bank) > bank_sub:
        bank = bank[np.random.default_rng(0).choice(len(bank), bank_sub, replace=False)]
    index = GoalDirectedBank(bank, spec.obs_dim, 48, hand_weight)
    ctrl = FlowController(wm=wm, norm=norm, spec=spec, ck=ck, flow=flow, ddpm=ddpm, index=index,
                          dev=dev, lo=lo, hi=hi, candidates=candidates, select=select,
                          flow_steps=flow_steps, step_deg=step_deg, inner_thr_deg=inner_thr_deg,
                          max_hold=max_hold, exec_k=exec_k, fine_deg=fine_deg, fine_candidates=fine_candidates,
                          cfg_weight=cfg_weight)
    return ctrl, wm, norm, spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--flow-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=60000)
    p.add_argument("--max-episode-steps", type=int, default=150)
    p.add_argument("--step-deg", type=float, default=25.0)
    p.add_argument("--inner-thr-deg", type=float, default=12.0)
    p.add_argument("--max-hold", type=int, default=16)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--select", default="progress", choices=["progress", "trust", "lookahead"])
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--fine-deg", type=float, default=0.0, help="switch to fine CEM primitive within this angle of the goal (0=off)")
    p.add_argument("--fine-candidates", type=int, default=512)
    p.add_argument("--synthetic-goal", action="store_true", help="condition on current-state-with-object-rotated-to-subgoal instead of a retrieved bank state (cleaner directional signal)")
    p.add_argument("--cfg-weight", type=float, default=1.0, help="classifier-free guidance weight (>1 amplifies goal-conditioning; needs a flow trained with --cond-dropout)")
    p.add_argument("--hand-weight", type=float, default=0.02)
    p.add_argument("--bank-sub", type=int, default=12000)
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    ctrl, wm, norm, spec = build_flow_controller(
        args.model_path, args.flow_path, dev, lo, hi, candidates=args.candidates, select=args.select,
        flow_steps=args.flow_steps, step_deg=args.step_deg, inner_thr_deg=args.inner_thr_deg,
        max_hold=args.max_hold, exec_k=args.exec_k, hand_weight=args.hand_weight, bank_sub=args.bank_sub,
        fine_deg=args.fine_deg, fine_candidates=args.fine_candidates, cfg_weight=args.cfg_weight)
    ctrl.synthetic_goal = args.synthetic_goal
    ag, dgo = spec.obs_dim, spec.obs_dim + spec.goal_dim

    successes, final_gaps = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        s = flatten_obs(obs)
        ctrl.reset(s[dgo + 3:dgo + 7])
        term = trunc = False; info = {}
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(ctrl.act(s))
            s = flatten_obs(obs)
        successes.append(float(info.get("is_success", 0.0)))
        final_gaps.append(np.degrees(quat_geodesic(s[ag + 3:ag + 7], s[dgo + 3:dgo + 7])))
    env.close()
    print(json.dumps({"event": "dexterous_flow_eval", "task": task.name,
                      "episodes": args.episodes, "success_rate": round(float(np.mean(successes)), 3),
                      "median_final_gap_deg": round(float(np.median(final_gaps)), 1),
                      "step_deg": args.step_deg, "candidates": args.candidates, "select": args.select,
                      "max_episode_steps": args.max_episode_steps}), flush=True)


if __name__ == "__main__":
    main()
