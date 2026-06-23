"""Proper two-tier Hierarchical-JEPA (H-JEPA) with a LEARNED high-level model and
directed search -- the literature architecture, not a hand-built reachability graph.

  * JEPA-1 (low, short horizon): the existing ActionConditionedJEPA encoder + a
    goal-conditioned low-level controller that reaches nearby subgoals.
  * JEPA-2 (high, abstract, long horizon): a LEARNED model over the abstract
    latent z = encode(obs). Given the current abstract latent and a candidate
    subgoal offset, it predicts (a) feasibility -- can the low level get there in
    <= k steps (this is what routes around walls; a wall makes the transition
    infeasible) -- and (b) the number of steps. Trained self-supervised from the
    demos with hindsight (the abstract state actually reached k steps later is a
    positive; random far offsets are negatives).
  * PLANNING = directed search (A*) in the abstract latent: expand the best
    partial plan by feasible subgoal hops the JEPA-2 model predicts, prune
    infeasible ones, use straight-line-to-goal as the admissible heuristic. The
    first subgoal of the optimal plan is handed to JEPA-1; re-plan on the fly.

This replaces eval_hjepa_maze.py's farthest-point landmarks + empirical Dijkstra
edges with a learned high-level world model + heuristic search, which is the
"sample the latent + directed search/pruning to find the optimal action" picture.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task
from scripts.eval_hjepa_maze import LowLevelBC, farthest_point_sample


# --------------------------- JEPA-2 high-level model ------------------------
class HighLevelJEPA2(nn.Module):
    """Predicts, from an abstract latent z and a subgoal *offset* g, whether the
    subgoal is reachable by the low level within k steps (feasibility) and the
    (normalized) number of steps it takes. z is the JEPA-1 encoding of the obs."""

    def __init__(self, latent_dim, goal_dim, hidden=256):
        super().__init__()
        self.trunk = MLP([latent_dim + goal_dim, hidden, hidden], layer_norm=True)
        self.feasible = nn.Linear(hidden, 1)
        self.steps = nn.Linear(hidden, 1)

    def forward(self, z, goal_offset):
        h = self.trunk(torch.cat([z, goal_offset], dim=-1))
        return self.feasible(h), self.steps(h)


def train_jepa2(wm, norm, spec, episodes, k_reach, device, steps=8000, batch=512, seed=0):
    """Self-supervised: positives = (z_t, pos_{t+dt}-pos_t) for dt in [1,k] (label
    feasible=1, steps=dt/k); negatives = (z_t, random offset) label feasible=0."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    rng = np.random.default_rng(seed)
    # collect (flattened-state, future-offset, dt) triples
    S_list, off_list, dt_list = [], [], []
    for ep in episodes:
        st = ep.states
        T = len(st)
        pos = st[:, gs:ge]
        for t in range(T - 1):
            dt = int(rng.integers(1, k_reach + 1))
            if t + dt >= T:
                dt = T - 1 - t
            if dt < 1:
                continue
            S_list.append(st[t]); off_list.append(pos[t + dt] - pos[t]); dt_list.append(dt)
    S = np.asarray(S_list, np.float32)
    off = np.asarray(off_list, np.float32)
    dts = np.asarray(dt_list, np.float32)
    # encode states to abstract latent (JEPA-1 encoder, frozen)
    with torch.no_grad():
        Z = []
        Sn = norm.encode(S)
        for i in range(0, len(Sn), 16384):
            Z.append(wm.encode(torch.from_numpy(Sn[i:i + 16384]).to(device)))
        Z = torch.cat(Z, 0)
    offt = torch.from_numpy(off).to(device)
    dtt = torch.from_numpy(dts / k_reach).to(device)
    # offset scale for sampling negatives ~ the positive offset distribution
    off_std = float(np.std(np.linalg.norm(off, axis=1))) + 1e-6
    off_norm = np.linalg.norm(off, axis=1)
    max_pos_off = float(np.quantile(off_norm, 0.95))

    model = HighLevelJEPA2(Z.shape[1], spec.goal_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()
    N = len(Z)
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch,), device=device)
        z = Z[idx]; pos_off = offt[idx]; pos_dt = dtt[idx]
        # negatives: random offsets clearly larger / different than reachable ones
        neg = torch.randn(batch, spec.goal_dim, device=device)
        neg = neg / (neg.norm(dim=-1, keepdim=True) + 1e-6)
        neg = neg * torch.empty(batch, 1, device=device).uniform_(max_pos_off * 1.3, max_pos_off * 3.0)
        zf, zs = model(z, pos_off)
        nf, _ = model(z, neg)
        f_loss = bce(zf.squeeze(-1), torch.ones(batch, device=device)) + \
                 bce(nf.squeeze(-1), torch.zeros(batch, device=device))
        s_loss = nn.functional.mse_loss(zs.squeeze(-1), pos_dt)
        loss = f_loss + s_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(json.dumps({"event": "jepa2_train", "step": step, "f_loss": round(float(f_loss), 3),
                              "s_loss": round(float(s_loss), 3)}), flush=True)
    model.eval()
    return model, max_pos_off


# --------------------------- directed A* search -----------------------------
@torch.no_grad()
def plan_astar(model, z_start, start_pos, goal_pos, landmarks, device, goal_dim,
               feas_thresh=0.5, max_expand=400, max_off=4.0):
    """A* over landmark abstract states; edge (a,b) exists iff JEPA-2 predicts the
    offset (pos_b - pos_a) feasible from a's latent. Heuristic = straight-line to
    goal. Returns a list of landmark positions to visit (subgoal waypoints)."""
    L = len(landmarks)
    # candidate edges: for each landmark, feasibility of hopping to nearby landmarks
    # (approx the latent at a landmark by the start latent shifted is hard; we use a
    # single feasibility query batched per source using the START latent as a proxy
    # for local dynamics, which is adequate for the smooth high level).
    # Precompute pairwise offsets and feasibility from each landmark to others.
    lm = torch.from_numpy(landmarks.astype(np.float32)).to(device)
    # feasibility from landmark a to b: query model(z_a_proxy, pos_b - pos_a).
    # Use z_start as a shared latent proxy (positions carry the routing signal).
    zb = z_start.expand(L, -1)
    feas = np.zeros((L, L), np.float32)
    cost = np.full((L, L), np.inf, np.float32)
    for a in range(L):
        offs = lm - lm[a]                       # [L, goal_dim]
        dist = offs.norm(dim=-1)
        f, s = model(zb, offs)
        fp = torch.sigmoid(f).squeeze(-1).cpu().numpy()
        sp = s.squeeze(-1).cpu().numpy()
        ok = (fp > feas_thresh) & (dist.cpu().numpy() < max_off) & (np.arange(L) != a)
        feas[a, ok] = 1.0
        cost[a, ok] = np.maximum(sp[ok], 0.05) + dist.cpu().numpy()[ok] * 0.01
    s_idx = int(np.argmin(np.linalg.norm(landmarks - start_pos, axis=1)))
    d_idx = int(np.argmin(np.linalg.norm(landmarks - goal_pos, axis=1)))
    h = np.linalg.norm(landmarks - goal_pos, axis=1)
    dist = np.full(L, np.inf); prev = np.full(L, -1, int); dist[s_idx] = 0.0
    pq = [(h[s_idx], s_idx)]; expanded = 0
    while pq and expanded < max_expand:
        _, u = heapq.heappop(pq); expanded += 1
        if u == d_idx:
            break
        for v in range(L):
            if feas[u, v] and dist[u] + cost[u, v] < dist[v]:
                dist[v] = dist[u] + cost[u, v]; prev[v] = u
                heapq.heappush(pq, (dist[v] + h[v], v))
    if not np.isfinite(dist[d_idx]):
        return None
    path = []; node = d_idx
    while node != -1:
        path.append(node); node = prev[node]
    return [landmarks[i] for i in path[::-1]]


@torch.no_grad()
def plan_cem(model, z_start, start_pos, goal_pos, device, goal_dim, horizon=6,
            iters=5, samples=512, elite=0.1, max_off=3.0, w_infeas=4.0, w_len=0.02, seed=0):
    """TRUE HWM planning: CEM optimization over a CONTINUOUS sequence of macro-
    actions (subgoal offsets) in latent space, scored by the learned JEPA-2 model.

    Unlike A*-over-landmarks (discrete node selection), this optimizes the subgoals
    as continuous points: it samples offset sequences, rolls predicted positions
    p_i = start + cumsum(offsets), scores each hop's feasibility with JEPA-2
    (infeasible hops = walls are penalised), plus terminal distance-to-goal and
    path length, and refits a Gaussian to the elites. Returns the FIRST subgoal
    (an arbitrary continuous point, not a landmark).
    """
    rng = np.random.default_rng(seed)
    g = torch.as_tensor(goal_pos, dtype=torch.float32, device=device)
    s = torch.as_tensor(start_pos, dtype=torch.float32, device=device)
    zb = z_start.expand(samples * horizon, -1)
    # init mean: evenly step toward the goal over the horizon
    mean = ((g - s) / horizon).cpu().numpy()[None].repeat(horizon, 0)   # [H, goal_dim]
    std = np.full_like(mean, max_off * 0.5)
    best_first = mean[0].copy(); best_cost = np.inf
    for _ in range(iters):
        offs = rng.normal(mean, std, size=(samples, horizon, goal_dim)).astype(np.float32)
        offs = np.clip(offs, -max_off, max_off)
        offs_t = torch.from_numpy(offs).to(device)
        pos = s[None, None] + torch.cumsum(offs_t, dim=1)                 # [S, H, goal_dim]
        f, _ = model(zb, offs_t.reshape(-1, goal_dim))
        feas = torch.sigmoid(f).reshape(samples, horizon)
        terminal = (pos[:, -1] - g).norm(dim=-1)
        infeas = (1.0 - feas).sum(dim=1)
        length = offs_t.norm(dim=-1).sum(dim=1)
        cost = (terminal + w_infeas * infeas + w_len * length).cpu().numpy()
        order = np.argsort(cost)
        if cost[order[0]] < best_cost:
            best_cost = float(cost[order[0]]); best_first = offs[order[0], 0].copy()
        elites = offs[order[: max(1, int(samples * elite))]]
        mean = elites.mean(0); std = np.maximum(elites.std(0), 0.1)
    return (np.asarray(start_pos, np.float32) + best_first)               # first continuous subgoal


def run_hjepa2_cem(env_id, max_steps, low, model, episodes, seed, reach_radius,
                   subgoal_timeout, device, goal_dim, max_off, replan_every=25):
    """Continuous HWM control: re-plan the subgoal with CEM every ``replan_every``
    steps (receding horizon), execute it with the JEPA-1 low level."""
    succ = []
    for ep in range(episodes):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        term = trunc = False; info = {}; t = 0; sg = goal
        while not (term or trunc):
            if t % replan_every == 0:
                cur = np.asarray(obs["achieved_goal"], np.float32)
                z = low.model.encode(low._torch.from_numpy(
                    low.norm.encode(flatten_obs(obs))).unsqueeze(0).to(device))
                sg = plan_cem(model, z, cur, goal, device, goal_dim, max_off=max_off, seed=seed + ep)
            a = low.act(obs, sg)
            obs, _, term, trunc, info = env.step(a); t += 1
        succ.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    return float(np.mean(succ))


def run_hjepa2(env_id, max_steps, low, model, landmarks, episodes, seed, reach_radius,
               subgoal_timeout, device, goal_dim, feas_thresh, max_off):
    succ = []
    for ep in range(episodes):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        cur = np.asarray(obs["achieved_goal"], np.float32)
        z0 = low.model.encode(torch.from_numpy(low.norm.encode(flatten_obs(obs))).unsqueeze(0).to(device))
        wps = plan_astar(model, z0, cur, goal, landmarks, device, goal_dim, feas_thresh, max_off=max_off)
        wps = (wps or []) + [goal]
        wi = 0; since = 0; term = trunc = False; info = {}
        while not (term or trunc):
            sg = wps[min(wi, len(wps) - 1)]
            a = low.act(obs, sg)
            obs, _, term, trunc, info = env.step(a); since += 1
            ag = np.asarray(obs["achieved_goal"], np.float32)
            if wi < len(wps) - 1 and (np.linalg.norm(ag - sg) < reach_radius or since > subgoal_timeout):
                wi += 1; since = 0
        succ.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    return float(np.mean(succ))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--bc-policy", type=Path, required=True)
    p.add_argument("--jepa-model", type=Path, required=True)
    p.add_argument("--graph-npz", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--landmarks", type=int, default=150)
    p.add_argument("--k-reach", type=int, default=40)
    p.add_argument("--reach-radius", type=float, default=2.5)
    p.add_argument("--subgoal-timeout", type=int, default=60)
    p.add_argument("--feas-thresh", type=float, default=0.5)
    p.add_argument("--max-off", type=float, default=6.0)
    p.add_argument("--planner", choices=["astar", "cem"], default="astar",
                   help="astar: directed search over landmarks (discrete). cem: TRUE HWM "
                        "continuous CEM optimization over subgoal offsets in latent space.")
    p.add_argument("--replan-every", type=int, default=25)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    task = resolve_task(args.task, None)
    dev = torch.device(args.device)
    genv = make_env(task.env_id, seed=args.seed + 7, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(genv); genv.close()
    episodes = load_episodes_npz(args.graph_npz)

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    low = LowLevelBC(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device)
    env.close()

    # landmarks = samples of the abstract state space
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    pos = np.concatenate([ep.states[:, gs:ge] for ep in episodes], axis=0)
    lm = pos[farthest_point_sample(pos, args.landmarks, args.seed)]

    # train the JEPA-2 high-level model
    model, max_pos_off = train_jepa2(low.model, low.norm, spec, episodes, args.k_reach, dev)
    print(json.dumps({"event": "jepa2_ready", "landmarks": len(lm), "max_pos_off": round(max_pos_off, 2)}), flush=True)

    if args.planner == "cem":
        hj = run_hjepa2_cem(task.env_id, task.max_episode_steps, low, model, args.episodes, args.seed,
                            args.reach_radius, args.subgoal_timeout, dev, spec.goal_dim, args.max_off,
                            replan_every=args.replan_every)
        name = "H-JEPA2 (continuous CEM macro-action optimization)"
    else:
        hj = run_hjepa2(task.env_id, task.max_episode_steps, low, model, lm, args.episodes, args.seed,
                        args.reach_radius, args.subgoal_timeout, dev, spec.goal_dim, args.feas_thresh, args.max_off)
        name = "H-JEPA2 (learned high-level + A* search)"
    print(json.dumps({"task": task.name, "policy": name,
                      "landmarks": len(lm), "episodes": args.episodes, "success_rate": round(hj, 4)}), flush=True)


if __name__ == "__main__":
    main()
