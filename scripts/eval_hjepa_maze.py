"""Hierarchical JEPA (H-JEPA) for long-horizon maze navigation.

Flat goal-conditioned control fails on long mazes: the straight-line-to-goal
heuristic walks into walls and sparse long-horizon credit assignment is hard.
H-JEPA splits control:
  * LOW level  - the existing goal-conditioned HER policy on the JEPA latent,
    which reliably reaches *nearby* subgoals.
  * HIGH level - a subgoal graph LEARNED from the agent's own exploration: sample
    landmarks in achieved_goal space, connect two landmarks if the agent actually
    got from one to the other within k steps (empirical reachability -> edges only
    exist where trajectories went, i.e. *around* walls), then Dijkstra a subgoal
    path to the goal. Execute one subgoal at a time with the low level, advance on
    reach, re-plan as needed.

This uses the JEPA predictor's regime where it's reliable (smooth macro-steps),
unlike contact-rich primitive planning (which failed in P1). Reports hierarchical
vs flat (low level driven straight at the final goal).
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import make_env, flatten_obs, obs_spec_from_env
from jepa_robotics.data import collect_episodes
from jepa_robotics.tasks import resolve_task


# ----------------------------- subgoal graph --------------------------------
def farthest_point_sample(points, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(points)
    if n <= k:
        return np.arange(n)
    idx = [int(rng.integers(n))]
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(points - points[j], axis=1))
    return np.array(idx)


def build_subgoal_graph(episodes, spec, n_landmarks, k_reach, seed=0):
    """Landmarks in achieved_goal (xy) space + empirical k-step reachability edges."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    pos = np.concatenate([ep.states[:, gs:ge] for ep in episodes], axis=0)
    lm_idx = farthest_point_sample(pos, n_landmarks, seed)
    landmarks = pos[lm_idx]  # [L, goal_dim]

    def nearest(p):
        return int(np.argmin(np.linalg.norm(landmarks - p, axis=1)))

    L = len(landmarks)
    best = np.full((L, L), np.inf)
    for ep in episodes:
        traj = ep.states[:, gs:ge]
        lm_seq = [nearest(traj[t]) for t in range(len(traj))]
        for t in range(len(traj)):
            a = lm_seq[t]
            for dt in range(1, k_reach + 1):
                if t + dt >= len(traj):
                    break
                b = lm_seq[t + dt]
                if a != b and dt < best[a, b]:
                    best[a, b] = dt
    return landmarks, best


def dijkstra_path(adj, src, dst):
    L = len(adj)
    dist = [np.inf] * L
    prev = [-1] * L
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v in range(L):
            w = adj[u][v]
            if np.isfinite(w) and d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    if not np.isfinite(dist[dst]):
        return None
    path = []
    node = dst
    while node != -1:
        path.append(node)
        node = prev[node]
    return path[::-1]


# ----------------------------- low-level wrapper ----------------------------
class LowLevel:
    """Goal-conditioned SB3 HER policy: pursues whatever we put in desired_goal."""

    def __init__(self, sb3_path, env, device="cpu"):
        from sb3_contrib import TQC
        self.model = TQC.load(str(sb3_path), env=env, device=device)

    def act(self, obs, subgoal):
        o = {k: np.array(v, copy=True) for k, v in obs.items()}
        o["desired_goal"] = np.asarray(subgoal, dtype=np.float32)
        a, _ = self.model.predict(o, deterministic=True)
        return a


class LowLevelBC:
    """Goal-conditioned BC policy on the JEPA latent (for envs with no scripted/SB3
    HER low level, e.g. AntMaze where the low level is BC'd from offline data)."""

    def __init__(self, wm_path, bc_path, low, high, device="cpu"):
        import torch
        from jepa_robotics.evaluate import load_jepa_artifact, load_policy_artifact
        self._torch = torch
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        self.policy, pcfg = load_policy_artifact(Path(bc_path), self.dev)
        # A raw policy (train_gcrl_raw.py) acts on the normalized observation
        # directly; a latent policy acts on the JEPA encoding of it.
        self.raw = bool(pcfg.get("raw", False))
        self.model.eval(); self.policy.eval()
        self.low, self.high = low, high

    def act(self, obs, subgoal):
        o = {k: np.array(v, copy=True) for k, v in obs.items()}
        o["desired_goal"] = np.asarray(subgoal, dtype=np.float32)
        s = self._torch.from_numpy(self.norm.encode(flatten_obs(o))).unsqueeze(0).to(self.dev)
        with self._torch.no_grad():
            z = s if self.raw else self.model.encode(s)
            a = self.policy(z)[0].cpu().numpy()
        return np.clip(a, self.low, self.high).astype(np.float32)


class LowLevelChunk:
    """Action-chunked goal-conditioned walker: predicts a chunk of H actions and
    executes it (receding horizon, re-predicting on chunk exhaustion or subgoal
    change). The coherent chunk keeps the ant's gait stable so it stops falling."""

    def __init__(self, wm_path, bc_path, low, high, device="cpu", replan=None):
        import torch
        from jepa_robotics.evaluate import load_jepa_artifact
        from scripts.train_chunked_walker import ChunkPolicy
        self._torch = torch
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        art = torch.load(Path(bc_path), map_location=self.dev, weights_only=False)
        cfg = art["config"]
        self.raw = bool(cfg.get("raw", False))
        self.chunk = int(cfg["chunk"])
        self.policy = ChunkPolicy(int(cfg["latent_dim"]), int(cfg["action_dim"]),
                                  self.chunk, int(cfg["hidden_dim"])).to(self.dev)
        self.policy.load_state_dict(art["policy"]); self.policy.eval()
        self.model.eval()
        self.low, self.high = low, high
        self.replan = replan or self.chunk
        self._buf = []
        self._last_sg = None

    def act(self, obs, subgoal):
        sg = np.asarray(subgoal, dtype=np.float32)
        if (not self._buf) or self._last_sg is None or np.linalg.norm(sg - self._last_sg) > 1e-6:
            o = {k: np.array(v, copy=True) for k, v in obs.items()}
            o["desired_goal"] = sg
            s = self._torch.from_numpy(self.norm.encode(flatten_obs(o))).unsqueeze(0).to(self.dev)
            with self._torch.no_grad():
                z = s if self.raw else self.model.encode(s)
                chunk = self.policy(z)[0].cpu().numpy()
            self._buf = list(chunk[: self.replan])
            self._last_sg = sg
        a = self._buf.pop(0)
        return np.clip(a, self.low, self.high).astype(np.float32)


class LowLevelFlow:
    """Action-chunked rectified-flow walker: samples a coherent H-step gait chunk
    from the conditional flow (multimodality preserved, no BC mush) and executes
    it receding-horizon. Conditions on the JEPA latent (+ raw obs if concat_raw)."""

    def __init__(self, wm_path, bc_path, low, high, device="cpu", replan=None, flow_steps=10):
        import torch
        from jepa_robotics.evaluate import load_jepa_artifact
        from scripts.train_flow_walker import FlowNet
        self._torch = torch
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        art = torch.load(Path(bc_path), map_location=self.dev, weights_only=False)
        cfg = art["config"]
        self.chunk = int(cfg["chunk"]); self.action_dim = int(cfg["action_dim"])
        self.concat_raw = bool(cfg["concat_raw"]); self.chunk_dim = int(cfg["chunk_dim"])
        self.net = FlowNet(int(cfg["chunk_dim"]), int(cfg["cond_dim"]), int(cfg["hidden"])).to(self.dev)
        self.net.load_state_dict(art["flow"]); self.net.eval(); self.model.eval()
        self.low, self.high = low, high
        self.replan = replan or self.chunk
        self.flow_steps = flow_steps
        self._buf = []
        self._last_sg = None

    def act(self, obs, subgoal):
        sg = np.asarray(subgoal, dtype=np.float32)
        if (not self._buf) or self._last_sg is None or np.linalg.norm(sg - self._last_sg) > 1e-6:
            o = {k: np.array(v, copy=True) for k, v in obs.items()}
            o["desired_goal"] = sg
            s = self._torch.from_numpy(self.norm.encode(flatten_obs(o))).unsqueeze(0).to(self.dev)
            with self._torch.no_grad():
                z = self.model.encode(s)
                c = self._torch.cat([z, s], dim=1) if self.concat_raw else z
                x = self.net.sample(c, self.chunk_dim, self.flow_steps)[0].cpu().numpy()
            chunk = x.reshape(self.chunk, self.action_dim)
            self._buf = list(chunk[: self.replan])
            self._last_sg = sg
        a = self._buf.pop(0)
        return np.clip(a, self.low, self.high).astype(np.float32)


class LowLevelInverse:
    """Self-supervised inverse chunk low level.

    The high level sets ``desired_goal`` to the current subgoal. The inverse
    prior receives ``(z_t, z_subgoal)`` and proposes the first action of an
    H-step chunk. No action labels are copied at runtime.
    """

    def __init__(self, wm_path, inverse_path, low, high, device="cpu", target_horizon=None):
        import torch
        from jepa_robotics.envs import flatten_obs, goal_state_from_state
        from jepa_robotics.evaluate import load_jepa_artifact
        from jepa_robotics.algos.priors import InversePrior
        self._torch = torch
        self._flatten_obs = flatten_obs
        self._goal_state_from_state = goal_state_from_state
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        art = torch.load(Path(inverse_path), map_location=self.dev, weights_only=False)
        self.ckpt = art
        self.prior = InversePrior(
            int(art["cond_dim"]),
            int(art["chunk_dim"]),
            int(art["hidden"]),
            int(art["n_blocks"]),
        ).to(self.dev)
        self.prior.load_state_dict(art["state_dict"])
        self.prior.eval(); self.model.eval()
        self.low, self.high = low, high
        self.target_horizon = target_horizon or int(art["H"])

    def act(self, obs, subgoal):
        o = {k: np.array(v, copy=True) for k, v in obs.items()}
        o["desired_goal"] = np.asarray(subgoal, dtype=np.float32)
        raw = self._flatten_obs(o)
        target = self._goal_state_from_state(raw, self.spec)
        s = self._torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev)
        tgt = self._torch.from_numpy(self.norm.encode(target)).unsqueeze(0).to(self.dev)
        with self._torch.no_grad():
            z = self.model.encode(s)
            z_goal = self.model.encode_target(tgt)
            horizons = list(self.ckpt.get("future_horizons", [int(self.ckpt["H"])]))
            h = float(self.target_horizon) / float(max(horizons))
            h_token = self._torch.tensor([[h]], dtype=z.dtype, device=self.dev)
            cond = self._torch.cat([z, z_goal, h_token], dim=-1)
            chunk = self.prior(cond).view(int(self.ckpt["H"]), int(self.ckpt["action_dim"]))
            a = chunk[0].cpu().numpy()
        return np.clip(a, self.low, self.high).astype(np.float32)


# ----------------------------- rollouts -------------------------------------
def run_flat(env_id, max_steps, low, episodes, seed):
    succ = []
    for ep in range(episodes):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        term = trunc = False; info = {}
        while not (term or trunc):
            a = low.act(obs, obs["desired_goal"])  # straight at the real goal
            obs, _, term, trunc, info = env.step(a)
        succ.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    return float(np.mean(succ))


def run_hjepa(env_id, max_steps, low, landmarks, adj, episodes, seed, reach_radius, subgoal_timeout):
    succ = []
    for ep in range(episodes):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        # plan subgoal path from nearest landmark-to-start to nearest-to-goal
        cur = np.asarray(obs["achieved_goal"], dtype=np.float32)
        s = int(np.argmin(np.linalg.norm(landmarks - cur, axis=1)))
        d = int(np.argmin(np.linalg.norm(landmarks - goal, axis=1)))
        path = dijkstra_path(adj, s, d)
        # subgoal queue = landmark waypoints then the true goal
        waypoints = [landmarks[i] for i in path] if path else []
        waypoints.append(goal)
        wi = 0; since = 0
        term = trunc = False; info = {}
        while not (term or trunc):
            sg = waypoints[min(wi, len(waypoints) - 1)]
            a = low.act(obs, sg)
            obs, _, term, trunc, info = env.step(a)
            since += 1
            ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
            # advance subgoal on reach or timeout
            if wi < len(waypoints) - 1 and (np.linalg.norm(ag - sg) < reach_radius or since > subgoal_timeout):
                wi += 1; since = 0
        succ.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    return float(np.mean(succ))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="point_umaze")
    p.add_argument("--low-type", default="sb3", choices=["sb3", "bc", "chunk", "flow", "inverse"])
    p.add_argument("--low-policy", type=Path, default=None, help="SB3 HER low-level .zip (low-type sb3)")
    p.add_argument("--bc-policy", type=Path, default=None, help="BC GoalConditionedPolicy .pt (low-type bc)")
    p.add_argument("--inverse-policy", type=Path, default=None, help="InversePrior .pt (low-type inverse)")
    p.add_argument("--jepa-model", type=Path, default=None, help="JEPA WM (.pt); fallback for sb3, encoder for bc")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--graph-steps", type=int, default=120000, help="exploration steps for the subgoal graph")
    p.add_argument("--graph-npz", type=Path, default=None,
                   help="Build the subgoal graph from these demo trajectories instead of fresh "
                        "exploration (needed when random exploration can't traverse the maze, e.g. AntMaze).")
    p.add_argument("--landmarks", type=int, default=60)
    p.add_argument("--k-reach", type=int, default=20)
    p.add_argument("--reach-radius", type=float, default=0.6)
    p.add_argument("--subgoal-timeout", type=int, default=40)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.jepa_model is not None:
        os.environ.setdefault("JEPA_MODEL_FALLBACK", str(args.jepa_model))
    task = resolve_task(args.task, None)

    # 1) data for the subgoal graph: demo trajectories (if given) or fresh exploration
    genv = make_env(task.env_id, seed=args.seed + 7, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(genv)
    if args.graph_npz is not None:
        from jepa_robotics.data import load_episodes_npz
        episodes = load_episodes_npz(args.graph_npz)
    else:
        episodes, _ = collect_episodes(genv, num_steps=args.graph_steps, seed=args.seed + 7,
                                       scripted_fraction=0.5, controller_gain=5.0, action_noise=0.3,
                                       controller=task.controller, log_every=0)
    genv.close()
    landmarks, adj = build_subgoal_graph(episodes, spec, args.landmarks, args.k_reach, seed=args.seed)
    n_edges = int(np.isfinite(adj).sum())
    print(f'{{"event": "graph", "landmarks": {len(landmarks)}, "edges": {n_edges}}}', flush=True)

    # 2) low level + head-to-head
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    if args.low_type == "bc":
        low = LowLevelBC(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high,
                         device=args.device)
    elif args.low_type == "chunk":
        low = LowLevelChunk(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high,
                            device=args.device)
    elif args.low_type == "flow":
        low = LowLevelFlow(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high,
                           device=args.device)
    elif args.low_type == "inverse":
        low = LowLevelInverse(args.jepa_model, args.inverse_policy, env.action_space.low, env.action_space.high,
                              device=args.device)
    else:
        low = LowLevel(args.low_policy, env, device=args.device)
    env.close()

    flat = run_flat(task.env_id, task.max_episode_steps, low, args.episodes, args.seed)
    print(f'{{"task": "{task.name}", "policy": "FLAT (low-level -> goal)", "episodes": {args.episodes}, "success_rate": {flat:.4f}}}', flush=True)
    hj = run_hjepa(task.env_id, task.max_episode_steps, low, landmarks, adj, args.episodes, args.seed,
                   args.reach_radius, args.subgoal_timeout)
    print(f'{{"task": "{task.name}", "policy": "H-JEPA (subgoal graph)", "landmarks": {len(landmarks)}, "episodes": {args.episodes}, "success_rate": {hj:.4f}}}', flush=True)


if __name__ == "__main__":
    main()
