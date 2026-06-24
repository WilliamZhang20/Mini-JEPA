"""HIQL (Hierarchical Implicit Q-Learning, Park et al. 2023) for AntMaze — the
actual SOTA offline recipe, built on the JEPA representation.

ONE goal-conditioned value V(s,g) (IQL expectile regression; learns clean
position-dependent distances on raw obs, where the predictive latent alone
collapses the critic), and TWO advantage-weighted policies extracted from it:

  * high level pi_h(s, g_final) -> subgoal OFFSET (a waypoint ~k steps ahead);
    weighted by the value improvement that subgoal buys toward the final goal.
    Operating on a coarse timescale is what makes the long-horizon advantage
    signal strong enough to learn — the thing flat IQL/TD3+BC can't do.
  * low level  pi_l(s, g_sub) -> action; standard IQL AWR reaching the subgoal.

Representation rep(s) = [normalized raw obs | JEPA latent]: the raw obs carries the
exact maze position the value needs; the JEPA latent is the learned feature that
helps the small policy MLP (raw-only policies failed here). Goal-conditioning uses
the 2-D achieved_goal position with hindsight relabeling.

Eval is self-contained and hierarchical: pi_h proposes a subgoal every k steps,
pi_l executes it — no subgoal graph / external planner.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task


class VNet(nn.Module):
    def __init__(self, rep_dim, goal_dim, hidden):
        super().__init__()
        self.net = MLP([rep_dim + goal_dim, hidden, hidden, 1], layer_norm=True)

    def forward(self, rep, g):
        return self.net(torch.cat([rep, g], dim=-1)).squeeze(-1)


class PiLow(nn.Module):
    def __init__(self, rep_dim, goal_dim, action_dim, hidden):
        super().__init__()
        self.net = MLP([rep_dim + goal_dim, hidden, hidden, action_dim], layer_norm=True)

    def forward(self, rep, g):
        return torch.tanh(self.net(torch.cat([rep, g], dim=-1)))


class PiHigh(nn.Module):
    """Outputs a subgoal OFFSET (Δposition) from the current achieved position."""

    def __init__(self, rep_dim, goal_dim, hidden):
        super().__init__()
        self.net = MLP([rep_dim + goal_dim, hidden, hidden, goal_dim], layer_norm=True)

    def forward(self, rep, g):
        return self.net(torch.cat([rep, g], dim=-1))


def expectile_loss(diff, tau):
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--beta-low", type=float, default=3.0)
    p.add_argument("--beta-high", type=float, default=3.0)
    p.add_argument("--adv-clip", type=float, default=10.0)
    p.add_argument("--subgoal-k", type=int, default=25)
    p.add_argument("--goal-thresh", type=float, default=0.5)
    p.add_argument("--p-curr", type=float, default=0.2, help="prob the (low) goal is the immediate next state")
    p.add_argument("--p-rand", type=float, default=0.3, help="prob the goal is a random state (cross-traj)")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    a_lo = env.action_space.low; a_hi = env.action_space.high
    env.close()

    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    # flat arrays + per-state episode-end index for hindsight sampling
    states, ach, ep_end_of, act = [], [], [], []
    for ep in eps:
        T = len(ep.actions)
        base = len(states)
        for t in range(T):
            states.append(ep.states[t]); ach.append(ep.states[t, gs:ge])
            act.append(ep.actions[t]); ep_end_of.append(base + T - 1)  # last transition-start index
    states = np.asarray(states, np.float32); ach = np.asarray(ach, np.float32)
    act = np.asarray(act, np.float32); ep_end = np.asarray(ep_end_of, np.int64)
    # next-state achieved (for reward): ach at index+1 (state after the action)
    nstate = np.concatenate([states[1:], states[-1:]], 0)
    nach = np.concatenate([ach[1:], ach[-1:]], 0)
    Ntot = len(states)
    print(json.dumps({"event": "hiql_data", "transitions": Ntot, "rep": "raw+latent"}), flush=True)

    Sn = norm.encode(states); Nn = norm.encode(nstate)
    with torch.no_grad():
        def rep_of(arr):
            reps = []
            for i in range(0, len(arr), 16384):
                a = torch.from_numpy(arr[i:i + 16384]).to(dev)
                reps.append(torch.cat([a, wm.encode(a)], dim=1))
            return torch.cat(reps, 0)
        REP = rep_of(Sn); REP2 = rep_of(Nn)
    rep_dim = REP.shape[1]
    ACH = torch.from_numpy(ach).to(dev); NACH = torch.from_numpy(nach).to(dev)
    ACT = torch.from_numpy(act).to(dev); EPEND = torch.from_numpy(ep_end).to(dev)
    IDX_ALL = torch.arange(Ntot, device=dev)

    V = VNet(rep_dim, spec.goal_dim, args.hidden).to(dev)
    Vt = VNet(rep_dim, spec.goal_dim, args.hidden).to(dev); Vt.load_state_dict(V.state_dict())
    pil = PiLow(rep_dim, spec.goal_dim, spec.action_dim, args.hidden).to(dev)
    pih = PiHigh(rep_dim, spec.goal_dim, args.hidden).to(dev)
    v_opt = torch.optim.Adam(V.parameters(), lr=args.lr)
    l_opt = torch.optim.Adam(pil.parameters(), lr=args.lr)
    h_opt = torch.optim.Adam(pih.parameters(), lr=args.lr)

    def sample_goals(idx):
        # future index within the same episode (>= idx+1), clamped
        end = EPEND[idx]
        u = torch.rand(len(idx), device=dev)
        fut = (idx + 1 + (torch.rand(len(idx), device=dev) * (end - idx).clamp(min=1)).long()).clamp(max=end)
        # immediate / random mixes for the LOW goal
        g = ACH[fut].clone()
        curr_mask = u < args.p_curr
        g[curr_mask] = NACH[idx][curr_mask]
        rand_mask = u > (1 - args.p_rand)
        g[rand_mask] = ACH[torch.randint(0, Ntot, (int(rand_mask.sum()),), device=dev)]
        return g

    @torch.no_grad()
    def evaluate(n_eval, seed=20000):
        succ = []
        for ep in range(n_eval):
            e = make_env(task.env_id, seed=seed + ep, max_episode_steps=task.max_episode_steps)
            obs, _ = e.reset(seed=seed + ep)
            goal = torch.as_tensor(obs["desired_goal"], dtype=torch.float32, device=dev).unsqueeze(0)
            term = trunc = False; info = {}; t = 0; sg = goal
            while not (term or trunc):
                sa = norm.encode(flatten_obs(obs))
                r = torch.from_numpy(sa).unsqueeze(0).to(dev)
                rep = torch.cat([r, wm.encode(r)], dim=1)
                cur = torch.as_tensor(obs["achieved_goal"], dtype=torch.float32, device=dev).unsqueeze(0)
                if t % args.subgoal_k == 0:
                    sg = cur + pih(rep, goal)            # high level proposes a waypoint
                a = pil(rep, sg)[0].cpu().numpy()
                a = np.clip(a_lo + (a + 1) * 0.5 * (a_hi - a_lo), a_lo, a_hi)
                obs, _, term, trunc, info = e.step(a.astype(np.float32)); t += 1
            succ.append(float(info.get("is_success", info.get("success", 0.0))))
            e.close()
        return float(np.mean(succ))

    def save_to(path):
        torch.save({"V": V.state_dict(), "pi_low": pil.state_dict(), "pi_high": pih.state_dict(),
                    "config": {"rep_dim": rep_dim, "goal_dim": spec.goal_dim, "action_dim": spec.action_dim,
                               "hidden": args.hidden, "subgoal_k": args.subgoal_k}}, path)

    best_sr = -1.0
    best_path = args.out.with_name(args.out.stem + "_best.pt")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, Ntot, (args.batch_size,), device=dev)
        g = sample_goals(idx)
        reached = (torch.linalg.norm(NACH[idx] - g, dim=-1) < args.goal_thresh).float()
        r = reached - 1.0  # 0 at goal, -1 otherwise
        # ---- V: expectile regression toward r + gamma (1-done) V_target(s', g) ----
        with torch.no_grad():
            tgt = r + args.gamma * (1 - reached) * Vt(REP2[idx], g)
        v = V(REP[idx], g)
        v_loss = expectile_loss(tgt - v, args.expectile)
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()
        with torch.no_grad():
            for pt, ps in zip(Vt.parameters(), V.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * ps)
        # ---- low-level AWR: clone action a, weight = exp(beta * advantage) ----
        with torch.no_grad():
            adv_l = (r + args.gamma * (1 - reached) * V(REP2[idx], g) - V(REP[idx], g))
            w_l = torch.clamp(torch.exp(args.beta_low * adv_l), max=args.adv_clip)
        pred_a = pil(REP[idx], g)
        a_unit = (ACT[idx] - torch.as_tensor(a_lo, device=dev)) / torch.as_tensor(a_hi - a_lo, device=dev) * 2 - 1
        l_loss = (w_l * (pred_a - a_unit).pow(2).mean(-1)).mean()
        l_opt.zero_grad(); l_loss.backward(); l_opt.step()
        # ---- high-level AWR: clone subgoal offset (ach_{t+k}-ach_t) toward a far goal ----
        sub_idx = torch.minimum(idx + args.subgoal_k, EPEND[idx])
        gf = sample_goals(idx)  # far goal (reuse hindsight sampler)
        with torch.no_grad():
            adv_h = V(REP[sub_idx], gf) - V(REP[idx], gf)
            w_h = torch.clamp(torch.exp(args.beta_high * adv_h), max=args.adv_clip)
        pred_off = pih(REP[idx], gf)
        target_off = ACH[sub_idx] - ACH[idx]
        h_loss = (w_h * (pred_off - target_off).pow(2).mean(-1)).mean()
        h_opt.zero_grad(); h_loss.backward(); h_opt.step()

        if step % 10000 == 0:
            print(json.dumps({"event": "hiql", "step": step, "v_loss": round(float(v_loss), 4),
                              "l_loss": round(float(l_loss), 4), "h_loss": round(float(h_loss), 4),
                              "v_mean": round(float(v.mean()), 2), "adv_l": round(float(adv_l.mean()), 3),
                              "sps": round(step / (time.time() - t0), 1)}), flush=True)
        if step % args.eval_every == 0:
            sr = evaluate(args.eval_episodes)
            if sr > best_sr:  # keep the best checkpoint (the success curve is non-monotone)
                best_sr = sr; args.out.parent.mkdir(parents=True, exist_ok=True); save_to(best_path)
            print(json.dumps({"event": "hiql_eval", "step": step, "success_rate": round(sr, 4),
                              "best": round(best_sr, 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"V": V.state_dict(), "pi_low": pil.state_dict(), "pi_high": pih.state_dict(),
                "config": {"rep_dim": rep_dim, "goal_dim": spec.goal_dim, "action_dim": spec.action_dim,
                           "hidden": args.hidden, "subgoal_k": args.subgoal_k}},
               args.out)
    sr = evaluate(args.eval_episodes)
    print(json.dumps({"event": "hiql_final", "path": str(args.out), "success_rate": round(sr, 4)}), flush=True)


if __name__ == "__main__":
    main()
