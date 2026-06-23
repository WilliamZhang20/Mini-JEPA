"""Goal-conditioned IQL on RAW (normalized) observations — the canonical AntMaze
offline-RL recipe, used as a robust low-level walker for H-JEPA.

Why raw, not the JEPA latent: the value function needs to separate maze positions
to learn a position-dependent V; the predictive JEPA latent smears that out, so
latent IQL's critic collapsed (V ~ const, advantages ~0 -> degenerates to BC).
The raw observation carries the exact ant pose, so V learns properly and the
advantage-weighted policy becomes a reliable goal reacher.

IQL (Kostrikov et al. 2021), AntMaze settings (expectile 0.9, AWR beta 10):
  * V(s)      expectile regression toward min twin-Q  (in-sample max, no OOD query)
  * Q(s,a)    TD target r + gamma (1-done) V(s')        (in-sample bootstrap)
  * policy    advantage-weighted regression: w=exp(beta(Q-V)), weighted BC

Goal-conditioning + HER reuse ``build_transitions`` from train_offline_td3bc.
Output is a GoalConditionedPolicy that maps normalized [obs|achieved|desired] ->
action directly (config flag ``raw: true``), executed by eval_hjepa_maze's raw
low level.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.data import Normalizer, load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP, GoalConditionedPolicy
from jepa_robotics.tasks import resolve_task
from scripts.train_offline_td3bc import build_transitions


class VNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = MLP([in_dim, hidden, hidden, 1], layer_norm=True)

    def forward(self, s):
        return self.net(s)


class QNet(nn.Module):
    def __init__(self, in_dim, action_dim, hidden):
        super().__init__()
        self.q1 = MLP([in_dim + action_dim, hidden, hidden, 1], layer_norm=True)
        self.q2 = MLP([in_dim + action_dim, hidden, hidden, 1], layer_norm=True)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


def expectile_loss(diff, tau):
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True, help="JEPA WM (for its normalizer + spec)")
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.005, help="target soft-update")
    p.add_argument("--expectile", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--adv-clip", type=float, default=100.0)
    p.add_argument("--her-frac", type=float, default=0.8)
    p.add_argument("--reward-shift", type=float, default=-1.0,
                   help="add to reward (antmaze IQL uses r-1 so non-goal steps are penalized)")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    _, norm, _, _ = load_jepa_artifact(args.model_path, dev)  # reuse the WM's normalizer
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    compute_reward = getattr(env.unwrapped, "compute_reward", None)
    alow = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    ahigh = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    env.close()

    rng = np.random.default_rng(0)
    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    S, A, S2, R, D = build_transitions(eps, spec, norm, compute_reward, args.her_frac, rng)
    in_dim = S.shape[1]
    print(json.dumps({"event": "gcrl_raw_data", "transitions": len(S), "in_dim": in_dim,
                      "action_dim": spec.action_dim, "reward_mean": float(R.mean())}), flush=True)

    St = torch.from_numpy(S).to(dev); At = torch.from_numpy(A).to(dev)
    S2t = torch.from_numpy(S2).to(dev)
    Rt = torch.from_numpy(R).to(dev) + args.reward_shift
    Dt = torch.from_numpy(D).to(dev)
    N = len(St)

    actor = GoalConditionedPolicy(latent_dim=in_dim, action_dim=spec.action_dim, hidden_dim=args.hidden).to(dev)
    qnet = QNet(in_dim, spec.action_dim, args.hidden).to(dev)
    qtarget = QNet(in_dim, spec.action_dim, args.hidden).to(dev)
    qtarget.load_state_dict(qnet.state_dict())
    vnet = VNet(in_dim, args.hidden).to(dev)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q_opt = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    v_opt = torch.optim.Adam(vnet.parameters(), lr=args.lr)

    def scale(a):  # tanh policy [-1,1] -> action range
        return alow + (a + 1.0) * 0.5 * (ahigh - alow)

    def unscale(a):  # data action range -> [-1,1] for BC target
        return (a - alow) / (ahigh - alow) * 2.0 - 1.0

    a_unit = unscale(At)  # BC targets in tanh space

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        s, a, s2, r, d, au = St[idx], At[idx], S2t[idx], Rt[idx], Dt[idx], a_unit[idx]

        # V: expectile regression toward min twin-Q(s,a)  (Q detached)
        with torch.no_grad():
            q1t, q2t = qtarget(s, a)
            q_sa = torch.min(q1t, q2t)
        v = vnet(s)
        v_loss = expectile_loss(q_sa - v, args.expectile)
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        # Q: TD target r + gamma (1-done) V(s')
        with torch.no_grad():
            v_s2 = vnet(s2)
            target = r.unsqueeze(-1) + args.gamma * (1 - d.unsqueeze(-1)) * v_s2
        q1, q2 = qnet(s, a)
        q_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        q_opt.zero_grad(); q_loss.backward(); q_opt.step()

        # policy: advantage-weighted regression (weighted BC in tanh space)
        with torch.no_grad():
            q1d, q2d = qnet(s, a)
            adv = torch.min(q1d, q2d) - vnet(s)
            w = torch.clamp(torch.exp(args.beta * adv), max=args.adv_clip).squeeze(-1)
        pi = actor(s)
        a_loss = (w * (pi - au).pow(2).mean(dim=-1)).mean()
        a_opt.zero_grad(); a_loss.backward(); a_opt.step()

        with torch.no_grad():
            for pt, ps in zip(qtarget.parameters(), qnet.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * ps)

        if step % args.log_every == 0:
            print(json.dumps({"event": "iql_raw", "step": step,
                              "v_loss": round(float(v_loss), 4), "q_loss": round(float(q_loss), 4),
                              "a_loss": round(float(a_loss), 4), "v_mean": round(float(v.mean()), 3),
                              "adv_mean": round(float(adv.mean()), 4), "w_max": round(float(w.max()), 2)}),
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": actor.state_dict(),
                "config": {"latent_dim": in_dim, "action_dim": spec.action_dim,
                           "hidden_dim": args.hidden, "raw": True}},
               args.out)
    print(json.dumps({"event": "gcrl_raw_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
