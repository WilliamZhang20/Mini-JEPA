"""Offline IQL (Implicit Q-Learning, Kostrikov et al. 2021) on the frozen JEPA latent.

Behaviour cloning (and TD3+BC) fails on FrankaKitchen: cloning the demos compounds
error over the 280-step contact sequence and the policy flails. IQL abandons the BC
term entirely and learns purely from *value*:

  * a value net V(z) trained by EXPECTILE regression toward Q (tau>0.5 ->
    optimistic, approximates max over *in-distribution* actions without ever
    querying an OOD action — so the critic can't overestimate / diverge);
  * twin Q(z,a) trained with the TD target r + gamma * V(z')  (in-sample,
    no policy bootstrapping);
  * the policy is extracted by ADVANTAGE-WEIGHTED regression: weight =
    exp(beta * (Q - V)), so the *value function* picks which actions to follow
    and low-advantage demo actions get ~0 weight (this is what separates it from
    BC, which clones every action equally).

Runs entirely in the JEPA latent z = encode(obs); the actor is the same
GoalConditionedPolicy the eval loads, so it drops into the kitchen pipeline.
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

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP, GoalConditionedPolicy
from jepa_robotics.tasks import resolve_task
from scripts.train_offline_td3bc import build_transitions, build_transitions_nongoal


class Critic(nn.Module):
    """Twin Q over (latent, action), LayerNorm for offline stability."""

    def __init__(self, latent_dim, action_dim, hidden):
        super().__init__()
        self.q1 = MLP([latent_dim + action_dim, hidden, hidden, 1], layer_norm=True)
        self.q2 = MLP([latent_dim + action_dim, hidden, hidden, 1], layer_norm=True)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.q1(za), self.q2(za)


def expectile_loss(diff, tau):
    # asymmetric L2: weight tau above 0, (1-tau) below -> optimistic value for tau>0.5
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--steps", type=int, default=250000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005, help="target soft-update rate")
    p.add_argument("--expectile", type=float, default=0.7, help="IQL expectile tau (kitchen: 0.7)")
    p.add_argument("--beta", type=float, default=3.0, help="AWR inverse temperature")
    p.add_argument("--adv-clip", type=float, default=100.0, help="max advantage weight")
    p.add_argument("--her-frac", type=float, default=0.8)
    p.add_argument("--non-goal", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    latent_dim = int(cfg["latent_dim"])
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    compute_reward = getattr(env.unwrapped, "compute_reward", None)
    alow = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    ahigh = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    env.close()

    rng = np.random.default_rng(0)
    if args.non_goal:
        S, A, S2, R, D = build_transitions_nongoal(args.episodes_npz, args.max_episodes, norm)
    else:
        eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
        S, A, S2, R, D = build_transitions(eps, spec, norm, compute_reward, args.her_frac, rng)
    N = len(S)
    print(json.dumps({"event": "offline_data", "transitions": N, "latent_dim": latent_dim,
                      "action_dim": spec.action_dim}), flush=True)

    @torch.no_grad()
    def enc(arr):
        out = []
        for i in range(0, len(arr), 16384):
            out.append(wm.encode(torch.from_numpy(arr[i:i + 16384]).to(dev)))
        return torch.cat(out, 0)

    Z = enc(S); Z2 = enc(S2)
    # actions stored in env range -> map to the policy's tanh [-1,1] space for AWR target
    At = torch.from_numpy(A).to(dev)
    a_unit = (2.0 * (At - alow) / (ahigh - alow) - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    Rt = torch.from_numpy(R).to(dev); Dt = torch.from_numpy(D).to(dev)

    actor = GoalConditionedPolicy(latent_dim=latent_dim, action_dim=spec.action_dim, hidden_dim=args.hidden).to(dev)
    critic = Critic(latent_dim, spec.action_dim, args.hidden).to(dev)
    critic_t = Critic(latent_dim, spec.action_dim, args.hidden).to(dev)
    critic_t.load_state_dict(critic.state_dict())
    value = MLP([latent_dim, args.hidden, args.hidden, 1], layer_norm=True).to(dev)

    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    v_opt = torch.optim.Adam(value.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        z, au, z2, r, d = Z[idx], a_unit[idx], Z2[idx], Rt[idx].unsqueeze(-1), Dt[idx].unsqueeze(-1)

        # --- value: expectile regression toward in-sample Q (no OOD query) ---
        with torch.no_grad():
            q1t, q2t = critic_t(z, au)
            q_t = torch.min(q1t, q2t)
        v = value(z)
        v_loss = expectile_loss(q_t - v, args.expectile)
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        # --- Q: TD target uses V(z'), never a policy/OOD action ---
        with torch.no_grad():
            v2 = value(z2)
            target = r + args.gamma * (1 - d) * v2
        q1, q2 = critic(z, au)
        c_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()

        # --- policy: advantage-weighted regression (value decides, not cloning) ---
        with torch.no_grad():
            q1d, q2d = critic_t(z, au)
            adv = torch.min(q1d, q2d) - value(z)
            w = torch.exp(args.beta * adv).clamp(max=args.adv_clip).squeeze(-1)
        pi = actor(z)  # tanh [-1,1]
        a_loss = (w * (pi - au).pow(2).mean(dim=-1)).mean()
        a_opt.zero_grad(); a_loss.backward(); a_opt.step()

        with torch.no_grad():
            for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * ps)

        if step % 5000 == 0:
            print(json.dumps({"event": "iql", "step": step,
                              "v_loss": round(float(v_loss), 4), "q_loss": round(float(c_loss), 4),
                              "a_loss": round(float(a_loss), 4), "v_mean": round(float(v.mean()), 3),
                              "adv_mean": round(float(adv.mean()), 3), "w_max": round(float(w.max()), 1)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": actor.state_dict(),
                "config": {"latent_dim": latent_dim, "action_dim": spec.action_dim, "hidden_dim": args.hidden}},
               args.out)
    print(json.dumps({"event": "iql_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
