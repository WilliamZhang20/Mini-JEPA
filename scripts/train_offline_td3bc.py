"""Offline TD3+BC goal-conditioned controller on the frozen JEPA latent.

For AntMaze, online HER doesn't ignite (sparse 0/1 reward over 1000-step episodes)
and plain BC compounds error over the long horizon. TD3+BC (Fujimoto & Gu 2021)
is the canonical offline-RL fix: standard TD3 with a behaviour-cloning term added
to the actor loss, trained *purely offline* on the D4RL buffer (fast — no env
rollouts). We run it in the JEPA latent: the actor is the GoalConditionedPolicy on
encode(obs incl. desired_goal); a twin critic Q(z, a) is learned with HER-relabeled
goals so the reward signal is dense. Output is a much stronger nearby-goal walker
to use as the H-JEPA low level.
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


class Critic(nn.Module):
    """Twin Q over (latent, action)."""

    def __init__(self, latent_dim, action_dim, hidden):
        super().__init__()
        # LayerNorm critics prevent catastrophic Q overestimation on OOD actions
        # in offline RL (Ball et al. 2023) — essential for stability on kitchen.
        self.q1 = MLP([latent_dim + action_dim, hidden, hidden, 1], layer_norm=True)
        self.q2 = MLP([latent_dim + action_dim, hidden, hidden, 1], layer_norm=True)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.q1(za), self.q2(za)


def build_transitions_nongoal(npz_path, max_episodes, normalizer):
    """Non-goal offline transitions (e.g. FrankaKitchen): flat states + actions +
    the dataset's own dense reward (number of subtasks completed). No HER."""
    data = np.load(npz_path, allow_pickle=True)
    states, actions, rewards = data["states"], data["actions"], data["rewards"]
    S_list, A_list, S2_list, R_list, D_list = [], [], [], [], []
    for i in range(min(len(states), max_episodes)):
        S = np.asarray(states[i], np.float32); A = np.asarray(actions[i], np.float32)
        R = np.asarray(rewards[i], np.float32)
        # Sparsify a "stay-completed" cumulative reward into completion EVENTS
        # (delta>0), so returns are bounded (~#tasks) and the critic doesn't
        # diverge — kitchen's dense reward otherwise blows Q up.
        cum = np.maximum.accumulate(R)
        Rd = np.diff(np.concatenate([[0.0], cum]))
        Rd = np.clip(Rd, 0.0, None)
        T = min(len(A), len(Rd))
        for t in range(T):
            S_list.append(S[t]); A_list.append(A[t]); S2_list.append(S[t + 1])
            R_list.append(Rd[t]); D_list.append(1.0 if t == T - 1 else 0.0)
    Sn = normalizer.encode(np.asarray(S_list, np.float32))
    S2n = normalizer.encode(np.asarray(S2_list, np.float32))
    return (Sn, np.asarray(A_list, np.float32), S2n,
            np.asarray(R_list, np.float32), np.asarray(D_list, np.float32))


def build_transitions(episodes, spec, normalizer, compute_reward, her_frac, rng):
    """Flatten episodes into (s, a, s', r, done, achieved') arrays with HER relabeling.

    Returns raw flattened states (we re-encode through the JEPA encoder in batches
    at train time so the encoder stays the single source of the latent)."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    S_list, A_list, S2_list, R_list, D_list = [], [], [], [], []
    for ep in episodes:
        S, A = ep.states, ep.actions
        T = len(A)
        for t in range(T):
            s = S[t].copy(); s2 = S[t + 1].copy()
            # HER: relabel desired_goal to a future achieved_goal (in both s and s')
            if rng.random() < her_frac:
                tf = int(rng.integers(t + 1, len(S)))
                g = S[tf, gs:ge]
                s[ds:de] = g; s2[ds:de] = g
            r = float(compute_reward(s2[gs:ge], s2[ds:de], {}))
            done = 1.0 if r >= 1.0 else (1.0 if t == T - 1 else 0.0)
            S_list.append(s); A_list.append(A[t]); S2_list.append(s2)
            R_list.append(r); D_list.append(done)
    S = normalizer.encode(np.asarray(S_list, dtype=np.float32))
    S2 = normalizer.encode(np.asarray(S2_list, dtype=np.float32))
    return (S, np.asarray(A_list, np.float32), S2,
            np.asarray(R_list, np.float32), np.asarray(D_list, np.float32))


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
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=2.5, help="TD3+BC: actor RL/BC balance.")
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--her-frac", type=float, default=0.8)
    p.add_argument("--raw", action="store_true",
                   help="Act on the raw normalized observation instead of the JEPA latent. The raw obs "
                        "carries the exact (maze) position the critic needs; the predictive latent smears "
                        "it, which caps the AntMaze walker.")
    p.add_argument("--non-goal", action="store_true",
                   help="Non-goal env (FrankaKitchen): use the stored dataset reward, no HER/compute_reward.")
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

    # encode all states once (frozen encoder) -> latent tensors on GPU
    @torch.no_grad()
    def enc(arr):
        out = []
        for i in range(0, len(arr), 16384):
            out.append(wm.encode(torch.from_numpy(arr[i:i + 16384]).to(dev)))
        return torch.cat(out, 0)

    if args.raw:
        # Use the normalized observation directly as the representation.
        latent_dim = S.shape[1]
        Z = torch.from_numpy(S).to(dev); Z2 = torch.from_numpy(S2).to(dev)
    else:
        Z = enc(S); Z2 = enc(S2)
    At = torch.from_numpy(A).to(dev); Rt = torch.from_numpy(R).to(dev); Dt = torch.from_numpy(D).to(dev)

    actor = GoalConditionedPolicy(latent_dim=latent_dim, action_dim=spec.action_dim, hidden_dim=args.hidden).to(dev)
    actor_t = GoalConditionedPolicy(latent_dim=latent_dim, action_dim=spec.action_dim, hidden_dim=args.hidden).to(dev)
    actor_t.load_state_dict(actor.state_dict())
    critic = Critic(latent_dim, spec.action_dim, args.hidden).to(dev)
    critic_t = Critic(latent_dim, spec.action_dim, args.hidden).to(dev)
    critic_t.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    def scale(a):  # tanh policy -> action range
        return alow + (a + 1.0) * 0.5 * (ahigh - alow)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        z, a, z2, r, d = Z[idx], At[idx], Z2[idx], Rt[idx], Dt[idx]
        with torch.no_grad():
            noise = (torch.randn_like(a) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
            a2 = (scale(actor_t(z2)) + noise).clamp(alow, ahigh)
            q1t, q2t = critic_t(z2, a2)
            target = r.unsqueeze(-1) + args.gamma * (1 - d.unsqueeze(-1)) * torch.min(q1t, q2t)
        q1, q2 = critic(z, scale_noop(a))
        c_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()

        a_loss_val = float("nan")
        if step % args.policy_delay == 0:
            pi = scale(actor(z))
            q1pi, _ = critic(z, pi)
            lam = args.alpha / (q1pi.abs().mean().detach() + 1e-6)
            bc = nn.functional.mse_loss(pi, a)
            a_loss = -lam * q1pi.mean() + bc
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            a_loss_val = float(a_loss)
            with torch.no_grad():
                for pt, ps in zip(actor_t.parameters(), actor.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * ps)
                for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * ps)
        if step % 5000 == 0:
            print(json.dumps({"event": "td3bc", "step": step, "critic_loss": round(float(c_loss), 4),
                              "actor_loss": round(a_loss_val, 4), "q_mean": round(float(q1.mean()), 3)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": actor.state_dict(),
                "config": {"latent_dim": latent_dim, "action_dim": spec.action_dim,
                           "hidden_dim": args.hidden, "raw": bool(args.raw)}},
               args.out)
    print(json.dumps({"event": "td3bc_saved", "path": str(args.out)}), flush=True)


def scale_noop(a):
    return a


if __name__ == "__main__":
    main()
