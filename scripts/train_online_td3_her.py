"""Offline->online TD3+HER fine-tuning of the AntMaze walker on the JEPA latent.

Every offline method caps at ~0.27 because the demos never show the ant FALLING
and recovering, so the learned gait is brittle. Online interaction fixes exactly
that: the agent experiences falls and learns to stay up. Prior from-scratch online
RL on antmaze never ignited (sparse reward, hard exploration); the difference here
is a WARM START from the 0.27 offline TD3+BC actor (it already walks, so HER on its
own rollouts gets diverse achieved goals) plus the offline demos seeding the pool.

TD3 + hindsight relabeling on the latent z=encode(obs); the actor is the same
GoalConditionedPolicy the H-JEPA low level loads, so the improved walker drops in.
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

from jepa_robotics.data import Episode, load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import GoalConditionedPolicy
from jepa_robotics.tasks import resolve_task
from scripts.train_offline_td3bc import Critic


def sample_her(pool, spec, normalizer, compute_reward, n, her_frac, rng):
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    S, A, S2, R, D = [], [], [], [], []
    ne = len(pool)
    for _ in range(n):
        ep = pool[int(rng.integers(ne))]
        T = len(ep.actions)
        t = int(rng.integers(T))
        s = ep.states[t].copy(); s2 = ep.states[t + 1].copy()
        if rng.random() < her_frac:
            tf = int(rng.integers(t + 1, len(ep.states)))
            g = ep.states[tf, gs:ge]
            s[ds:de] = g; s2[ds:de] = g
        r = float(compute_reward(s2[gs:ge], s2[ds:de], {}))
        d = 1.0 if r >= 1.0 else 0.0
        S.append(s); A.append(ep.actions[t]); S2.append(s2); R.append(r); D.append(d)
    return (normalizer.encode(np.asarray(S, np.float32)), np.asarray(A, np.float32),
            normalizer.encode(np.asarray(S2, np.float32)),
            np.asarray(R, np.float32), np.asarray(D, np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--init-actor", type=Path, required=True, help="offline TD3+BC actor to warm-start")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offline-episodes", type=int, default=600)
    p.add_argument("--env-steps", type=int, default=300000)
    p.add_argument("--ep-len", type=int, default=120, help="short online episodes for local goal-reaching")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--her-frac", type=float, default=0.8)
    p.add_argument("--expl-noise", type=float, default=0.2)
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--bc-coef", type=float, default=0.5, help="small BC anchor to the pool (stay near data)")
    p.add_argument("--eval-every", type=int, default=50000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    latent_dim = int(cfg["latent_dim"])
    env = make_env(task.env_id, seed=0, max_episode_steps=args.ep_len)
    spec = obs_spec_from_env(env)
    compute_reward = getattr(env.unwrapped, "compute_reward", None)
    alow = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    ahigh = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    a_lo = env.action_space.low; a_hi = env.action_space.high
    rng = np.random.default_rng(0)

    pool = load_episodes_npz(args.episodes_npz)[: args.offline_episodes]

    actor = GoalConditionedPolicy(latent_dim=latent_dim, action_dim=spec.action_dim, hidden_dim=args.hidden).to(dev)
    actor.load_state_dict(torch.load(args.init_actor, map_location=dev, weights_only=False)["policy"])
    actor_t = GoalConditionedPolicy(latent_dim=latent_dim, action_dim=spec.action_dim, hidden_dim=args.hidden).to(dev)
    actor_t.load_state_dict(actor.state_dict())
    critic = Critic(latent_dim, spec.action_dim, args.hidden).to(dev)
    critic_t = Critic(latent_dim, spec.action_dim, args.hidden).to(dev)
    critic_t.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    def scale(a):
        return alow + (a + 1.0) * 0.5 * (ahigh - alow)

    @torch.no_grad()
    def act(obs):
        s = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
        a = scale(actor(wm.encode(s)))[0].cpu().numpy()
        a = a + rng.normal(0, args.expl_noise, size=spec.action_dim)
        return np.clip(a, a_lo, a_hi).astype(np.float32)

    def update(step):
        S, A, S2, R, D = sample_her(pool, spec, norm, compute_reward, args.batch_size, args.her_frac, rng)
        z = wm.encode(torch.from_numpy(S).to(dev)); z2 = wm.encode(torch.from_numpy(S2).to(dev))
        at = torch.from_numpy(A).to(dev); rt = torch.from_numpy(R).to(dev); dt = torch.from_numpy(D).to(dev)
        with torch.no_grad():
            noise = (torch.randn_like(at) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
            a2 = (scale(actor_t(z2)) + noise).clamp(alow, ahigh)
            q1t, q2t = critic_t(z2, a2)
            target = rt.unsqueeze(-1) + args.gamma * (1 - dt.unsqueeze(-1)) * torch.min(q1t, q2t)
        q1, q2 = critic(z, at)
        c_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()
        if step % args.policy_delay == 0:
            pi = scale(actor(z))
            q1pi, _ = critic(z, pi)
            lam = 1.0 / (q1pi.abs().mean().detach() + 1e-6)
            a_loss = -lam * q1pi.mean() + args.bc_coef * nn.functional.mse_loss(pi, at)
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            with torch.no_grad():
                for pt, ps in zip(actor_t.parameters(), actor.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * ps)
                for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * ps)
        return float(c_loss)

    def save():
        torch.save({"policy": actor.state_dict(),
                    "config": {"latent_dim": latent_dim, "action_dim": spec.action_dim, "hidden_dim": args.hidden}},
                   args.out)

    total = 0; ep_idx = 0; t0 = time.time()
    while total < args.env_steps:
        obs, _ = env.reset(seed=10_000 + ep_idx)
        states = [flatten_obs(obs)]; actions = []
        term = trunc = False
        while not (term or trunc):
            a = act(obs)
            obs, _, term, trunc, _ = env.step(a)
            actions.append(a.astype(np.float32)); states.append(flatten_obs(obs)); total += 1
            for _ in range(args.updates_per_step):
                cl = update(total)
            if total % 5000 == 0:
                print(json.dumps({"event": "online", "env_steps": total, "pool": len(pool),
                                  "c_loss": round(cl, 3), "sps": round(total / (time.time() - t0), 1)}), flush=True)
            if total % args.eval_every == 0:
                save()
                print(json.dumps({"event": "ckpt", "env_steps": total, "path": str(args.out)}), flush=True)
            if total >= args.env_steps:
                break
        if actions:
            pool.append(Episode(states=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32)))
        ep_idx += 1
    env.close()
    save()
    print(json.dumps({"event": "online_saved", "path": str(args.out), "env_steps": total}), flush=True)


if __name__ == "__main__":
    main()
