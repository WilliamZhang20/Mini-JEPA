"""Latent Dreamer: a DL controller trained purely in JEPA-latent imagination.

This is "DL controller + DL JEPA" with no CEM. On top of the FROZEN JEPA encoder we
use a learned latent dynamics f(z,a)->z' (beats no-op, see train_latent_dynamics.py)
and a reward head r(z) (predicted task-count). An actor pi(z) and critic V(z) are
trained by IMAGINING short rollouts through f and maximising the lambda-return of
the reward head -- the actor's gradient flows analytically back through the
differentiable dynamics (Dreamer-style), so the world model trains the controller.

Stability ("don't lose on that"):
  * every imagined rollout STARTS from a real encoded latent z=encode(obs) -- the
    controller is always grounded on real JEPA output, never free-floating;
  * imagination horizon is capped to where f is trusted (default 8 steps);
  * LayerNorm critic + target critic + soft updates (the machinery that killed the
    kitchen Q-divergence earlier);
  * an optional behaviour-prior term keeps actions near the demo manifold so the
    actor can't exploit the model into OOD latents.
The actor is a GoalConditionedPolicy so eval loads it like any other policy.
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

from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP, GoalConditionedPolicy
from jepa_robotics.tasks import resolve_task
from scripts.train_latent_dynamics import EnsembleLatentDynamics, LatentDynamics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dynamics", type=Path, required=True)
    p.add_argument("--reward-head", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--horizon", type=int, default=8, help="imagination horizon (trusted dynamics range)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--bc-coef", type=float, default=0.1, help="behaviour-prior weight (anti-exploitation)")
    p.add_argument("--disagree-coef", type=float, default=1.0,
                   help="penalise ensemble dynamics disagreement in the imagined reward (anti-exploitation)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    L = int(cfg["latent_dim"])
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env)
    alow = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    ahigh = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    env.close()
    A = spec.action_dim

    # frozen world-model pieces
    dck = torch.load(args.dynamics, map_location=dev, weights_only=False)
    n_heads = int(dck.get("n_heads", 1))
    if n_heads > 1:
        dyn = EnsembleLatentDynamics(L, A, dck["hidden"], n_heads).to(dev)
    else:
        dyn = LatentDynamics(L, A, dck["hidden"]).to(dev)
    dyn.load_state_dict(dck["state_dict"]); dyn.eval()
    rck = torch.load(args.reward_head, map_location=dev, weights_only=False)
    rhead = MLP([rck["latent_dim"], rck["hidden"], rck["hidden"], 1]).to(dev)
    rhead.load_state_dict(rck["state_dict"]); rhead.eval()
    for m in (dyn, rhead):
        for q in m.parameters():
            q.requires_grad_(False)

    # real start-latent buffer (+ demo actions for the behaviour prior)
    data = np.load(args.episodes_npz, allow_pickle=True)
    states, actions = data["states"], data["actions"]
    Z_list, A_list = [], []
    with torch.no_grad():
        for i in range(min(len(states), args.max_episodes)):
            S = np.asarray(states[i], np.float32); a = np.asarray(actions[i], np.float32)
            z = wm.encode(torch.from_numpy(norm.encode(S)).to(dev))
            Z_list.append(z[:len(a)]); A_list.append(torch.from_numpy(a).to(dev))
    Z0 = torch.cat(Z_list, 0); A0 = torch.cat(A_list, 0)
    N = len(Z0)
    print(json.dumps({"event": "dreamer_data", "starts": N, "latent_dim": L, "horizon": args.horizon}), flush=True)

    actor = GoalConditionedPolicy(latent_dim=L, action_dim=A, hidden_dim=args.hidden).to(dev)
    critic = MLP([L, args.hidden, args.hidden, 1], layer_norm=True).to(dev)
    critic_t = MLP([L, args.hidden, args.hidden, 1], layer_norm=True).to(dev)
    critic_t.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    def scale(a):  # tanh [-1,1] -> action range
        return alow + (a + 1.0) * 0.5 * (ahigh - alow)

    H = args.horizon
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        z = Z0[idx]
        a_demo = A0[idx]

        # --- imagine H steps through the frozen dynamics (actor grads flow through) ---
        ensemble = n_heads > 1 and args.disagree_coef > 0.0
        zs = [z]; rs = []; a_all = []
        for t in range(H):
            a = actor(z)
            a_all.append(a)
            a_env = scale(a)
            # penalise reward where the ensemble disagrees -> actor avoids exploitable OOD latents
            disagree = dyn.disagreement(z, a_env) if ensemble else 0.0
            z = dyn(z, a_env)
            zs.append(z)
            rs.append(rhead(z).squeeze(-1) - args.disagree_coef * disagree)  # uncertainty-penalised reward
        a_first = a_all[0]
        Ztraj = torch.stack(zs, 0)                  # [H+1, B, L]
        R = torch.stack(rs, 0)                       # [H, B]

        with torch.no_grad():
            V_all = critic_t(Ztraj).squeeze(-1)     # [H+1, B]
        # TD(lambda) returns over the imagined trajectory
        returns = torch.zeros_like(R)
        nxt = V_all[H]
        for t in reversed(range(H)):
            nxt = R[t] + args.gamma * ((1 - args.lam) * V_all[t + 1] + args.lam * nxt)
            returns[t] = nxt

        # --- critic: regress V(z_t) -> lambda-returns (imagined states detached) ---
        V_pred = critic(Ztraj[:H].detach()).squeeze(-1)     # [H, B]
        c_loss = nn.functional.mse_loss(V_pred, returns.detach())
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()

        # --- actor: maximise imagined return (analytic grad through dyn) + behaviour prior ---
        actor_obj = -returns.mean()
        bc = nn.functional.mse_loss(scale(a_first), a_demo)
        a_loss = actor_obj + args.bc_coef * bc
        a_opt.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        a_opt.step()

        with torch.no_grad():
            for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * ps)

        if step % 5000 == 0:
            print(json.dumps({"event": "dreamer", "step": step,
                              "return": round(float(returns.mean()), 3), "v": round(float(V_pred.mean()), 3),
                              "c_loss": round(float(c_loss), 4), "bc": round(float(bc), 4),
                              "imag_reward": round(float(R.mean()), 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": actor.state_dict(),
                "config": {"latent_dim": L, "action_dim": A, "hidden_dim": args.hidden}}, args.out)
    print(json.dumps({"event": "dreamer_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
