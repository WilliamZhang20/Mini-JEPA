"""Action-chunked RECTIFIED-FLOW goal-conditioned walker for AntMaze — the kitchen
recipe applied to locomotion, and the real shot at a robust ant gait.

Why flow, not BC: with HER, a single state maps to MANY action chunks (one per
relabeled goal/direction), so behaviour cloning *averages* them into mush (the
single-step and chunked-BC walkers both collapsed). A flow/diffusion policy
instead learns the conditional *distribution* over chunks and *samples* a single
coherent gait toward the goal — multimodality preserved, gait phase consistent.

  * conditioning c = JEPA latent of the (HER-relabeled) observation, optionally
    concatenated with the raw normalized obs (`--concat-raw`, the kitchen fix for
    control precision -- the predictive latent alone sheds proprioceptive detail).
  * target x = the next-H action chunk (flattened).
  * rectified flow: x_t = (1-t) x0 + t x1, x0~N(0,I), x1=chunk; the velocity field
    v(x_t, t, c) regresses x1 - x0. Sampling = Euler-integrate the ODE noise->chunk.

Executed receding-horizon by eval_hjepa_hwm.py (flow-macro HWM) low level.
"""
from __future__ import annotations

import argparse
import json
import math
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
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task
from scripts.train_chunked_walker import build_chunks


def build_directed_chunks(episodes, spec, normalizer, chunk, rng, *,
                          max_relabel_h=60, min_progress=0.15):
    """(state_t with a NEAR relabeled goal, action chunk) kept only when the ant
    makes real progress toward that goal over the chunk.

    The default HER over the whole trajectory relabels to far/arbitrary goals and
    mixes in wandering/standing-still segments, so the BC/flow walker learns a
    slow, undirected gait (the AntMaze bottleneck). This de-blurs the walker
    toward DECISIVE locomotion (the 'no blurring' law for a gait): relabel the
    desired goal to a *nearby* future achieved position (t+1..max_relabel_h) and
    keep the sample only if the achieved xy moves at least ``min_progress`` closer
    to that goal across the chunk. The emphasized (desired-achieved) direction
    then always points at a goal the chunk actually advances toward.
    """
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    ds, de = spec.obs_dim + spec.goal_dim, spec.obs_dim + 2 * spec.goal_dim
    S_list, C_list = [], []
    for ep in episodes:
        S, A = ep.states, ep.actions
        T = len(A)
        for t in range(T):
            dh = int(rng.integers(1, max_relabel_h + 1))
            tf = min(t + dh, len(S) - 1)
            goal = S[tf, gs:ge]
            t_end = min(t + chunk, len(S) - 1)
            progress = float(np.linalg.norm(S[t, gs:ge] - goal) - np.linalg.norm(S[t_end, gs:ge] - goal))
            if progress < min_progress:
                continue
            s = S[t].copy()
            s[ds:de] = goal
            chunk_a = A[t: t + chunk]
            if len(chunk_a) < chunk:
                pad = np.repeat(chunk_a[-1:], chunk - len(chunk_a), axis=0)
                chunk_a = np.concatenate([chunk_a, pad], axis=0)
            S_list.append(s); C_list.append(chunk_a)
    S = normalizer.encode(np.asarray(S_list, np.float32))
    return S, np.asarray(C_list, np.float32)


def timestep_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowNet(nn.Module):
    """Velocity field v(x_t, t, c) for rectified flow over action chunks."""

    def __init__(self, chunk_dim, cond_dim, hidden=512, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.net = MLP([chunk_dim + t_dim + cond_dim, hidden, hidden, hidden, chunk_dim], layer_norm=True)

    def forward(self, x, t, c):
        te = timestep_embed(t, self.t_dim)
        return self.net(torch.cat([x, te, c], dim=-1))

    @torch.no_grad()
    def sample(self, c, chunk_dim, n_steps=10):
        x = torch.randn(c.shape[0], chunk_dim, device=c.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((c.shape[0], 1), i * dt, device=c.device)
            x = x + self(x, t, c) * dt
        return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--her-frac", type=float, default=0.85)
    p.add_argument("--directed", action="store_true",
                   help="Use directed-motion data prep (near relabel + progress filter) for a faster, decisive gait instead of whole-trajectory HER.")
    p.add_argument("--max-relabel-h", type=int, default=60,
                   help="[directed] relabel the goal to a future achieved position within this many steps.")
    p.add_argument("--min-progress", type=float, default=0.15,
                   help="[directed] keep a sample only if the ant closes at least this much xy distance to the relabeled goal over the chunk.")
    p.add_argument("--concat-raw", action="store_true",
                   help="condition on [JEPA latent | raw normalized obs] (kitchen fix for control precision)")
    p.add_argument("--emphasis-repeat", type=int, default=0,
                   help="Relocate-style input-feature emphasis: duplicate the (desired_goal - achieved_goal) xy vector this many times in the flow conditioning. This is the servo DIRECTION the walker must move (the locomotion analog of the palm-ball vector), so the sampled gait chunk heads at the live subgoal instead of averaging over HER directions.")
    p.add_argument("--agent-dims", default="27,29",
                   help="Normalized-obs slice (lo,hi) of the achieved-goal / agent xy.")
    p.add_argument("--goal-dims", default="29,31",
                   help="Normalized-obs slice (lo,hi) of the desired-goal xy.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()

    rng = np.random.default_rng(0)
    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    if args.directed:
        S, C = build_directed_chunks(eps, spec, norm, args.chunk, rng,
                                     max_relabel_h=args.max_relabel_h, min_progress=args.min_progress)
    else:
        S, C = build_chunks(eps, spec, norm, args.chunk, args.her_frac, rng)  # S normalized states, C chunks
    St = torch.from_numpy(S).to(dev)
    with torch.no_grad():
        Z = torch.cat([wm.encode(St[i:i + 16384]) for i in range(0, len(St), 16384)], 0)
    cond = torch.cat([Z, St], dim=1) if args.concat_raw else Z
    a_lo, a_hi = (int(x) for x in args.agent_dims.split(","))
    g_lo, g_hi = (int(x) for x in args.goal_dims.split(","))
    if args.emphasis_repeat > 0:
        delta = (St[:, g_lo:g_hi] - St[:, a_lo:a_hi]).repeat(1, args.emphasis_repeat)
        cond = torch.cat([cond, delta], dim=1)
    Ct = torch.from_numpy(C.reshape(len(C), -1)).to(dev)
    chunk_dim = Ct.shape[1]; cond_dim = cond.shape[1]; N = len(cond)
    print(json.dumps({"event": "flow_data", "n": N, "chunk_dim": chunk_dim, "cond_dim": cond_dim,
                      "concat_raw": bool(args.concat_raw), "emphasis_repeat": int(args.emphasis_repeat)}), flush=True)

    net = FlowNet(chunk_dim, cond_dim, args.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        x1 = Ct[idx]; c = cond[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=dev)
        xt = (1 - t) * x0 + t * x1
        v = net(xt, t, c)
        loss = nn.functional.mse_loss(v, x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0:
            print(json.dumps({"event": "flow_train", "step": step, "loss": round(float(loss), 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"flow": net.state_dict(),
                "config": {"chunk": args.chunk, "action_dim": spec.action_dim, "hidden": args.hidden,
                           "chunk_dim": chunk_dim, "cond_dim": cond_dim, "latent_dim": int(cfg["latent_dim"]),
                           "concat_raw": bool(args.concat_raw),
                           "emphasis_repeat": int(args.emphasis_repeat),
                           "agent_dims": [a_lo, a_hi], "goal_dims": [g_lo, g_hi]}},
               args.out)
    print(json.dumps({"event": "flow_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
