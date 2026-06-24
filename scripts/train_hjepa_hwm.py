"""Proper Hierarchical-JEPA high level (HWM), trained strictly ON TOP of the
frozen low-level JEPA, per the two-tier world-model recipe:

  * psi  (HighEncoder)   : MLP that compresses the frozen low latent z_t (192) ->
                           a small abstract latent z_high (~16). Trained from
                           scratch; low-level frozen underneath.
  * macro (MacroEncoder) : a GRU that compresses an N-step action chunk into ONE
                           macro-action vector m_t (one "hop").
  * g   (MacroPredictor) : g(z_high_t, m_t) -> z_high_{t+N}, trained with the SAME
                           JEPA recipe as the low level: stop-gradient target
                           (sg[psi(z_{t+N})]), VICReg variance+covariance to stop
                           collapse, normalized-MSE prediction loss.
  * dec (SubgoalDecoder) : abstract latent -> achieved_goal position, so a planned
                           abstract subgoal can be decoded into a goal the
                           UNCHANGED low-level position-conditioned policy reaches.

What never changes: the low-level encoder/predictor/policy. Everything here sits
on top. Planning (CEM over macro-actions through g) lives in eval_hjepa_hwm.py.
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
from jepa_robotics.models import MLP, covariance_regularizer, normalized_mse, variance_regularizer
from jepa_robotics.tasks import resolve_task


class HighEncoder(nn.Module):
    def __init__(self, low_dim, abstract_dim, hidden=256):
        super().__init__()
        self.net = MLP([low_dim, hidden, hidden, abstract_dim], layer_norm=True)

    def forward(self, z):
        return self.net(z)


class MacroEncoder(nn.Module):
    def __init__(self, action_dim, macro_dim, hidden=128):
        super().__init__()
        self.gru = nn.GRU(action_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, macro_dim)

    def forward(self, chunk):  # chunk: [B, N, action_dim]
        _, h = self.gru(chunk)
        return self.head(h[-1])


class MacroPredictor(nn.Module):
    def __init__(self, abstract_dim, macro_dim, hidden=256):
        super().__init__()
        self.net = MLP([abstract_dim + macro_dim, hidden, hidden, abstract_dim], layer_norm=True)

    def forward(self, z_high, m):
        return z_high + self.net(torch.cat([z_high, m], dim=-1))  # residual macro-step


class SubgoalDecoder(nn.Module):
    def __init__(self, abstract_dim, goal_dim, hidden=128):
        super().__init__()
        self.net = MLP([abstract_dim, hidden, hidden, goal_dim], layer_norm=True)

    def forward(self, z_high):
        return self.net(z_high)


def build_macro_data(episodes, spec, normalizer, wm, dev, stride, overlap=2):
    """Subsample trajectories at the macro stride: returns frozen low latents at
    t and t+N, the N-step action chunk, and the achieved_goal position at t."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    St, S2t, Ch, Pos = [], [], [], []
    step = max(1, stride // overlap)
    for ep in episodes:
        T = len(ep.actions)
        for t in range(0, T - stride, step):
            St.append(ep.states[t]); S2t.append(ep.states[t + stride])
            Ch.append(ep.actions[t:t + stride]); Pos.append(ep.states[t, gs:ge])
    St = normalizer.encode(np.asarray(St, np.float32))
    S2t = normalizer.encode(np.asarray(S2t, np.float32))
    with torch.no_grad():
        Z = torch.cat([wm.encode(torch.from_numpy(St[i:i + 16384]).to(dev)) for i in range(0, len(St), 16384)], 0)
        Z2 = torch.cat([wm.encode(torch.from_numpy(S2t[i:i + 16384]).to(dev)) for i in range(0, len(S2t), 16384)], 0)
    Ch = torch.from_numpy(np.asarray(Ch, np.float32)).to(dev)
    Pos = torch.from_numpy(np.asarray(Pos, np.float32)).to(dev)
    return Z, Z2, Ch, Pos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--holdout-frac", type=float, default=0.1, help="held-out trajectories for the generalization test")
    p.add_argument("--stride", type=int, default=30, help="macro-step N (env steps per hop)")
    p.add_argument("--abstract-dim", type=int, default=16)
    p.add_argument("--macro-dim", type=int, default=16)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=0.5)
    p.add_argument("--lambda-dec", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    low_dim = int(cfg["latent_dim"])
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()

    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    n_hold = max(1, int(len(eps) * args.holdout_frac))
    train_eps, hold_eps = eps[n_hold:], eps[:n_hold]
    Z, Z2, Ch, Pos = build_macro_data(train_eps, spec, norm, wm, dev, args.stride)
    Zh, Z2h, Chh, Posh = build_macro_data(hold_eps, spec, norm, wm, dev, args.stride)
    print(json.dumps({"event": "hwm_data", "train_macro": len(Z), "holdout_macro": len(Zh),
                      "stride": args.stride, "abstract_dim": args.abstract_dim}), flush=True)

    psi = HighEncoder(low_dim, args.abstract_dim, args.hidden).to(dev)
    macro = MacroEncoder(spec.action_dim, args.macro_dim).to(dev)
    g = MacroPredictor(args.abstract_dim, args.macro_dim, args.hidden).to(dev)
    dec = SubgoalDecoder(args.abstract_dim, spec.goal_dim).to(dev)
    params = list(psi.parameters()) + list(macro.parameters()) + list(g.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    N = len(Z)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        z, z2, ch, pos = Z[idx], Z2[idx], Ch[idx], Pos[idx]
        z_high = psi(z)
        with torch.no_grad():
            target = psi(z2)                       # stop-gradient target (JEPA recipe)
        m = macro(ch)
        pred = g(z_high, m)
        l_pred = normalized_mse(pred, target)
        l_var = variance_regularizer(z_high)       # VICReg variance term (anti-collapse)
        l_cov = covariance_regularizer(z_high)     # VICReg covariance term (decorrelate)
        l_dec = nn.functional.mse_loss(dec(z_high), pos)
        loss = l_pred + args.lambda_var * l_var + args.lambda_cov * l_cov + args.lambda_dec * l_dec
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            with torch.no_grad():
                zh = psi(Zh); mh = macro(Chh); ph = g(zh, mh)
                hold_pred = float(normalized_mse(ph, psi(Z2h)))
            print(json.dumps({"event": "hwm_train", "step": step, "pred": round(float(l_pred), 4),
                              "var": round(float(l_var), 3), "cov": round(float(l_cov), 3),
                              "dec": round(float(l_dec), 4), "holdout_pred": round(hold_pred, 4)}), flush=True)

    # macro-action statistics for CEM sampling
    with torch.no_grad():
        M = macro(Ch)
        m_mean = M.mean(0).cpu().numpy(); m_std = M.std(0).cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"psi": psi.state_dict(), "macro": macro.state_dict(), "g": g.state_dict(),
                "dec": dec.state_dict(),
                "config": {"low_dim": low_dim, "abstract_dim": args.abstract_dim, "macro_dim": args.macro_dim,
                           "hidden": args.hidden, "stride": args.stride, "action_dim": spec.action_dim,
                           "goal_dim": spec.goal_dim, "m_mean": m_mean.tolist(), "m_std": m_std.tolist()}},
               args.out)
    print(json.dumps({"event": "hwm_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
