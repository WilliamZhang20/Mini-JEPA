"""Gating test for a latent Dreamer: can a *dedicated* latent-dynamics net predict
the next JEPA latent better than a no-op?

The WM's built-in predictor is trained inside a crowded objective (VICReg + probes +
normalized pred loss) and on kitchen turns out worse-than-no-op in absolute terms.
Here we train ONE thing — f(z,a) -> z' (residual) — purely on next-latent MSE over
the frozen encoder, and report multi-step rollout error vs the no-op baseline
(predict z stays put). If this beats no-op, a latent actor-critic (Dreamer) is
viable; if not, the encoder latent is non-smooth and must be retrained.
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

from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden):
        super().__init__()
        self.net = MLP([latent_dim + action_dim, hidden, hidden, hidden, latent_dim], layer_norm=True)

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a], dim=-1))  # residual


class EnsembleLatentDynamics(nn.Module):
    """K independent residual dynamics heads (diverse inits + bootstrap-free).
    mean() for prediction; disagreement() = inter-head variance for the
    anti-model-exploitation penalty (Roadmap A #2)."""

    def __init__(self, latent_dim, action_dim, hidden, n_heads=5):
        super().__init__()
        self.heads = nn.ModuleList([LatentDynamics(latent_dim, action_dim, hidden) for _ in range(n_heads)])

    def all_heads(self, z, a):
        return torch.stack([h(z, a) for h in self.heads], 0)   # [K, B, L]

    def forward(self, z, a):
        return self.all_heads(z, a).mean(0)

    def disagreement(self, z, a):
        preds = self.all_heads(z, a)                            # [K, B, L]
        return preds.var(0).mean(-1)                            # [B] epistemic uncertainty


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=8, help="multi-step training horizon")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-heads", type=int, default=1, help=">1 trains an ensemble for disagreement penalty")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    latent_dim = int(cfg["latent_dim"])

    data = np.load(args.episodes_npz, allow_pickle=True)
    states, actions = data["states"], data["actions"]
    K = args.seq_len
    # encode all episodes into a single flat latent/action buffer; record valid
    # window starts (s..s+K all within one episode) so batching is a pure gather.
    Z_list, A_list, starts, off = [], [], [], 0
    with torch.no_grad():
        for i in range(min(len(states), args.max_episodes)):
            S = np.asarray(states[i], np.float32); A = np.asarray(actions[i], np.float32)
            z = wm.encode(torch.from_numpy(norm.encode(S)).to(dev))  # [T+1, L]
            T = len(A)
            Z_list.append(z[:T + 1]); A_list.append(torch.from_numpy(A).to(dev))
            starts.extend(range(off, off + max(0, T - K)))   # need t+K <= T (z index up to off+T)
            off += T + 1                                     # episodes are length T+1 in Z
    # NOTE: action index a is offset-aligned to Z index within an episode (a[t] pairs z[t]->z[t+1])
    Z_all = torch.cat(Z_list, 0)                              # [sum(T+1), L]
    # build action buffer aligned to Z indices (pad one slot per episode so a-index == z-index)
    A_all = torch.zeros(Z_all.shape[0], spec.action_dim, device=dev)
    p = 0
    for z, a in zip(Z_list, A_list):
        A_all[p:p + len(a)] = a; p += len(z)
    starts = torch.tensor(starts, device=dev, dtype=torch.long)
    real_change = (Z_all[1:] - Z_all[:-1]).norm(dim=-1).mean()
    print(json.dumps({"event": "encoded", "windows": int(len(starts)),
                      "real_step_change": round(float(real_change), 3),
                      "latent_norm": round(float(Z_all.norm(dim=-1).mean()), 3)}), flush=True)

    if args.n_heads > 1:
        dyn = EnsembleLatentDynamics(latent_dim, spec.action_dim, args.hidden, args.n_heads).to(dev)
    else:
        dyn = LatentDynamics(latent_dim, spec.action_dim, args.hidden).to(dev)
    opt = torch.optim.Adam(dyn.parameters(), lr=args.lr)
    ar = torch.arange(K, device=dev)

    def sample_batch():
        s = starts[torch.randint(0, len(starts), (args.batch_size,), device=dev)]  # [B]
        z0 = Z_all[s]                                         # [B, L]
        idx = s[:, None] + ar[None, :]                        # [B, K] action indices t..t+K-1
        aS = A_all[idx]                                       # [B, K, A]
        zT = Z_all[idx + 1]                                   # [B, K, L] targets z[t+1..t+K]
        return z0, aS, zT

    heads = list(dyn.heads) if args.n_heads > 1 else [dyn]

    def rollout_loss(head):
        z0, aS, zT = sample_batch()            # each head its own batch -> bootstrap diversity
        z = z0; loss = 0.0
        for k in range(K):
            z = head(z, aS[:, k])              # open-loop rollout (predicted feeds predicted)
            loss = loss + nn.functional.mse_loss(z, zT[:, k])
        return loss / K

    for step in range(1, args.steps + 1):
        loss = sum(rollout_loss(h) for h in heads) / len(heads)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0:
            print(json.dumps({"event": "dyn_train", "step": step, "rollout_mse": round(float(loss), 4)}), flush=True)

    # --- evaluation: open-loop k-step error vs no-op baseline ---
    dyn.eval()
    with torch.no_grad():
        errs = {k: [] for k in [1, 2, 4, 8]}; noop = {k: [] for k in [1, 2, 4, 8]}
        for _ in range(40):
            z0, aS, zT = sample_batch()
            z = z0
            for k in range(K):
                z = dyn(z, aS[:, k])
                kk = k + 1
                if kk in errs:
                    errs[kk].append(float((z - zT[:, k]).norm(dim=-1).mean()))
                    noop[kk].append(float((z0 - zT[:, k]).norm(dim=-1).mean()))
    summary = {k: {"dyn": round(np.mean(errs[k]), 3), "noop": round(np.mean(noop[k]), 3),
                   "ratio": round(np.mean(errs[k]) / np.mean(noop[k]), 3)} for k in errs}
    print(json.dumps({"event": "rollout_eval", "k_step_error": summary}), flush=True)
    beats = bool(summary[1]["ratio"] < 1.0)
    print(json.dumps({"event": "verdict", "beats_noop_1step": beats,
                      "msg": "latent Dreamer viable" if beats else "encoder latent non-smooth — retrain encoder"}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dyn.state_dict(), "latent_dim": latent_dim,
                "action_dim": spec.action_dim, "hidden": args.hidden, "n_heads": args.n_heads}, args.out)
    print(json.dumps({"event": "dyn_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
