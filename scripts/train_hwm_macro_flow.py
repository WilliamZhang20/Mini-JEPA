"""Train a future-conditioned rectified-flow prior over HWM macro-actions.

The HWM-CEM high level (arXiv:2604.03208 in this repo) plans by CEM over
macro-actions sampled from a GLOBAL Gaussian and scores the decoded terminal
position vs the goal. On a walled maze this hallucinates wall-crossing: the
Gaussian proposes macro-actions whose decoded subgoal heads straight at the goal
through a wall, because nothing constrains the proposal to feasible transitions.

Fix (the flow-on-the-manifold theme applied to the macro search): learn a
flow prior over macro-actions conditioned on the current abstract latent and a
target position,

    z_high = psi(encode(s_t))                      # frozen HWM high-encoder
    macro  = macro_encoder(a_{t:t+N})              # frozen HWM macro-encoder
    target = achieved_goal_{t + k*N}   (random k)  # a demonstrated future xy
    flow( macro | z_high, target )

Every sample is an ON-MANIFOLD, demonstrated macro-action, so chaining samples
through g explores only feasible (around-wall) trajectories. At plan time we
condition on (z_high, goal_xy), sample a few macros, roll them through the frozen
g, decode the next subgoal, and pick the best -- feasible by construction.

The HWM (psi/macro/g/dec) is FROZEN; only the macro-flow is trained on top.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.priors import EpsNet
from scripts.train_hjepa_hwm import HighEncoder, MacroEncoder, MacroPredictor, SubgoalDecoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--hwm", type=Path, required=True, help="Frozen HWM checkpoint (psi/macro/g/dec).")
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--max-future-hops", type=int, default=6,
                   help="Condition on futures 1..K macro-hops ahead (random per sample) so the flow learns to head toward near and far targets.")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    art = torch.load(args.hwm, map_location=dev, weights_only=False)
    c = art["config"]
    psi = HighEncoder(c["low_dim"], c["abstract_dim"], c["hidden"]).to(dev); psi.load_state_dict(art["psi"]); psi.eval()
    macro = MacroEncoder(c["action_dim"], c["macro_dim"]).to(dev); macro.load_state_dict(art["macro"]); macro.eval()
    for m in (psi, macro):
        for q in m.parameters():
            q.requires_grad_(False)
    stride = int(c["stride"])
    macro_dim = int(c["macro_dim"])

    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim

    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    Zc, Tgt, Mac = [], [], []
    with torch.no_grad():
        for ep in eps:
            S = ep.states.astype(np.float32); A = ep.actions.astype(np.float32)
            T = len(A)
            if T < 2 * stride:
                continue
            Sn = norm.encode(S).astype(np.float32)
            for t in range(0, T - stride, max(1, stride // 2)):
                zc = None
                # future targets at 1..K hops (clamped to trajectory end)
                for k in range(1, args.max_future_hops + 1):
                    tf = t + k * stride
                    if tf >= len(S):
                        tf = len(S) - 1
                    if zc is None:
                        zc = psi(wm.encode(torch.from_numpy(Sn[t:t + 1]).to(dev)))[0]
                    m = macro(torch.from_numpy(A[t:t + stride][None]).to(dev))[0]
                    Zc.append(zc.cpu().numpy()); Tgt.append(S[tf, gs:ge].copy()); Mac.append(m.cpu().numpy())
                    if tf == len(S) - 1:
                        break
    Zc = torch.from_numpy(np.asarray(Zc, np.float32)).to(dev)
    Tgt = torch.from_numpy(np.asarray(Tgt, np.float32)).to(dev)
    Mac = torch.from_numpy(np.asarray(Mac, np.float32)).to(dev)
    Cond = torch.cat([Zc, Tgt], dim=1)
    cond_dim = Cond.shape[1]
    print(json.dumps({"event": "macro_flow_data", "n": int(len(Cond)), "cond_dim": int(cond_dim),
                      "macro_dim": macro_dim, "stride": stride}), flush=True)

    net = EpsNet(macro_dim, cond_dim, args.hidden, n_blocks=args.n_blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    N = len(Cond)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        x1 = Mac[idx]; cond = Cond[idx]
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=dev)
        xt = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        pred_v = net(xt, tau * args.flow_steps, cond)
        loss = nn.functional.mse_loss(pred_v, x1 - x0)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for key, val in net.state_dict().items():
                ema[key].mul_(0.999).add_(val.detach(), alpha=0.001)
        if step == 1 or step % 5000 == 0:
            print(json.dumps({"event": "macro_flow_train", "step": step, "loss": round(float(loss), 5)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema, "state_dict": net.state_dict(),
                "config": {"macro_dim": macro_dim, "cond_dim": cond_dim, "hidden": args.hidden,
                           "n_blocks": args.n_blocks, "flow_steps": args.flow_steps,
                           "goal_dim": spec.goal_dim, "abstract_dim": c["abstract_dim"], "hwm": str(args.hwm)}},
               args.out)
    print(json.dumps({"event": "macro_flow_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
