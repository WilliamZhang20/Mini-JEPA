"""Diffusion policy (action-chunked) on the frozen JEPA latent — the research-grade
low-level controller for FrankaKitchen.

Flat feedforward control (BC / TD3+BC / latent Dreamer) all hit ~0 on kitchen for two
reasons the diagnostics isolated: (a) a per-step deterministic policy compounds error
over the 280-step contact sequence, and (b) it averages the *multimodal* demos
(different task orders) into mush. A diffusion policy attacks both:

  * it predicts a CHUNK of H future actions at once (fewer decision points ->
    far less compounding error — the ACT/Diffusion-Policy insight), and
  * it models a full multimodal action distribution via DDPM denoising (no averaging;
    it can commit to one of several valid behaviours).

Conditioning is the JEPA latent z = encode(obs) (in-thesis: the controller reads JEPA
output). eps_theta(a_noisy, t, z) is a residual MLP; standard DDPM noise-prediction loss.
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
from jepa_robotics.algos.priors import EpsNet, make_ddpm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=400)
    p.add_argument("--chunk", type=int, default=8, help="action-chunk horizon H")
    p.add_argument("--obs-hist", type=int, default=1, help="number of past JEPA latents to condition on (temporal context)")
    p.add_argument("--raw-obs", action="store_true",
                   help="ablation: condition on normalized RAW obs instead of the JEPA latent (tests whether JEPA helps)")
    p.add_argument("--concat-raw", action="store_true",
                   help="condition on [raw obs ++ JEPA latent] — raw precision + JEPA structure (provable floor: cannot lose to raw)")
    p.add_argument("--steps", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--objective", choices=["diffusion", "flow"], default="diffusion",
                   help="diffusion=DDPM noise-prediction; flow=conditional/rectified flow matching (faster ODE sampling)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm-start the policy weights from an existing checkpoint (online fine-tuning)")
    p.add_argument("--progress-cond", action="store_true",
                   help="append a scalar 'subtasks completed so far' (/4) to the conditioning — a lightweight "
                        "hierarchy that tells the policy its stage in the sequence (targets chaining/full-4)")
    p.add_argument("--cfg-dropout", type=float, default=0.0,
                   help="classifier-free guidance: prob of zeroing the conditioning during training so the net "
                        "also learns the unconditional model (enables guided sampling at eval)")
    p.add_argument("--subtask-cond", action="store_true",
                   help="skill-hierarchy: append the target-subtask one-hot (4-d, from a labeled npz with 'targets') "
                        "to the conditioning so the flow policy is a subtask-conditioned skill")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    L = int(cfg["latent_dim"])
    H = args.chunk

    data = np.load(args.episodes_npz, allow_pickle=True)
    states, actions = data["states"], data["actions"]
    rewards = data["rewards"] if "rewards" in data else None
    targets = data["targets"] if ("targets" in data and args.subtask_cond) else None
    A_dim = int(np.asarray(actions[0]).shape[1])
    HH = args.obs_hist
    obs_w = int(norm.encode(np.asarray(states[0], np.float32)[:1]).shape[1])
    base_dim = obs_w if args.raw_obs else (obs_w + L if args.concat_raw else L)
    feat_dim = base_dim + (1 if args.progress_cond else 0) + (4 if args.subtask_cond else 0)

    def featurize(Sn):  # Sn: [T, obs_w] normalized obs -> per-frame conditioning feature
        if args.raw_obs:
            return Sn
        z = wm.encode(Sn)
        return torch.cat([Sn, z], dim=-1) if args.concat_raw else z

    cond_list, chunk_list = [], []
    with torch.no_grad():
        for i in range(min(len(states), args.max_episodes)):
            S = np.asarray(states[i], np.float32); Aep = np.asarray(actions[i], np.float32)
            T = len(Aep)
            if T <= H:
                continue
            z = featurize(torch.from_numpy(norm.encode(S)).to(dev))  # [T+1, base_dim]
            if args.progress_cond:
                # progress at state t = subtasks completed entering state t (from the count reward)
                R = np.asarray(rewards[i], np.float32)
                prog = np.concatenate([[0.0], np.maximum.accumulate(R)])[: len(z)] / 4.0  # [T+1]
                z = torch.cat([z, torch.from_numpy(prog.astype(np.float32)).to(dev).unsqueeze(-1)], dim=-1)
            if args.subtask_cond:
                # target-subtask one-hot at state t (pad the last state with the final target)
                tg = np.asarray(targets[i], np.float32)                      # [T, 4]
                tg = np.concatenate([tg, tg[-1:]], axis=0)[: len(z)]         # [T+1, 4]
                z = torch.cat([z, torch.from_numpy(tg).to(dev)], dim=-1)
            for t in range(T - H):
                # condition on the last HH latents ending at t (pad with z[0] at the start)
                hist = torch.cat([z[max(0, t - (HH - 1) + h)] for h in range(HH)], dim=-1)  # [HH*L]
                cond_list.append(hist)
                chunk_list.append(torch.from_numpy(Aep[t:t + H].reshape(-1)).to(dev))  # [H*A]
    Cond = torch.stack(cond_list, 0)              # [N, HH*L]
    Chunk = torch.stack(chunk_list, 0)            # [N, H*A]
    N = len(Cond); chunk_dim = H * A_dim; cond_dim = HH * feat_dim
    print(json.dumps({"event": "diff_data", "pairs": N, "chunk_dim": chunk_dim,
                      "cond_dim": cond_dim, "action_dim": A_dim, "H": H, "obs_hist": HH,
                      "raw_obs": bool(args.raw_obs), "concat_raw": bool(args.concat_raw),
                      "progress_cond": bool(args.progress_cond), "subtask_cond": bool(args.subtask_cond)}), flush=True)

    ddpm = make_ddpm(args.diffusion_steps, dev)
    abar = ddpm["abar"]
    net = EpsNet(chunk_dim, cond_dim, args.hidden, n_blocks=args.n_blocks).to(dev)
    if args.init_from is not None:   # warm-start (online fine-tuning) from an existing policy
        ick = torch.load(args.init_from, map_location=dev, weights_only=False)
        net.load_state_dict(ick["ema"]); print(json.dumps({"event": "warm_start", "from": str(args.init_from)}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # EMA weights (diffusion policies rely heavily on EMA for stable sampling)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema_decay = 0.999

    Tsteps = args.diffusion_steps
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), device=dev)
        a0 = Chunk[idx]; c = Cond[idx]
        if args.cfg_dropout > 0:  # CFG: randomly drop conditioning -> net also learns the unconditional model
            keep = (torch.rand(c.shape[0], 1, device=dev) >= args.cfg_dropout).float()
            c = c * keep
        if args.objective == "diffusion":
            t = torch.randint(0, Tsteps, (args.batch_size,), device=dev)
            noise = torch.randn_like(a0)
            ab = abar[t][:, None]
            a_t = torch.sqrt(ab) * a0 + torch.sqrt(1 - ab) * noise
            pred = net(a_t, t, c)
            loss = nn.functional.mse_loss(pred, noise)
        else:  # conditional flow matching: straight path noise->data, predict velocity
            tau = torch.rand(args.batch_size, device=dev)            # continuous t in [0,1]
            x0 = torch.randn_like(a0)                                # noise endpoint
            x_t = (1 - tau)[:, None] * x0 + tau[:, None] * a0        # linear interpolant
            target = a0 - x0                                         # constant velocity to data
            pred = net(x_t, tau * Tsteps, c)                         # scale t to the embedding range used by diffusion
            loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
        if step % 5000 == 0:
            print(json.dumps({"event": "diff_train", "step": step, "loss": round(float(loss), 5)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": ema, "state_dict": net.state_dict(),
                "chunk_dim": chunk_dim, "cond_dim": cond_dim, "action_dim": A_dim, "H": H,
                "obs_hist": HH, "latent_dim": L, "hidden": args.hidden, "n_blocks": args.n_blocks,
                "diffusion_steps": args.diffusion_steps, "objective": args.objective,
                "raw_obs": bool(args.raw_obs), "concat_raw": bool(args.concat_raw),
                "progress_cond": bool(args.progress_cond), "cfg_dropout": float(args.cfg_dropout),
                "subtask_cond": bool(args.subtask_cond)}, args.out)
    print(json.dumps({"event": "diff_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
