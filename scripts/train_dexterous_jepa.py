"""Train the DexterousJEPA transformer world model for Shadow Hand tasks.

Self-supervised JEPA objective on offline demos:

    z_t      = encode(s_t)
    z_future = target_encoder(s_{t+h})            # EMA, stop-gradient
    pred     = predict(z_t, a_{t:t+h}, h)
    loss = normalized_mse(pred, sg[z_future])     # JEPA prediction
         + VICReg(z_t)                            # anti-collapse
         + state-probe MSE                        # decodable geometry
         + contact-consistency MSE                # DexWM-style fingertip/relative detail

Saves in the load_jepa_artifact format with ``config["arch"]="dexterous"`` so it
runs under every existing planner/eval. GPU-train in the GPU session:

    python scripts/train_dexterous_jepa.py --task adroit_relocate \
      --episodes-npz runs/adroit_relocate/data/relocate_expert_demos.npz \
      --out runs/adroit_relocate/checkpoints/relocate_dexterous_jepa.pt \
      --horizons 1,2,4,8,16 --latent-dim 192 --d-model 256 --enc-depth 4 \
      --dyn-depth 4 --heads 8 --ensemble-heads 3 --contact-dims 30,39 \
      --steps 60000 --batch-size 256 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import fit_normalizer, load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.models import DexterousJEPA, normalized_mse, variance_regularizer, covariance_regularizer
from jepa_robotics.tasks import resolve_task


def parse_ints(s):
    return [int(x) for x in s.split(",")]


def parse_groups(value):
    """Parse comma-separated anatomical token slices, e.g. ``61:69,69:76``."""
    groups = []
    for item in value.split(","):
        lo, hi = item.split(":")
        groups.append((int(lo), int(hi)))
    return groups


def quat_geodesic_loss(pred_q: torch.Tensor, true_q: torch.Tensor) -> torch.Tensor:
    """Smooth surrogate for the rotation angle between two (raw) quaternions.

    Normalizes each and uses ``1 - |q_a . q_b|`` — the ``abs`` handles the
    quaternion double cover (q and -q are the same orientation), which plain
    component-wise MSE in z-scored state space does not. Monotone in the geodesic
    angle, bounded, and gradient-stable near 0 (unlike ``2*acos``)."""
    pred_q = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    true_q = true_q / true_q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dot = (pred_q * true_q).sum(-1).abs().clamp(max=1.0)
    return (1.0 - dot).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizons", type=parse_ints, default=[1, 2, 4, 8, 16])
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--latent-dim", type=int, default=192)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=4)
    p.add_argument("--dyn-depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ensemble-heads", type=int, default=3)
    p.add_argument(
        "--latent-slots",
        type=int,
        default=1,
        help="Number of structured latent slots; >1 enables recurrent slot dynamics.",
    )
    p.add_argument(
        "--pose-relation",
        action="store_true",
        help="Add a raw goal-relative translation + SO(3) log-map token.",
    )
    p.add_argument("--contact-dims", default=None, help="lo,hi raw-state slice for the contact-consistency head (e.g. 30,39 for Relocate palm-ball+ball-target)")
    p.add_argument(
        "--token-groups",
        type=parse_groups,
        default=None,
        help="Anatomical state slices encoded as local tokens instead of one token per scalar.",
    )
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=0.5)
    p.add_argument("--lambda-state", type=float, default=1.0)
    p.add_argument("--lambda-contact", type=float, default=1.0)
    # Rollout-decode supervision + object-pose emphasis. Planning decodes
    # *predicted* rollout latents into an object pose; without decoding the
    # predicted (not just the encoder) latent, and without weighting the object
    # over the 24-DoF hand, the state probe is calibrated on the wrong latents and
    # the object quaternion (4 of 75 dims, tiny per-step motion) is swamped.
    p.add_argument("--object-dims", default=None,
                   help="lo,hi raw-state slice of the object pose (pos3+quat4), e.g. 61,68 (achieved_goal).")
    p.add_argument("--lambda-pred-state", type=float, default=1.0,
                   help="Decode the h-step predicted rollout latent back to its future state.")
    p.add_argument("--lambda-object", type=float, default=5.0,
                   help="Extra weight on the object pose in the rollout decode (pos MSE + geodesic quaternion).")
    p.add_argument(
        "--object-probe",
        action="store_true",
        help="Decode achieved pose from the first object-specialized latent slot.",
    )
    p.add_argument(
        "--lambda-object-probe",
        type=float,
        default=5.0,
        help="Weight for the dedicated object-slot rollout pose loss.",
    )
    p.add_argument("--init-model", type=Path, default=None,
                   help="Warm-start weights and reuse the normalizer for cumulative on-policy world-model calibration.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()
    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    init_ckpt = torch.load(args.init_model, map_location="cpu", weights_only=False) if args.init_model else None
    if init_ckpt is not None:  # reuse the init normalizer so encodings stay consistent across rounds
        from jepa_robotics.data import Normalizer
        norm = Normalizer(mean=np.asarray(init_ckpt["normalizer"]["mean"], np.float32),
                          std=np.asarray(init_ckpt["normalizer"]["std"], np.float32))
    else:
        norm = fit_normalizer(episodes)
    max_h = max(args.horizons)
    contact = tuple(parse_ints(args.contact_dims)) if args.contact_dims else None
    token_groups = tuple(args.token_groups or ())
    obj = tuple(parse_ints(args.object_dims)) if args.object_dims else None  # (pos3+quat4) raw slice
    mean_t = torch.as_tensor(norm.mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(norm.std, dtype=torch.float32, device=dev)

    # Precompute per-episode normalized states; sample (ep, t, h) transitions on the fly.
    ep_states = [norm.encode(ep.states.astype(np.float32)) for ep in episodes if len(ep.actions) > max_h]
    ep_actions = [ep.actions.astype(np.float32) for ep in episodes if len(ep.actions) > max_h]
    lengths = np.array([len(a) for a in ep_actions])
    ep_ids = np.arange(len(ep_actions))
    rng = np.random.default_rng(0)

    # Warm-start: rebuild the model with the CHECKPOINT's architecture (not the CLI
    # defaults) so the weights load — e.g. dyn_depth/latent_dim may differ.
    arch = dict(latent_dim=args.latent_dim, d_model=args.d_model, enc_depth=args.enc_depth,
                dyn_depth=args.dyn_depth, heads=args.heads, ensemble_heads=args.ensemble_heads,
                latent_slots=args.latent_slots)
    if init_ckpt is not None:
        c = init_ckpt["config"]
        for k in arch:
            if c.get(k) is not None:
                arch[k] = c[k]
        if c.get("token_groups") is not None:
            token_groups = tuple(tuple(group) for group in c["token_groups"])
        args.pose_relation = bool(c.get("pose_relation_dims"))
        args.object_probe = bool(c.get("object_probe", False))
    pose_relation_dims = (
        (spec.obs_dim, spec.obs_dim + spec.goal_dim) if args.pose_relation else None
    )
    wm = DexterousJEPA(
        state_dim=spec.state_dim, action_dim=spec.action_dim, max_horizon=max_h,
        contact_dims=contact, token_groups=token_groups, dropout=args.dropout, **arch,
        pose_relation_dims=pose_relation_dims,
        state_mean=torch.as_tensor(norm.mean, dtype=torch.float32),
        state_std=torch.as_tensor(norm.std, dtype=torch.float32),
        object_probe_dims=obj if args.object_probe else None,
    ).to(dev)
    if init_ckpt is not None:
        wm.load_state_dict(init_ckpt["model"])
        print(json.dumps({"event": "dex_jepa_warmstart", "from": str(args.init_model), "arch": arch}), flush=True)
    opt = torch.optim.AdamW(wm.parameters(), lr=args.lr, weight_decay=1e-4)
    nparams = sum(p.numel() for p in wm.parameters()) / 1e6
    print(json.dumps({"event": "dex_jepa_data", "episodes": len(ep_actions), "state_dim": spec.state_dim,
                      "action_dim": spec.action_dim, "params_M": round(nparams, 2),
                      "contact_dims": contact, "token_groups": token_groups,
                      "latent_slots": arch["latent_slots"],
                      "pose_relation_dims": pose_relation_dims,
                      "object_probe": bool(args.object_probe)}), flush=True)

    def sample_batch(h):
        cur, fut_seq, chunks = [], [], []
        for _ in range(args.batch_size):
            e = int(rng.choice(ep_ids))
            T = lengths[e]
            t = int(rng.integers(0, T - h))
            cur.append(ep_states[e][t])
            fs = ep_states[e][t + 1:t + 1 + h]                      # true s_{t+1..t+h}
            if len(fs) < max_h:
                fs = np.concatenate([fs, np.repeat(fs[-1:], max_h - len(fs), 0)], 0)
            fut_seq.append(fs)
            a = ep_actions[e][t:t + h]
            if len(a) < max_h:
                a = np.concatenate([a, np.zeros((max_h - len(a), spec.action_dim), np.float32)], 0)
            chunks.append(a)
        return (torch.from_numpy(np.stack(cur)).to(dev),
                torch.from_numpy(np.stack(fut_seq)).to(dev),          # [B, max_h, state]
                torch.from_numpy(np.stack(chunks)).to(dev))

    for step in range(1, args.steps + 1):
        h = int(rng.choice(args.horizons))
        s_t, fut_seq, chunks = sample_batch(h)
        B = s_t.shape[0]
        z = wm.encode(s_t)
        roll = wm.predict_rollout(z, chunks, h)                       # [B, h, latent] full trajectory
        # Endpoint latent JEPA (cheap: B encodes) keeps the latent well-formed...
        with torch.no_grad():
            target = wm.encode_target(fut_seq[:, h - 1])              # [B, latent]
        loss = normalized_mse(roll[:, -1], target)
        # ...and DENSE DECODE (below) anchors EVERY rollout step's object pose to
        # the truth — the fix for dynamics drift, and cheap (probe MLP only, no
        # extra encodes). true_seq is just data.
        true_seq = fut_seq[:, :h].reshape(B * h, -1)
        loss = loss + args.lambda_var * variance_regularizer(z) + args.lambda_cov * covariance_regularizer(z)
        loss = loss + args.lambda_state * torch.nn.functional.mse_loss(wm.state_probe(z), s_t)
        if contact is not None:
            lo, hi = contact
            loss = loss + args.lambda_contact * torch.nn.functional.mse_loss(wm.contact_consistency(z), s_t[:, lo:hi])
        # DENSE decode: decode every predicted rollout latent, object-pose weighted
        # with a geodesic quaternion loss, over the whole trajectory.
        roll_state = wm.state_probe(roll.reshape(B * h, -1))
        loss = loss + args.lambda_pred_state * torch.nn.functional.mse_loss(roll_state, true_seq)
        geo = torch.tensor(0.0, device=dev)
        if obj is not None:
            olo, ohi = obj
            pred_raw = roll_state * std_t + mean_t
            true_raw = true_seq * std_t + mean_t
            pos_mse = torch.nn.functional.mse_loss(pred_raw[:, olo:olo + 3], true_raw[:, olo:olo + 3])
            geo = quat_geodesic_loss(pred_raw[:, olo + 3:ohi], true_raw[:, olo + 3:ohi])
            loss = loss + args.lambda_object * (pos_mse + geo)
            if args.object_probe:
                object_norm = wm.predict_object(roll.reshape(B * h, -1))
                object_raw = object_norm * std_t[olo:ohi] + mean_t[olo:ohi]
                object_pos = torch.nn.functional.mse_loss(
                    object_raw[:, :3], true_raw[:, olo:olo + 3]
                )
                object_geo = quat_geodesic_loss(
                    object_raw[:, 3:], true_raw[:, olo + 3:ohi]
                )
                loss = loss + args.lambda_object_probe * (object_pos + object_geo)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        wm.update_target(args.ema)
        if step == 1 or step % 2000 == 0:
            log = {"event": "dex_jepa_train", "step": step, "h": h, "loss": round(float(loss.detach()), 4)}
            if obj is not None:
                log["obj_geo"] = round(float(geo.detach()), 4)  # ~rotation error surrogate on the predicted future
            print(json.dumps(log), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": wm.state_dict(),
        "normalizer": {"mean": norm.mean, "std": norm.std},
        "spec": spec.__dict__,
        "config": {
            "task": args.task, "env_id": task.env_id, "arch": "dexterous",
            "horizons": args.horizons, "latent_dim": arch["latent_dim"], "hidden_dim": arch["d_model"],
            "d_model": arch["d_model"], "enc_depth": arch["enc_depth"], "dyn_depth": arch["dyn_depth"],
            "heads": arch["heads"], "max_horizon": max_h, "ensemble_heads": arch["ensemble_heads"],
            "latent_slots": arch["latent_slots"],
            "contact_dims": list(contact) if contact else None,
            "token_groups": [list(group) for group in token_groups] if token_groups else None,
            "object_dims": list(obj) if obj else None,
            "pose_relation_dims": list(pose_relation_dims) if pose_relation_dims else None,
            "object_probe": bool(args.object_probe),
            "lambda_pred_state": args.lambda_pred_state, "lambda_object": args.lambda_object,
            "lambda_object_probe": args.lambda_object_probe,
        },
    }, args.out)
    print(json.dumps({"event": "dex_jepa_saved", "path": str(args.out), "params_M": round(nparams, 2)}), flush=True)


if __name__ == "__main__":
    main()
