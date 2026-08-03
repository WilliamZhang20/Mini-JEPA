"""Train an action-conditioned JEPA world model for latent planning.

Thin CLI over ``jepa_robotics.algos.world_models.latent_jepa``. Unlike
``jepa_robotics.train`` (which supervises only the listed horizons and targets
goal-env geometry), this trains the dense multi-step rollout objective a
planner needs, and it accepts several episode sources so on-policy trial data
can be mixed into the demonstration set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.world_models.latent_jepa import (
    RolloutWindows,
    dense_rollout_loss,
    rollout_accuracy,
)
from jepa_robotics.data import Normalizer, fit_normalizer, load_episodes_npz, load_spec_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.models import ActionConditionedJEPA
from jepa_robotics.tasks import resolve_task, task_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes-npz", type=Path, nargs="+", required=True,
                   help="One or more episode sources; demos and on-policy trials are concatenated.")
    p.add_argument("--max-episodes", type=int, nargs="+", default=None,
                   help="Per-source episode cap, in the same order as --episodes-npz.")
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=16, help="Dense rollout length used for training.")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--transition-depth", type=int, default=2)
    p.add_argument("--ensemble-heads", type=int, default=4)
    p.add_argument("--inverse-horizon", type=int, default=4)
    p.add_argument("--no-latent-norm", action="store_true")
    p.add_argument("--shared-action-encoder", action="store_true",
                   help="Share one action encoder across ensemble heads (the old layout). "
                        "Default gives each head its own, so heads can disagree about "
                        "unfamiliar ACTIONS -- the disagreement a planner needs.")
    p.add_argument("--train-steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--bootstrap", type=float, default=0.8)
    p.add_argument("--lambda-pred-probe", type=float, default=0.5)
    p.add_argument("--lambda-probe", type=float, default=0.1)
    p.add_argument("--lambda-var", type=float, default=0.02)
    p.add_argument("--lambda-cov", type=float, default=0.005)
    p.add_argument("--lambda-inverse", type=float, default=0.5)
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--resume-path", type=Path, default=None,
                   help="Warm-start from an existing artifact of identical architecture.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    task = resolve_task(args.task, None)

    caps = args.max_episodes or [None] * len(args.episodes_npz)
    if len(caps) != len(args.episodes_npz):
        raise SystemExit("--max-episodes must have one entry per --episodes-npz")
    episodes, spec = [], None
    for path, cap in zip(args.episodes_npz, caps):
        loaded = load_episodes_npz(path)
        if cap is not None:
            loaded = loaded[: int(cap)]
        episodes.extend(loaded)
        spec = spec or load_spec_npz(path)
        print(json.dumps({"event": "source", "path": str(path), "episodes": len(loaded)}), flush=True)
    if spec is None:
        env = make_env(task.env_id, seed=args.seed)
        spec = obs_spec_from_env(env)
        env.close()

    order = rng.permutation(len(episodes))
    n_hold = max(1, int(len(episodes) * args.holdout_frac))
    hold_eps = [episodes[i] for i in order[:n_hold]]
    train_eps = [episodes[i] for i in order[n_hold:]]

    resume = torch.load(args.resume_path, map_location="cpu", weights_only=False) if args.resume_path else None
    if resume is not None and "normalizer" in resume:
        normalizer = Normalizer(
            mean=np.asarray(resume["normalizer"]["mean"], dtype=np.float32),
            std=np.asarray(resume["normalizer"]["std"], dtype=np.float32),
        )
    else:
        normalizer = fit_normalizer(train_eps)

    train_w = RolloutWindows.from_episodes(train_eps, normalizer, args.horizon, device)
    hold_w = RolloutWindows.from_episodes(hold_eps, normalizer, args.horizon, device)
    print(json.dumps({
        "event": "windows", "train": len(train_w), "holdout": len(hold_w),
        "state_dim": int(spec.state_dim), "action_dim": int(spec.action_dim),
    }), flush=True)

    model = ActionConditionedJEPA(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_horizon=args.horizon,
        predictor_mode="recurrent",
        transition_depth=args.transition_depth,
        ensemble_heads=args.ensemble_heads,
        inverse_dynamics=True,
        inverse_horizon=args.inverse_horizon,
        latent_norm=not args.no_latent_norm,
        per_head_action_encoder=not args.shared_action_encoder,
    ).to(device)
    if resume is not None:
        model.load_state_dict(resume["model"])
        print(json.dumps({"event": "resumed", "path": str(args.resume_path)}), flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"event": "model", "params": int(n_params)}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.train_steps, eta_min=args.lr * 0.05)
    weights = {
        "pred_probe": args.lambda_pred_probe,
        "probe": args.lambda_probe,
        "var": args.lambda_var,
        "cov": args.lambda_cov,
        "inverse": args.lambda_inverse,
    }
    gen = torch.Generator(device=device).manual_seed(args.seed)

    config = {
        "task": task.name,
        "env_id": task.env_id,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "max_horizon": args.horizon,
        "predictor_mode": "recurrent",
        "residual_prediction": False,
        "transition_depth": args.transition_depth,
        "ensemble_heads": args.ensemble_heads,
        "inverse_dynamics": True,
        "inverse_horizon": args.inverse_horizon,
        "latent_norm": not args.no_latent_norm,
        "per_head_action_encoder": not args.shared_action_encoder,
        "trainer": "dense_rollout",
        "sources": [str(x) for x in args.episodes_npz],
    }
    out_path = args.model_path or (
        task_dir(args.output_root, task) / "checkpoints" / f"{task.slug}_jepa_planner_model.pt"
    )

    def save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
            "spec": spec.__dict__,
            "config": config,
        }, out_path)

    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        state, actions, futures = train_w.sample(args.batch_size, gen)
        loss, metrics = dense_rollout_loss(
            model, state, actions, futures,
            weights=weights, bootstrap=args.bootstrap, generator=gen,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        model.update_target(args.ema)
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            acc = rollout_accuracy(model, hold_w, samples=4096, seed=args.seed)
            print(json.dumps({
                "event": "train", "step": step, **metrics,
                "holdout_ratio_static": acc["ratio_static"],
                "holdout_state_rmse": acc["state_rmse"],
                "lr": float(sched.get_last_lr()[0]),
                "elapsed_s": round(time.time() - t0, 1),
            }), flush=True)
            save()

    final = rollout_accuracy(model, hold_w, samples=8192, seed=args.seed + 1)
    save()
    print(json.dumps({"event": "saved", "path": str(out_path), "accuracy": final}), flush=True)


if __name__ == "__main__":
    main()
