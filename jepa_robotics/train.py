from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import JEPATrajectoryDataset, collect_episodes, fit_normalizer
from .envs import ObsSpec, goal_state_from_state, make_env
from .models import ActionConditionedJEPA, normalized_mse, variance_regularizer
from .tasks import resolve_task, task_dir


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item) for item in value.split(",") if item.strip()})
    if not horizons or min(horizons) < 1:
        raise argparse.ArgumentTypeError("horizons must be positive integers, e.g. 1,2,4,8")
    return horizons


def build_goal_states(raw_state: torch.Tensor, normalizer, spec: ObsSpec) -> torch.Tensor:
    raw_goal = raw_state.clone()
    desired_start = spec.obs_dim + spec.goal_dim
    desired = raw_goal[:, desired_start : desired_start + spec.goal_dim]
    raw_goal[:, spec.obs_dim : spec.obs_dim + spec.goal_dim] = desired
    if spec.obs_dim >= spec.goal_dim:
        raw_goal[:, : spec.goal_dim] = desired
    return normalizer.encode_tensor(raw_goal)


def compute_loss(
    model: ActionConditionedJEPA,
    batch: dict[str, torch.Tensor],
    *,
    horizons: list[int],
    normalizer,
    spec: ObsSpec,
    weights: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    state = batch["state"].to(device)
    raw_state = batch["raw_state"].to(device)
    actions = batch["actions"].to(device)
    future_states = batch["future_states"].to(device)
    raw_future_states = batch["raw_future_states"].to(device)
    distance = batch["distance"].to(device).unsqueeze(-1)

    z = model.encode(state)
    pred_losses = []
    pred_probe_losses = []
    pred_goal_losses = []
    pred_achieved_losses = []
    goal_state = build_goal_states(raw_state, normalizer, spec)
    with torch.no_grad():
        goal_z = model.encode_target(goal_state)
    for i, horizon in enumerate(horizons):
        pred_z = model.predict(z, actions, horizon)
        with torch.no_grad():
            target_z = model.encode_target(future_states[:, i])
        pred_losses.append(normalized_mse(pred_z, target_z))

        pred_state = model.state_probe(pred_z)
        pred_probe_losses.append(F.mse_loss(pred_state, future_states[:, i]))
        future_achieved = raw_future_states[:, i, spec.obs_dim : spec.obs_dim + spec.goal_dim]
        desired_start = spec.obs_dim + spec.goal_dim
        future_desired = raw_future_states[:, i, desired_start : desired_start + spec.goal_dim]
        future_distance = torch.linalg.norm(future_achieved - future_desired, dim=-1)
        if future_distance.max() > 1e-6:
            future_distance = future_distance / future_distance.max().detach()
        pred_goal_distance = 1.0 - F.cosine_similarity(pred_z, goal_z, dim=-1, eps=1e-6)
        pred_goal_losses.append(F.smooth_l1_loss(pred_goal_distance, future_distance))
        if spec.is_goal_env and spec.goal_dim > 0:
            pred_raw_state = normalizer.decode_tensor(pred_state)
            pred_achieved = pred_raw_state[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
            pred_achieved_losses.append(F.smooth_l1_loss(pred_achieved, future_achieved))

    loss_pred = torch.stack(pred_losses).mean()
    loss_pred_probe = torch.stack(pred_probe_losses).mean()
    loss_pred_goal = torch.stack(pred_goal_losses).mean()
    if pred_achieved_losses:
        loss_pred_achieved = torch.stack(pred_achieved_losses).mean()
    else:
        loss_pred_achieved = torch.zeros((), dtype=state.dtype, device=device)
    loss_var = variance_regularizer(z)
    current_state_probe = model.state_probe(z)
    loss_probe = F.mse_loss(current_state_probe, state)
    loss_distance = F.smooth_l1_loss(model.distance_probe(z), distance)
    if spec.is_goal_env and spec.goal_dim > 0:
        current_raw_probe = normalizer.decode_tensor(current_state_probe)
        current_achieved = raw_state[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
        loss_achieved = F.smooth_l1_loss(
            current_raw_probe[:, spec.obs_dim : spec.obs_dim + spec.goal_dim],
            current_achieved,
        )
    else:
        loss_achieved = torch.zeros((), dtype=state.dtype, device=device)

    latent_goal_distance = 1.0 - F.cosine_similarity(z, goal_z, dim=-1, eps=1e-6)
    physical_goal_distance = distance.squeeze(-1)
    if physical_goal_distance.max() > 1e-6:
        physical_goal_distance = physical_goal_distance / physical_goal_distance.max().detach()
    loss_goal_geometry = F.smooth_l1_loss(latent_goal_distance, physical_goal_distance)

    total = (
        loss_pred
        + weights["var"] * loss_var
        + weights["probe"] * loss_probe
        + weights["achieved"] * loss_achieved
        + weights["distance"] * loss_distance
        + weights["goal"] * loss_goal_geometry
        + weights["pred_probe"] * loss_pred_probe
        + weights["pred_achieved"] * loss_pred_achieved
        + weights["pred_goal"] * loss_pred_goal
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "pred": float(loss_pred.detach().cpu()),
        "var": float(loss_var.detach().cpu()),
        "probe": float(loss_probe.detach().cpu()),
        "achieved": float(loss_achieved.detach().cpu()),
        "distance": float(loss_distance.detach().cpu()),
        "goal": float(loss_goal_geometry.detach().cpu()),
        "pred_probe": float(loss_pred_probe.detach().cpu()),
        "pred_achieved": float(loss_pred_achieved.detach().cpu()),
        "pred_goal": float(loss_pred_goal.detach().cpu()),
    }
    return total, metrics


@torch.no_grad()
def evaluate_mpc(
    model: ActionConditionedJEPA,
    env,
    *,
    normalizer,
    spec: ObsSpec,
    device: torch.device,
    episodes: int,
    candidates: int,
    horizon: int,
    seed: int,
) -> dict[str, float]:
    successes = []
    final_distances = []
    rng = np.random.default_rng(seed)
    model.eval()
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = truncated = False
        final_info = {}
        while not (terminated or truncated):
            raw_state = np.concatenate(
                [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]
            ).astype(np.float32)
            state = torch.from_numpy(normalizer.encode(raw_state)).unsqueeze(0).to(device)
            z = model.encode(state).repeat(candidates, 1)

            low = env.action_space.low.astype(np.float32)
            high = env.action_space.high.astype(np.float32)
            action_seq = rng.uniform(low, high, size=(candidates, horizon, spec.action_dim)).astype(
                np.float32
            )
            action_tensor = torch.from_numpy(action_seq).to(device)
            pred_z = model.predict(z, action_tensor, horizon)

            goal_state = goal_state_from_state(raw_state, spec)
            goal_norm = torch.from_numpy(normalizer.encode(goal_state)).unsqueeze(0).to(device)
            goal_z = model.encode_target(goal_norm)
            scores = torch.sum((F.normalize(pred_z, dim=-1) - F.normalize(goal_z, dim=-1)) ** 2, dim=-1)
            best = int(torch.argmin(scores).cpu())
            obs, _, terminated, truncated, final_info = env.step(action_seq[best, 0])

        successes.append(float(final_info.get("is_success", 0.0)))
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired = np.asarray(obs["desired_goal"], dtype=np.float32)
        final_distances.append(float(np.linalg.norm(achieved - desired)))

    model.train()
    return {
        "mpc_success": float(np.mean(successes)) if successes else 0.0,
        "mpc_final_distance": float(np.mean(final_distances)) if final_distances else float("nan"),
    }


def save_checkpoint(path: Path, model, optimizer, normalizer, spec, args, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
            "spec": spec.__dict__,
            "args": vars(args),
            "step": step,
        },
        path,
    )


def save_model_artifact(path: Path, model, normalizer, spec, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
            "spec": spec.__dict__,
            "config": {
                "task": args.task,
                "env_id": args.env_id,
                "horizons": args.horizons,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "max_horizon": max(args.horizons),
                "predictor_mode": args.predictor_mode,
                "residual_prediction": args.residual_prediction,
            },
        },
        path,
    )


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small action-conditioned JEPA on Gymnasium Robotics.")
    parser.add_argument("--task", default=None, choices=["fetch_reach", "fetch_pick_place", "adroit_door"])
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--collect-steps", type=int, default=100_000)
    parser.add_argument("--collect-log-every", type=int, default=10_000)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizons", type=parse_horizons, default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--predictor-mode", choices=["direct", "rollout"], default="direct")
    parser.add_argument("--residual-prediction", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ema", type=float, default=0.995)
    parser.add_argument("--scripted-fraction", type=float, default=0.6)
    parser.add_argument("--controller-gain", type=float, default=5.0)
    parser.add_argument("--action-noise", type=float, default=0.15)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--lambda-var", type=float, default=0.02)
    parser.add_argument("--lambda-probe", type=float, default=0.05)
    parser.add_argument("--lambda-achieved", type=float, default=0.0)
    parser.add_argument("--lambda-distance", type=float, default=0.1)
    parser.add_argument("--lambda-goal", type=float, default=0.05)
    parser.add_argument("--lambda-pred-probe", type=float, default=0.1)
    parser.add_argument("--lambda-pred-achieved", type=float, default=0.0)
    parser.add_argument("--lambda-pred-goal", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--mpc-candidates", type=int, default=128)
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Final model-only artifact written after training finishes.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Use tiny settings for compile/runtime checks.")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.collect_steps = 80
    args.train_steps = 3
    args.batch_size = 16
    args.latent_dim = 32
    args.hidden_dim = 64
    args.max_episode_steps = 20
    args.eval_episodes = 1
    args.mpc_candidates = 8
    args.mpc_horizon = min(max(args.horizons), 4)
    args.log_every = 1


def main() -> None:
    parser = make_argparser()
    args = parser.parse_args()
    task = resolve_task(args.task, args.env_id)
    args.task = task.name
    args.env_id = task.env_id
    if args.horizons is None:
        args.horizons = parse_horizons(task.horizons)
    if args.max_episode_steps is None:
        args.max_episode_steps = task.max_episode_steps
    out_dir = task_dir(args.output_root, task)
    if args.save_path is None:
        args.save_path = out_dir / "checkpoints" / f"{task.slug}_jepa_checkpoint.pt"
    if args.model_path is None:
        args.model_path = out_dir / "checkpoints" / f"{task.slug}_jepa_model.pt"
    if args.smoke:
        apply_smoke_overrides(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(json.dumps({"event": "config", **vars(args), "device": str(device)}, default=str))

    env = make_env(args.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    episodes, spec = collect_episodes(
        env,
        num_steps=args.collect_steps,
        seed=args.seed,
        scripted_fraction=args.scripted_fraction,
        controller_gain=args.controller_gain,
        action_noise=args.action_noise,
        controller=task.controller,
        log_every=args.collect_log_every,
    )
    normalizer = fit_normalizer(episodes)
    dataset = JEPATrajectoryDataset(episodes, normalizer, spec, args.horizons)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = ActionConditionedJEPA(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_horizon=max(args.horizons),
        predictor_mode=args.predictor_mode,
        residual_prediction=args.residual_prediction,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = {
        "var": args.lambda_var,
        "probe": args.lambda_probe,
        "achieved": args.lambda_achieved,
        "distance": args.lambda_distance,
        "goal": args.lambda_goal,
        "pred_probe": args.lambda_pred_probe,
        "pred_achieved": args.lambda_pred_achieved,
        "pred_goal": args.lambda_pred_goal,
    }

    step = 0
    while step < args.train_steps:
        for batch in loader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(
                model,
                batch,
                horizons=args.horizons,
                normalizer=normalizer,
                spec=spec,
                weights=weights,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target(args.ema)
            if step == 1 or step % args.log_every == 0 or step >= args.train_steps:
                print(json.dumps({"event": "train", "step": step, **metrics}))
            if step >= args.train_steps:
                break

    eval_env = make_env(args.env_id, seed=args.seed + 10_000, max_episode_steps=args.max_episode_steps)
    eval_metrics = evaluate_mpc(
        model,
        eval_env,
        normalizer=normalizer,
        spec=spec,
        device=device,
        episodes=args.eval_episodes,
        candidates=args.mpc_candidates,
        horizon=args.mpc_horizon,
        seed=args.seed + 20_000,
    )
    print(json.dumps({"event": "eval", **eval_metrics}))
    save_checkpoint(args.save_path, model, optimizer, normalizer, spec, args, step)
    print(json.dumps({"event": "saved", "path": str(args.save_path)}))
    save_model_artifact(args.model_path, model, normalizer, spec, args)
    print(json.dumps({"event": "saved_model", "path": str(args.model_path)}))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
