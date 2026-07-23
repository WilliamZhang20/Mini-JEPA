"""Train the goal-frame equivariant event HWM for FetchSlide."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.algos.world_models.ballistic import (
    EquivariantBallisticHWM,
    canonical_ballistic_features,
    world_to_goal_frame,
)
from jepa_robotics.evaluate import load_jepa_artifact


def encode(world_model, normalizer, states, device, *, target=False, batch=8192):
    encoded = []
    fn = world_model.encode_target if target else world_model.encode
    with torch.no_grad():
        for lo in range(0, len(states), batch):
            value = torch.from_numpy(normalizer.encode(states[lo : lo + batch])).to(device)
            encoded.append(fn(value).cpu())
    return torch.cat(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trials-npz", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init-path", type=Path, default=None,
                        help="Warm-start a same-shape equivariant HWM for self-supervised on-policy calibration.")
    parser.add_argument("--train-steps", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=640)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--endpoint-weight", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--distance-bins", type=int, default=10,
                        help="Sample endpoint-distance quantile bins uniformly so rare accurate strikes are not drowned out.")
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    world_model, normalizer, spec, cfg = load_jepa_artifact(args.model_path, device)
    world_model.eval()
    datasets = [np.load(path) for path in args.trials_npz]
    pre = np.concatenate([np.asarray(data["pre_states"], np.float32) for data in datasets])
    final = np.concatenate([np.asarray(data["final_states"], np.float32) for data in datasets])
    macros_np = np.concatenate([np.asarray(data["macros"], np.float32) for data in datasets])
    final_distance = np.concatenate([
        np.asarray(data["final_distances"], np.float32)
        if "final_distances" in data.files else
        np.linalg.norm(
            np.asarray(data["final_states"], np.float32)[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
            - np.asarray(data["final_states"], np.float32)[
                :, spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim
            ],
            axis=-1,
        )
        for data in datasets
    ])

    z = encode(world_model, normalizer, pre, device)
    z_target = encode(world_model, normalizer, final, device, target=True)
    obj = pre[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
    final_obj = final[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
    target_displacement = world_to_goal_frame(
        final_obj - obj, pre, spec.obs_dim, spec.goal_dim
    )
    features_np = canonical_ballistic_features(
        pre, macros_np, spec.obs_dim, spec.goal_dim
    )

    order = rng.permutation(len(pre))
    n_val = max(1, int(round(len(order) * args.heldout_fraction)))
    val_np, train_np = order[:n_val], order[n_val:]
    initial = (
        torch.load(args.init_path, map_location=device, weights_only=False)
        if args.init_path is not None else None
    )
    if initial is not None:
        feature_mean = np.asarray(initial["feature_mean"], np.float32)
        feature_std = np.asarray(initial["feature_std"], np.float32)
    else:
        feature_mean = features_np[train_np].mean(axis=0)
        feature_std = features_np[train_np].std(axis=0).clip(1e-4)
    features_np = (features_np - feature_mean) / feature_std

    # Quantile bins are geometric supervision, not reward labels. Uniform bin
    # sampling gives the low-distance decision boundary equal representation.
    quantiles = np.quantile(final_distance[train_np], np.linspace(0, 1, args.distance_bins + 1))
    bin_id = np.clip(np.digitize(final_distance, quantiles[1:-1]), 0, args.distance_bins - 1)
    train_bins = [
        torch.from_numpy(train_np[bin_id[train_np] == index]).to(device)
        for index in range(args.distance_bins)
    ]
    train_bins = [indices for indices in train_bins if len(indices)]

    z = z.to(device)
    z_target = z_target.to(device)
    features = torch.from_numpy(features_np).to(device)
    target_displacement = torch.from_numpy(target_displacement).to(device)
    val_idx = torch.from_numpy(val_np).to(device)
    model = EquivariantBallisticHWM(
        int(cfg["latent_dim"]),
        int(features.shape[1]),
        hidden=args.hidden,
        n_heads=args.heads,
        n_blocks=args.blocks,
    ).to(device)
    if initial is not None:
        if initial.get("architecture") != "equivariant_v2":
            raise ValueError("--init-path must be an equivariant_v2 checkpoint")
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in range(1, args.train_steps + 1):
        per_bin = max(1, args.batch_size // len(train_bins))
        idx = torch.cat([
            indices[torch.randint(0, len(indices), (per_bin,), device=device)]
            for indices in train_bins
        ])
        if len(idx) < args.batch_size:
            pool = train_bins[torch.randint(0, len(train_bins), ()).item()]
            idx = torch.cat([idx, pool[torch.randint(0, len(pool), (args.batch_size - len(idx),), device=device)]])
        idx = idx[: args.batch_size]
        pred_z, pred_displacement = model(z[idx], features[idx])
        keep = (torch.rand(len(idx), args.heads, device=device) < 0.8).float()
        latent_error = nn.functional.smooth_l1_loss(
            pred_z, z_target[idx, None].expand_as(pred_z), reduction="none"
        ).mean(dim=-1)
        endpoint_error = nn.functional.smooth_l1_loss(
            pred_displacement,
            target_displacement[idx, None].expand_as(pred_displacement),
            reduction="none",
            beta=0.02,
        ).mean(dim=-1)
        loss = ((latent_error + args.endpoint_weight * endpoint_error) * keep).sum()
        loss = loss / keep.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10_000 == 0:
            print(json.dumps({
                "event": "equivariant_ballistic_train",
                "step": step,
                "loss": float(loss.detach().cpu()),
            }), flush=True)

    with torch.no_grad():
        pred_z, pred_displacement = model(z[val_idx], features[val_idx])
        endpoint_error = torch.linalg.norm(
            pred_displacement.mean(dim=1) - target_displacement[val_idx], dim=-1
        )
        disagreement = pred_displacement.std(dim=1).norm(dim=-1)
        latent_mse = (pred_z.mean(dim=1) - z_target[val_idx]).square().mean()
    metrics = {
        "endpoint_mae": float(endpoint_error.mean().cpu()),
        "endpoint_p90": float(torch.quantile(endpoint_error, 0.9).cpu()),
        "disagreement_mean": float(disagreement.mean().cpu()),
        "latent_mse": float(latent_mse.cpu()),
        "train_trials": int(len(train_np)),
        "val_trials": int(len(val_np)),
    }
    print(json.dumps({"event": "equivariant_ballistic_validation", **metrics}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "equivariant_v2",
        "state_dict": model.state_dict(),
        "latent_dim": int(cfg["latent_dim"]),
        "feature_dim": int(features.shape[1]),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "hidden": int(args.hidden),
        "heads": int(args.heads),
        "blocks": int(args.blocks),
        "endpoint_weight": float(args.endpoint_weight),
        "model_path": str(args.model_path),
        "trials_npz": [str(path) for path in args.trials_npz],
        "validation": metrics,
        "max_duration": int(datasets[0]["max_duration"]),
        "distance_bins": int(args.distance_bins),
        "seed": int(args.seed),
        "init_path": None if args.init_path is None else str(args.init_path),
    }, args.out)
    print(json.dumps({"event": "equivariant_ballistic_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
