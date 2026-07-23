"""Train an event-conditioned ballistic HWM on FetchSlide strike trials."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.algos.world_models.ballistic import BallisticHWM
from jepa_robotics.evaluate import load_jepa_artifact


def encode(wm, norm, states, device, target=False, batch=8192):
    out = []
    fn = wm.encode_target if target else wm.encode
    with torch.no_grad():
        for lo in range(0, len(states), batch):
            s = torch.from_numpy(norm.encode(states[lo : lo + batch])).to(device)
            out.append(fn(s).cpu())
    return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--trials-npz", type=Path, nargs="+", required=True,
                   help="One or more macro-trial datasets; repeat a path to upweight on-policy calibration data.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init-path", type=Path, default=None,
                   help="Warm-start a same-shape ballistic HWM for on-policy calibration rounds.")
    p.add_argument("--train-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=5)
    p.add_argument("--concat-raw", action="store_true",
                   help="Append normalized pre-impact state for precise contact geometry.")
    p.add_argument("--endpoint-weight", type=float, default=20.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--heldout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=131)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, device)
    wm.eval()
    datasets = [np.load(path) for path in args.trials_npz]
    pre = np.concatenate([np.asarray(data["pre_states"], np.float32) for data in datasets])
    final = np.concatenate([np.asarray(data["final_states"], np.float32) for data in datasets])
    macro = torch.from_numpy(np.concatenate([np.asarray(data["macros"], np.float32) for data in datasets]))
    z = encode(wm, norm, pre, device)
    z_target = encode(wm, norm, final, device, target=True)
    obj = pre[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
    final_obj = final[:, spec.obs_dim : spec.obs_dim + spec.goal_dim]
    displacement = torch.from_numpy((final_obj - obj).astype(np.float32))
    side = torch.from_numpy(norm.encode(pre).astype(np.float32)) if args.concat_raw else None
    order = rng.permutation(len(pre))
    n_val = max(1, int(round(len(order) * args.heldout_fraction)))
    val_idx = torch.from_numpy(order[:n_val])
    train_idx = torch.from_numpy(order[n_val:])

    side_dim = int(side.shape[1]) if side is not None else 0
    model = BallisticHWM(int(cfg["latent_dim"]), macro.shape[1], args.hidden,
                         args.heads, side_dim=side_dim).to(device)
    if args.init_path is not None:
        init = torch.load(args.init_path, map_location=device, weights_only=False)
        model.load_state_dict(init["state_dict"])
    z, z_target, macro, displacement = [x.to(device) for x in (z, z_target, macro, displacement)]
    if side is not None:
        side = side.to(device)
    train_idx, val_idx = train_idx.to(device), val_idx.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(1, args.train_steps + 1):
        idx = train_idx[torch.randint(0, len(train_idx), (args.batch_size,), device=device)]
        pred_z, pred_disp = model(z[idx], macro[idx], None if side is None else side[idx])
        # Bootstrap heads so disagreement remains meaningful off support.
        keep = (torch.rand(len(idx), args.heads, device=device) < 0.8).float()
        latent_err = (pred_z - z_target[idx, None]).square().mean(dim=-1)
        endpoint_err = (pred_disp - displacement[idx, None]).square().mean(dim=-1)
        loss = ((latent_err + args.endpoint_weight * endpoint_err) * keep).sum() / keep.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 10_000 == 0:
            print(json.dumps({"event": "ballistic_hwm_train", "step": step,
                              "loss": float(loss.detach().cpu())}), flush=True)

    with torch.no_grad():
        pred_z, pred_disp = model(z[val_idx], macro[val_idx], None if side is None else side[val_idx])
        mean_disp = pred_disp.mean(dim=1)
        endpoint_error = torch.linalg.norm(mean_disp - displacement[val_idx], dim=-1)
        disagreement = pred_disp.std(dim=1).norm(dim=-1)
        latent_mse = (pred_z.mean(dim=1) - z_target[val_idx]).square().mean()
    metrics = {
        "endpoint_mae": float(endpoint_error.mean().cpu()),
        "endpoint_p90": float(torch.quantile(endpoint_error, 0.9).cpu()),
        "disagreement_mean": float(disagreement.mean().cpu()),
        "latent_mse": float(latent_mse.cpu()),
        "train_trials": int(len(train_idx)),
        "val_trials": int(len(val_idx)),
    }
    print(json.dumps({"event": "ballistic_hwm_validation", **metrics}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "latent_dim": int(cfg["latent_dim"]),
        "macro_dim": int(macro.shape[1]), "hidden": args.hidden, "heads": args.heads,
        "side_dim": side_dim, "concat_raw": bool(args.concat_raw),
        "endpoint_weight": args.endpoint_weight, "model_path": str(args.model_path),
        "trials_npz": [str(path) for path in args.trials_npz], "validation": metrics,
        "max_duration": int(datasets[0]["max_duration"]),
        "init_path": None if args.init_path is None else str(args.init_path),
    }, args.out)
    print(json.dumps({"event": "ballistic_hwm_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
