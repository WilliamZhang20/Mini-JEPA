"""Train a JEPA-latent Kitchen subtask-completion probe.

The labels are recovered from demonstration progression rather than from a
runtime reward or environment oracle.  ``label_kitchen_subtasks.py`` assigns
each transition to the next subtask that the demonstration completes.  Once a
task's final assigned transition has passed, that task is monotonically marked
complete for the rest of the trajectory.

At evaluation this probe replaces ``info['step_task_completions']`` as the
specialist-switch predicate.  Environment completion is still allowed for
benchmark scoring, but is not fed to the controller.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.algos.control.completion import LatentCompletionProbe
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.algos.task_families.kitchen import KITCHEN_TASKS


def progression_labels(targets: np.ndarray, state_count: int, num_tasks: int) -> np.ndarray:
    """Infer monotone completion labels from next-completed-task annotations."""
    targets = np.asarray(targets, dtype=np.float32)
    n = min(len(targets), state_count - 1)
    valid = targets[:n].sum(axis=-1) > 0
    task = np.where(valid, targets[:n].argmax(axis=-1), -1)
    labels = np.zeros((state_count, num_tasks), dtype=np.float32)
    for task_id in range(num_tasks):
        idx = np.flatnonzero(task == task_id)
        if len(idx):
            # Action idx[-1] is the completion transition; state idx[-1] + 1
            # is the first observation in which the predicate is satisfied.
            labels[min(int(idx[-1]) + 1, state_count - 1) :, task_id] = 1.0
    return labels


def encode_states(wm, norm, states: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    latents = []
    with torch.no_grad():
        for lo in range(0, len(states), batch_size):
            s = torch.from_numpy(norm.encode(states[lo : lo + batch_size])).to(device)
            latents.append(wm.encode(s).cpu())
    return torch.cat(latents, dim=0)


def precision_recall(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = prob >= thresholds[None]
    truth = y > 0.5
    tp = (pred & truth).sum(axis=0)
    fp = (pred & ~truth).sum(axis=0)
    fn = (~pred & truth).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    return precision.astype(np.float32), recall.astype(np.float32)


def calibrate_thresholds(y: np.ndarray, prob: np.ndarray, min_precision: float) -> np.ndarray:
    """Choose the highest-recall validation threshold meeting precision target."""
    thresholds = []
    grid = np.linspace(0.50, 0.999, 500, dtype=np.float32)
    for task_id in range(y.shape[1]):
        best_t, best_recall = float(grid[-1]), -1.0
        fallback_t, fallback_precision, fallback_recall = float(grid[-1]), -1.0, -1.0
        truth = y[:, task_id] > 0.5
        for threshold in grid:
            pred = prob[:, task_id] >= threshold
            tp = int((pred & truth).sum())
            fp = int((pred & ~truth).sum())
            fn = int((~pred & truth).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if (precision, recall) > (fallback_precision, fallback_recall):
                fallback_t, fallback_precision, fallback_recall = (
                    float(threshold), precision, recall
                )
            if precision >= min_precision and recall > best_recall:
                best_t, best_recall = float(threshold), recall
        if best_recall < 0:
            # Some labels are intrinsically ambiguous at a frame boundary. Do
            # not silently fall back to an arbitrary 0.99; select the safest
            # observed operating point and report its actual precision.
            best_t = fallback_t
        thresholds.append(best_t)
    return np.asarray(thresholds, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--labeled-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--encode-batch-size", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--concat-raw", action="store_true",
                   help="Append normalized live state to z. Useful when the frozen latent loses fine contact-threshold detail.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--heldout-fraction", type=float, default=0.2)
    p.add_argument("--min-precision", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wm, norm, _spec, cfg = load_jepa_artifact(args.model_path, device)
    wm.eval()
    for parameter in wm.parameters():
        parameter.requires_grad_(False)

    data = np.load(args.labeled_npz, allow_pickle=True)
    task_names = (
        [str(name) for name in data["task_names"].tolist()]
        if "task_names" in data.files else list(KITCHEN_TASKS)
    )
    states_eps = data["states"]
    targets_eps = data["targets"]
    completion_eps = data["completions"] if "completions" in data.files else None
    order = np.random.default_rng(args.seed).permutation(len(states_eps))
    n_val = max(1, int(round(len(order) * args.heldout_fraction)))
    split = {
        "train": order[n_val:],
        "val": order[:n_val],
    }
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, ep_ids in split.items():
        states, labels = [], []
        for ep_id in ep_ids:
            s = np.asarray(states_eps[ep_id], dtype=np.float32)
            states.append(s)
            if completion_eps is not None:
                labels.append(np.asarray(completion_eps[ep_id], dtype=np.float32)[: len(s)])
            else:
                labels.append(progression_labels(targets_eps[ep_id], len(s), len(task_names)))
        raw = np.concatenate(states, axis=0)
        y = torch.from_numpy(np.concatenate(labels, axis=0))
        features = encode_states(wm, norm, raw, device, args.encode_batch_size)
        if args.concat_raw:
            features = torch.cat([features, torch.from_numpy(norm.encode(raw))], dim=-1)
        tensors[name] = (features, y)
        print(json.dumps({"event": "completion_data", "split": name,
                          "episodes": int(len(ep_ids)), "states": int(len(y)),
                          "positive_rate": y.mean(dim=0).tolist()}), flush=True)

    z_train, y_train = (x.to(device) for x in tensors["train"])
    z_val, y_val = (x.to(device) for x in tensors["val"])
    input_dim = int(z_train.shape[-1])
    probe = LatentCompletionProbe(input_dim, len(task_names), args.hidden).to(device)
    positives = y_train.sum(dim=0)
    negatives = y_train.shape[0] - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 30.0)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, len(z_train), (args.batch_size,), device=device)
        logits = probe(z_train[idx])
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y_train[idx], pos_weight=pos_weight
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 5_000 == 0:
            print(json.dumps({"event": "completion_train", "step": step,
                              "loss": float(loss.detach().cpu())}), flush=True)

    with torch.no_grad():
        val_prob = torch.sigmoid(probe(z_val)).cpu().numpy()
    val_y = y_val.cpu().numpy()
    thresholds = calibrate_thresholds(val_y, val_prob, args.min_precision)
    precision, recall = precision_recall(val_y, val_prob, thresholds)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-8)
    metrics = {
        "thresholds": thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }
    print(json.dumps({"event": "completion_validation", **metrics}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": probe.state_dict(),
        "latent_dim": int(cfg["latent_dim"]),
        "input_dim": input_dim,
        "concat_raw": bool(args.concat_raw),
        "num_tasks": len(task_names),
        "hidden": int(args.hidden),
        "task_names": task_names,
        "thresholds": thresholds,
        "validation": metrics,
        "model_path": str(args.model_path),
        "labeled_npz": str(args.labeled_npz),
        "label_source": "demonstration_progression",
        "aligned_replay_labels": bool(completion_eps is not None),
        "seed": int(args.seed),
    }, args.out)
    print(json.dumps({"event": "completion_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
