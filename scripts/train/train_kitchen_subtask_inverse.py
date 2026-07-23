"""Train a per-subtask segment-pure inverse chunk specialist for FrankaKitchen.

This ports the Relocate possession-specialist recipe to the sequential Kitchen
task. Kitchen's failure mode is chaining 4 subtasks from sparse full-sequence
data: one global policy blurs 4 distinct action manifolds together and stalls
mid-sequence. The fix (mirroring Relocate law 1, "no blurring of the expert
action manifold") is to split the regime into segment-pure specialists -- ONE
inverse per subtask, trained ONLY on transitions the labeler tagged as working
toward that subtask.

Each specialist learns the self-supervised inverse map

    z_t = encoder(s_t)
    z_future = target_encoder(s_{t+h})
    inverse(z_t, z_future, h [, raw, emphasis]) -> a_{t:t+H-1}

with the current subtask's object qpos dims duplicated in the conditioning
(Relocate law 3, input-feature emphasis) so the chunk servos to the live object.
The demo *segments* for this subtask are saved in the checkpoint so eval can
build a per-subtask demo-locked future index (Relocate law 2, future coherence).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.algos.task_families.kitchen import KITCHEN_OBJ_DIMS, KITCHEN_TASKS
from jepa_robotics.algos.priors import InversePrior, parse_horizons


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--labeled-npz", type=Path, required=True,
                   help="Subtask-labeled npz (states, actions, targets one-hot) from label_kitchen_subtasks.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subtask", required=True,
                   help="Task name or zero-based index in the labeled dataset's task_names array.")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--future-horizons", type=parse_horizons, default=None)
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument("--max-segments", type=int, default=400,
                   help="Cap on demo segments stored in the checkpoint for the eval future index.")
    p.add_argument("--seg-pad", type=int, default=2,
                   help="Frames appended past the labeled segment so the future index can track through completion.")
    p.add_argument("--allow-cross-segment", action="store_true",
                   help="Legacy behavior: allow a task chunk/future to cross into the next task. "
                        "Default keeps every training action and future segment-pure.")
    p.add_argument("--train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--concat-raw", action="store_true")
    p.add_argument("--emphasis-repeat", type=int, default=8,
                   help="Times to duplicate the emphasis dims in the conditioning (0 disables).")
    p.add_argument("--emphasis-dims", default=None,
                   help="Override emphasis slice 'lo,hi'. Default = the subtask's object qpos dims. For approach-dominated subtasks (e.g. light switch) the object dim is near-constant until toggled, so emphasize the robot arm joints '0,9' whose pose carries the approach/reachability signal.")
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    torch.manual_seed(args.seed)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    data = np.load(args.labeled_npz, allow_pickle=True)
    task_names = (
        [str(name) for name in data["task_names"].tolist()]
        if "task_names" in data.files else list(KITCHEN_TASKS)
    )
    try:
        subtask_id = int(args.subtask)
        if not 0 <= subtask_id < len(task_names):
            raise ValueError
        task_name = task_names[subtask_id]
    except (ValueError, IndexError):
        task_name = str(args.subtask)
        if task_name not in task_names:
            raise ValueError(f"Task {task_name!r} is absent from labeled task_names={task_names}")
        subtask_id = task_names.index(task_name)
    if args.emphasis_dims is not None:
        emph_lo, emph_hi = (int(x) for x in args.emphasis_dims.split(","))
    else:
        emph_lo, emph_hi = KITCHEN_OBJ_DIMS[task_name]
    States, Actions, Targets = data["states"], data["actions"], data["targets"]
    H = args.chunk
    future_horizons = args.future_horizons or [H]
    max_future = max(max(future_horizons), H)

    cur_states, fut_states, fut_targets, h_tokens, chunks = [], [], [], [], []
    segments: list[np.ndarray] = []
    with torch.no_grad():
        for ep_i in range(min(len(States), args.max_episodes)):
            S = np.asarray(States[ep_i], dtype=np.float32)
            A = np.asarray(Actions[ep_i], dtype=np.float32)
            T = np.asarray(Targets[ep_i], dtype=np.float32)
            n = min(len(A), len(T))
            if n < max_future + 1:
                continue
            lab = T[:n].argmax(-1)
            valid = T[:n].sum(-1) > 0
            mask = valid & (lab == subtask_id)
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            seg_lo, seg_hi = int(idxs.min()), int(idxs.max())
            # Store the demo segment (states) for this subtask, padded past
            # completion so the demo-locked index can track to the goal frame.
            seg_end = min(seg_hi + 1 + args.seg_pad, len(S))
            if len(segments) < args.max_segments and seg_end - seg_lo > 1:
                segments.append(S[seg_lo:seg_end].copy())
            Sn = torch.from_numpy(norm.encode(S)).to(dev)
            z_target = wm.encode_target(Sn)
            for t in idxs:
                t = int(t)
                if t + max_future >= len(S):
                    continue
                for future_h in future_horizons:
                    span = max(H, future_h)
                    if not args.allow_cross_segment and not bool(mask[t : t + span].all()):
                        continue
                    cur_states.append(Sn[t])
                    fut_states.append(Sn[t + future_h])
                    fut_targets.append(z_target[t + future_h])
                    h_tokens.append(torch.tensor([float(future_h) / float(max(future_horizons))], dtype=Sn.dtype, device=dev))
                    chunks.append(torch.from_numpy(A[t : t + H].reshape(-1)).to(dev))

    SCur = torch.stack(cur_states)
    SFut = torch.stack(fut_states)
    ZFut = torch.stack(fut_targets)
    HTok = torch.stack(h_tokens)
    Chunk = torch.stack(chunks)

    def make_cond(idx: torch.Tensor) -> torch.Tensor:
        s = SCur[idx]
        z = wm.encode(s)
        parts = [z, ZFut[idx], HTok[idx]]
        if args.concat_raw:
            parts.extend([s, SFut[idx]])
        if args.emphasis_repeat > 0:
            parts.append(s[:, emph_lo:emph_hi].repeat(1, args.emphasis_repeat))
        return torch.cat(parts, dim=-1)

    with torch.no_grad():
        cond_dim = int(make_cond(torch.arange(1, device=dev)).shape[1])

    prior = InversePrior(cond_dim, Chunk.shape[1], args.hidden, args.n_blocks).to(dev)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-4)
    seg_arr = np.array([s.astype(np.float32) for s in segments], dtype=object)
    print(json.dumps({"event": "kitchen_subtask_data", "subtask": task_name, "pairs": int(SCur.shape[0]),
                      "segments": int(len(segments)), "cond_dim": cond_dim, "chunk_dim": int(Chunk.shape[1]),
                      "emphasis_dims": f"{emph_lo},{emph_hi}", "emphasis_repeat": args.emphasis_repeat}), flush=True)
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, SCur.shape[0], (args.batch_size,), device=dev)
        with torch.no_grad():
            cond = make_cond(idx)
        pred = prior(cond)
        loss = nn.functional.mse_loss(pred, Chunk[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 5000 == 0:
            print(json.dumps({"event": "kitchen_subtask_train", "subtask": task_name, "step": step, "loss": float(loss.detach().cpu())}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "cond_dim": cond_dim,
            "chunk_dim": int(Chunk.shape[1]),
            "action_dim": int(spec.action_dim),
            "H": int(H),
            "latent_dim": int(cfg["latent_dim"]),
            "hidden": int(args.hidden),
            "n_blocks": int(args.n_blocks),
            "future_horizons": future_horizons,
            "concat_raw": bool(args.concat_raw),
            "subtask": int(subtask_id),
            "subtask_name": task_name,
            "task_names": task_names,
            "allow_cross_segment": bool(args.allow_cross_segment),
            "emphasis_dims": f"{emph_lo},{emph_hi}" if args.emphasis_repeat > 0 else None,
            "emphasis_repeat": int(args.emphasis_repeat),
            "segments": seg_arr,
            "model_path": str(args.model_path),
            "labeled_npz": str(args.labeled_npz),
        },
        args.out,
    )
    print(json.dumps({"event": "kitchen_subtask_saved", "path": str(args.out), "subtask": task_name}), flush=True)


if __name__ == "__main__":
    main()
