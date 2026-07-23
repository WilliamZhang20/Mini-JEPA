"""Evaluate the action-chunked diffusion policy on FrankaKitchen.

Receding-horizon control: encode obs -> z (JEPA latent), sample an H-action chunk by
DDPM reverse diffusion conditioned on z, execute the first ``exec-k`` actions, then
replan. Uses the EMA weights (diffusion policies sample much better from EMA).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.task_families.kitchen import KITCHEN_TASKS, KitchenScheduler
from jepa_robotics.algos.priors import EpsNet, make_ddpm, sample_chunk

DEFAULT_TASKS = KITCHEN_TASKS
Scheduler = KitchenScheduler


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="franka_kitchen")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--exec-k", type=int, default=4, help="actions executed per replan (receding horizon)")
    p.add_argument("--cfg-weight", type=float, default=1.0, help="classifier-free guidance weight (1.0 = none)")
    p.add_argument("--init-noise-scale", type=float, default=1.0,
                   help="Scale the initial noise for diffusion/flow sampling.")
    p.add_argument("--action-scale", type=float, default=1.0,
                   help="Scale sampled actions before clipping/execution.")
    p.add_argument("--subtask-timeout", type=int, default=0,
                   help="scheduler: if a target subtask isn't done within this many steps, rotate to the next "
                        "incomplete one (breaks the stuck-at-2 dead-end). 0 = no rotation.")
    p.add_argument("--task-order", default=",".join(DEFAULT_TASKS),
                   help="Comma-separated one-hot order used by a subtask-conditioned skill checkpoint.")
    p.add_argument("--collect-out", type=Path, default=None,
                   help="self-imitation: save rollouts reaching >= --collect-min-tasks as a labeled npz "
                        "(states, actions, target one-hot) to augment the scarce full-sequence training data")
    p.add_argument("--collect-min-tasks", type=int, default=4)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--video-out", type=Path, default=None, help="record an mp4 of the policy")
    p.add_argument("--video-episodes", type=int, default=6)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--video-tail", type=int, default=30,
                   help="extra frames simulated AFTER success so the final motion (e.g. cabinet slide) "
                        "completes instead of the clip cutting off at the completion threshold")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    tasks_order = [t.strip() for t in args.task_order.split(",") if t.strip()]
    if len(tasks_order) != 4:
        raise ValueError(f"--task-order must contain exactly 4 task names, got {tasks_order}")

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    ck = torch.load(args.policy, map_location=dev, weights_only=False)
    net = EpsNet(ck["chunk_dim"], ck["cond_dim"], ck["hidden"], n_blocks=ck["n_blocks"]).to(dev)
    net.load_state_dict(ck["ema"] if args.use_ema else ck["state_dict"])
    net.eval()
    ddpm = make_ddpm(ck["diffusion_steps"], dev)
    H, A_dim, chunk_dim = ck["H"], ck["action_dim"], ck["chunk_dim"]
    HH = int(ck.get("obs_hist", 1))
    objective = ck.get("objective", "diffusion")
    raw_obs = bool(ck.get("raw_obs", False))
    concat_raw = bool(ck.get("concat_raw", False))
    progress_cond = bool(ck.get("progress_cond", False))
    subtask_cond = bool(ck.get("subtask_cond", False))

    def encode(obs, progress=0.0, target=None):
        x = torch.from_numpy(norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev)
        if raw_obs:
            z = x
        else:
            zz = wm.encode(x)
            z = torch.cat([x, zz], dim=-1) if concat_raw else zz   # [raw++latent] or latent
        if progress_cond:  # append live "subtasks done so far" (/4)
            z = torch.cat([z, torch.full((1, 1), float(progress), device=dev)], dim=-1)
        if subtask_cond:   # append the target-subtask one-hot (the scheduler's choice)
            oh = torch.zeros(1, 4, device=dev)
            if target is not None:
                oh[0, target] = 1.0
            z = torch.cat([z, oh], dim=-1)
        return z

    from collections import deque
    tasks = []
    col_S, col_A, col_T = [], [], []   # self-imitation collection
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=args.seed + ep)
        progress = 0.0; done_tasks = set(); sched = Scheduler(tasks_order, args.subtask_timeout); target = sched.update(done_tasks)
        z = encode(obs, progress, target)
        hist = deque([z] * HH, maxlen=HH)             # consecutive latent history
        term = trunc = False; info = {}; step_i = 0; chunk = None; j = 0
        ep_S = [flatten_obs(obs).copy()]; ep_A = []; ep_T = []
        while not (term or trunc):
            if step_i % args.exec_k == 0:
                cond = torch.cat(list(hist), dim=-1)  # [1, HH*L]
                chunk = (
                    sample_chunk(
                        net,
                        ddpm,
                        cond,
                        chunk_dim,
                        dev,
                        objective,
                        cfg_weight=args.cfg_weight,
                        init_noise_scale=args.init_noise_scale,
                    )[0].cpu().numpy().reshape(H, A_dim)
                    * args.action_scale
                )
                j = 0
            a = np.clip(chunk[min(j, H - 1)], low, high).astype(np.float32)
            ep_A.append(a.copy()); ep_T.append(target)   # action + the subtask it pursued
            obs, _, term, trunc, info = env.step(a)
            ep_S.append(flatten_obs(obs).copy())
            done_tasks |= set(info.get("step_task_completions", []))
            target = sched.update(done_tasks)          # scheduler (with optional stall-rotation)
            progress = float(info.get("tasks_done", 0)) / 4.0
            hist.append(encode(obs, progress, target))
            step_i += 1; j += 1
        nt = int(info.get("tasks_done", 0))
        tasks.append(nt)
        if args.collect_out is not None and nt >= args.collect_min_tasks and len(ep_A) > 16:
            oh = np.zeros((len(ep_A), 4), np.float32); oh[np.arange(len(ep_A)), ep_T] = 1.0
            col_S.append(np.asarray(ep_S, np.float32)); col_A.append(np.asarray(ep_A, np.float32)); col_T.append(oh)
        env.close()
    if args.collect_out is not None:
        args.collect_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.collect_out, states=np.array(col_S, dtype=object),
                 actions=np.array(col_A, dtype=object), targets=np.array(col_T, dtype=object))
        print(f'{{"event": "collected", "out": "{args.collect_out}", "kept_episodes": {len(col_S)}, '
              f'"of": {args.episodes}, "min_tasks": {args.collect_min_tasks}}}', flush=True)
    tasks = np.array(tasks)
    print(f"RESULT diffusion-policy(exec_k={args.exec_k}) mean_tasks={tasks.mean():.2f}/4 "
          f"full4={np.mean(tasks >= 4):.2f} >=1={np.mean(tasks >= 1):.2f} "
          f">=2={np.mean(tasks >= 2):.2f} >=3={np.mean(tasks >= 3):.2f}", flush=True)

    if args.video_out is not None:
        import imageio.v2 as imageio
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        frames, vtasks = [], []
        for ep in range(args.video_episodes):
            env = make_env(task.env_id, seed=args.seed + 5000 + ep, max_episode_steps=task.max_episode_steps,
                           render_mode="rgb_array", width=args.width, height=args.height)
            low, high = env.action_space.low, env.action_space.high
            obs, _ = env.reset(seed=args.seed + 5000 + ep)
            progress = 0.0; done_tasks = set(); sched = Scheduler(tasks_order, args.subtask_timeout); target = sched.update(done_tasks)
            z = encode(obs, progress, target); hist = deque([z] * HH, maxlen=HH)
            term = trunc = False; info = {}; step_i = 0; chunk = None; j = 0
            f = env.render()
            if f is not None: frames.append(f)
            while not (term or trunc):
                if step_i % args.exec_k == 0:
                    cond = torch.cat(list(hist), dim=-1)
                    chunk = (
                        sample_chunk(
                            net,
                            ddpm,
                            cond,
                            chunk_dim,
                            dev,
                            objective,
                            cfg_weight=args.cfg_weight,
                            init_noise_scale=args.init_noise_scale,
                        )[0].cpu().numpy().reshape(H, A_dim)
                        * args.action_scale
                    )
                    j = 0
                a = np.clip(chunk[min(j, H - 1)], low, high).astype(np.float32)
                obs, _, term, trunc, info = env.step(a)
                done_tasks |= set(info.get("step_task_completions", []))
                target = sched.update(done_tasks)
                progress = float(info.get("tasks_done", 0)) / 4.0
                hist.append(encode(obs, progress, target)); step_i += 1; j += 1
                f = env.render()
                if f is not None: frames.append(f)
            # follow-through tail: the episode terminates the instant the last task crosses its
            # completion threshold, cutting the motion short. Keep stepping the underlying sim with
            # the last action (continuing then holding the chunk) so the final slide/open completes.
            if chunk is not None:
                for k in range(args.video_tail):
                    a = np.clip(chunk[min(j + k, H - 1)], low, high).astype(np.float32)
                    try:
                        env.unwrapped.step(a)
                    except Exception:
                        if frames: frames.append(frames[-1])   # fallback: hold the last frame
                        continue
                    f = env.render()
                    if f is not None: frames.append(f)
            vtasks.append(int(info.get("tasks_done", 0)))
            env.close()
        imageio.mimsave(args.video_out, frames, fps=args.fps, format="FFMPEG")
        print(f'{{"event": "recorded", "video": "{args.video_out}", '
              f'"episodes": {args.video_episodes}, "mean_tasks": {float(np.mean(vtasks)):.2f}}}', flush=True)


if __name__ == "__main__":
    main()
