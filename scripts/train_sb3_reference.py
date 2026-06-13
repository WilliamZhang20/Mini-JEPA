"""Train a max-performance RL reference (TQC+HER) on a Fetch task and record it.

Reproduces the RL-Zoo recipe (TQC + HerReplayBuffer + TimeFeatureWrapper) but
trains directly on the gymnasium-robotics ``-v4`` env, so the reference is a
genuine max-perf agent on the exact env the JEPA agent uses (no v1->v4 transfer
gap). Trains up to a step budget or a wall-clock cap (whichever first), saves the
model, then records a high-res MP4 and reports success rate.

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/train_sb3_reference.py \
        --task fetch_slide --train-seconds 12600 --total-steps 3000000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import make_env
from jepa_robotics.tasks import resolve_task


def build_env(env_id: str, max_steps: int, seed: int, render_mode=None, width=None, height=None):
    from sb3_contrib.common.wrappers import TimeFeatureWrapper

    env = make_env(env_id, seed=seed, max_episode_steps=max_steps,
                   render_mode=render_mode, width=width, height=height)
    return TimeFeatureWrapper(env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--env-id", default=None)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--train-seconds", type=int, default=12_600, help="Wall-clock training cap (stops early if hit).")
    p.add_argument("--max-steps", type=int, default=50, help="Episode length / HER horizon.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--save-model", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --save-model already exists, continue training from it instead of starting over.",
    )
    args = p.parse_args()

    from sb3_contrib import TQC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.her import HerReplayBuffer

    task = resolve_task(args.task, args.env_id)
    env_id = args.env_id or task.env_id
    save_model = args.save_model or Path(f"runs/{task.slug}/checkpoints/{task.slug}_tqc_reference.zip")
    video_out = args.video_out or Path(f"runs/{task.slug}/videos/reference_{task.slug}.mp4")
    save_model.parent.mkdir(parents=True, exist_ok=True)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(env_id, args.max_steps, args.seed)

    if args.resume and save_model.exists():
        print(f'{{"event": "resume", "model": "{save_model}"}}', flush=True)
        model = TQC.load(str(save_model), env=env, device=args.device)
        resume_learning_starts = int(getattr(model, "num_timesteps", 0)) + args.max_steps + 1
        model.learning_starts = max(int(getattr(model, "learning_starts", 0)), resume_learning_starts)
        print(
            f'{{"event": "resume_warmup", "num_timesteps": {model.num_timesteps}, '
            f'"learning_starts": {model.learning_starts}}}',
            flush=True,
        )
    else:
        # RL-Zoo TQC+HER hyperparameters for FetchSlide.
        model = TQC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            batch_size=512,
            gamma=0.98,
            learning_rate=1e-3,
            tau=0.005,
            buffer_size=1_000_000,
            learning_starts=1000,
            verbose=1,
            device=args.device,
            seed=args.seed,
        )

    class TimeBudget(BaseCallback):
        def __init__(self, seconds):
            super().__init__()
            self.seconds = seconds
            self.start = time.time()

        def _on_step(self) -> bool:
            if time.time() - self.start > self.seconds:
                print(f'{{"event": "time_budget_reached", "seconds": {self.seconds}}}', flush=True)
                return False
            return True

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=str(save_model.parent), name_prefix=save_model.stem
    )
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=[TimeBudget(args.train_seconds), ckpt_cb],
                progress_bar=False, log_interval=50, reset_num_timesteps=not args.resume)
    model.save(str(save_model))
    print(f'{{"event": "trained", "minutes": {round((time.time()-t0)/60,1)}, "model": "{save_model}"}}', flush=True)

    # Record + score with the SAME wrapper so obs spaces match exactly.
    rec_env = build_env(env_id, args.max_steps, args.seed, render_mode="rgb_array",
                        width=args.width, height=args.height)
    frames, successes = [], []
    for ep in range(args.eval_episodes):
        obs, _ = rec_env.reset(seed=args.seed + 1000 + ep)
        frame = rec_env.render()
        if frame is not None:
            frames.append(frame)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = rec_env.step(action)
            frame = rec_env.render()
            if frame is not None:
                frames.append(frame)
        successes.append(float(info.get("is_success", 0.0)))
    rec_env.close()
    imageio.mimsave(video_out, frames, fps=args.fps, format="FFMPEG")
    print(f'{{"event": "recorded", "video": "{video_out}", "episodes": {args.eval_episodes}, '
          f'"success_rate": {float(np.mean(successes)):.3f}}}', flush=True)


if __name__ == "__main__":
    main()
