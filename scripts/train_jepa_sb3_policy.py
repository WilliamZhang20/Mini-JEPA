"""Train a no-BC-ceiling Fetch policy with real-env RL on frozen JEPA latents.

This is the performance-first path for hard contact tasks like FetchSlide:
initialize the representation from a trained JEPA world model, then optimize a
TQC+HER actor/critic directly against the environment success reward. It keeps
JEPA in the loop as the state abstraction, but avoids the ceiling imposed by a
scripted or behaviour-cloned reference.
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
from jepa_robotics.sb3_jepa import JEPALatentExtractor
from jepa_robotics.tasks import resolve_task


def build_env(env_id: str, max_steps: int, seed: int, render_mode=None, width=None, height=None):
    env = make_env(
        env_id,
        seed=seed,
        max_episode_steps=max_steps,
        render_mode=render_mode,
        width=width,
        height=height,
    )
    return env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--env-id", default=None)
    p.add_argument("--jepa-model-path", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--train-seconds", type=int, default=10_800)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--save-model", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--save-replay-buffer", action="store_true")
    p.add_argument(
        "--replay-checkpoint-freq",
        type=int,
        default=None,
        help=(
            "If set with --save-replay-buffer, save full replay-buffer snapshots "
            "at this lower frequency while keeping lightweight model checkpoints "
            "at --checkpoint-freq."
        ),
    )
    p.add_argument("--replay-buffer-path", type=Path, default=None)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--net-arch", default="512,512,512")
    p.add_argument("--n-critics", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--fixed-ent-coef", type=float, default=None)
    p.add_argument("--latent-layer-norm", action="store_true")
    p.add_argument("--collapse-critic-loss", type=float, default=100.0)
    p.add_argument("--collapse-actor-loss", type=float, default=1000.0)
    p.add_argument("--collapse-ent-coef", type=float, default=0.5)
    p.add_argument(
        "--resume",
        action="store_true",
        help="If --save-model already exists, continue training from it instead of starting over.",
    )
    p.add_argument(
        "--resume-warmup-steps",
        type=int,
        default=None,
        help=(
            "On resume, delay gradient updates until this many fresh env steps "
            "have been collected. Defaults to one episode plus one step."
        ),
    )
    p.add_argument(
        "--resume-policy-warmup",
        action="store_true",
        help=(
            "When --resume-warmup-steps delays training, still collect warmup "
            "rollouts with the loaded policy instead of SB3's random warmup actions."
        ),
    )
    args = p.parse_args()

    from sb3_contrib import TQC
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.her import HerReplayBuffer

    task = resolve_task(args.task, args.env_id)
    env_id = args.env_id or task.env_id
    max_steps = args.max_steps or task.max_episode_steps
    save_model = args.save_model or Path(f"runs/{task.slug}/checkpoints/{task.slug}_jepa_tqc.zip")
    video_out = args.video_out or Path(f"runs/{task.slug}/videos/jepa_latent_tqc_{task.slug}.mp4")
    save_model.parent.mkdir(parents=True, exist_ok=True)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(env_id, max_steps, args.seed)
    net_arch = [int(part) for part in args.net_arch.split(",") if part.strip()]
    if args.resume and save_model.exists():
        print(f'{{"event": "resume", "model": "{save_model}"}}', flush=True)
        model = TQC.load(str(save_model), env=env, device=args.device)
        if args.replay_buffer_path is not None:
            print(f'{{"event": "load_replay_buffer", "path": "{args.replay_buffer_path}"}}', flush=True)
            model.load_replay_buffer(str(args.replay_buffer_path))
        warmup_steps = args.resume_warmup_steps if args.resume_warmup_steps is not None else max_steps + 1
        resume_learning_starts = int(getattr(model, "num_timesteps", 0)) + max(0, warmup_steps)
        model.learning_starts = max(int(getattr(model, "learning_starts", 0)), resume_learning_starts)
        model.batch_size = args.batch_size
        model.gradient_steps = args.gradient_steps
        model.gamma = args.gamma
        model.tau = args.tau
        if args.resume_policy_warmup and warmup_steps > 0:
            original_sample_action = model._sample_action

            def policy_warmup_sample_action(learning_starts, action_noise=None, n_envs=1):
                return original_sample_action(0, action_noise=action_noise, n_envs=n_envs)

            model._sample_action = policy_warmup_sample_action
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        for optimizer_name in ("actor", "critic"):
            optimizer = getattr(getattr(model, optimizer_name, None), "optimizer", None)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
        if getattr(model, "ent_coef_optimizer", None) is not None:
            for group in model.ent_coef_optimizer.param_groups:
                group["lr"] = args.learning_rate
        if args.fixed_ent_coef is not None:
            model.ent_coef_optimizer = None
            model.log_ent_coef = None
            model.ent_coef_tensor = torch.tensor(float(args.fixed_ent_coef), device=model.device)
        print(
            f'{{"event": "resume_warmup", "num_timesteps": {model.num_timesteps}, '
            f'"learning_starts": {model.learning_starts}, "learning_rate": {args.learning_rate}, '
            f'"fixed_ent_coef": {args.fixed_ent_coef}, "warmup_steps": {warmup_steps}, '
            f'"policy_warmup": {str(args.resume_policy_warmup).lower()}}}',
            flush=True,
        )
    else:
        model = TQC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            policy_kwargs=dict(
                features_extractor_class=JEPALatentExtractor,
                features_extractor_kwargs=dict(
                    model_path=str(args.jepa_model_path),
                    device=args.device,
                    layer_norm=args.latent_layer_norm,
                ),
                net_arch=net_arch,
                n_critics=args.n_critics,
            ),
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            tau=args.tau,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            gradient_steps=args.gradient_steps,
            ent_coef=args.fixed_ent_coef if args.fixed_ent_coef is not None else "auto",
            verbose=1,
            device=args.device,
            seed=args.seed,
        )

    class TimeBudget(BaseCallback):
        def __init__(self, seconds: int):
            super().__init__()
            self.seconds = seconds
            self.start = time.time()

        def _on_step(self) -> bool:
            if time.time() - self.start > self.seconds:
                print(f'{{"event": "time_budget_reached", "seconds": {self.seconds}}}', flush=True)
                return False
            return True

    class CollapseGuard(BaseCallback):
        def __init__(self, critic_loss: float, actor_loss: float, ent_coef: float):
            super().__init__()
            self.critic_loss = critic_loss
            self.actor_loss = actor_loss
            self.ent_coef = ent_coef
            self.stopped = False

        def _on_step(self) -> bool:
            values = getattr(self.model.logger, "name_to_value", {})
            critic = values.get("train/critic_loss")
            actor = values.get("train/actor_loss")
            entropy = values.get("train/ent_coef")
            if critic is None and actor is None and entropy is None:
                return True
            critic_bad = critic is not None and float(critic) > self.critic_loss
            actor_bad = actor is not None and abs(float(actor)) > self.actor_loss
            entropy_bad = entropy is not None and float(entropy) > self.ent_coef
            if critic_bad or actor_bad or entropy_bad:
                print(
                    f'{{"event": "collapse_guard_stop", "critic_loss": {critic}, '
                    f'"actor_loss": {actor}, "ent_coef": {entropy}}}',
                    flush=True,
                )
                self.stopped = True
                return False
            return True

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(save_model.parent),
        name_prefix=save_model.stem,
        save_replay_buffer=args.save_replay_buffer and args.replay_checkpoint_freq is None,
    )
    callbacks_list = [TimeBudget(args.train_seconds), ckpt_cb]
    if args.save_replay_buffer and args.replay_checkpoint_freq is not None:
        replay_ckpt_cb = CheckpointCallback(
            save_freq=args.replay_checkpoint_freq,
            save_path=str(save_model.parent),
            name_prefix=f"{save_model.stem}_replay_snapshot",
            save_replay_buffer=True,
        )
        callbacks_list.append(replay_ckpt_cb)
    eval_env = build_env(env_id, max_steps, args.seed + 10_000)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_model.parent / f"{save_model.stem}_best"),
        log_path=str(save_model.parent / f"{save_model.stem}_eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=min(args.eval_episodes, 10),
        deterministic=True,
        render=False,
        verbose=1,
    )
    collapse_guard = CollapseGuard(
        critic_loss=args.collapse_critic_loss,
        actor_loss=args.collapse_actor_loss,
        ent_coef=args.collapse_ent_coef,
    )
    callbacks_list.extend([eval_cb, collapse_guard])
    callbacks = CallbackList(callbacks_list)
    t0 = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=False,
        log_interval=50,
        reset_num_timesteps=not args.resume,
    )
    model.save(str(save_model))
    if args.save_replay_buffer:
        replay_out = save_model.with_name(f"{save_model.stem}_replay_buffer.pkl")
        model.save_replay_buffer(str(replay_out))
        print(f'{{"event": "saved_replay_buffer", "path": "{replay_out}"}}', flush=True)
    eval_env.close()
    print(f'{{"event": "trained", "minutes": {round((time.time() - t0) / 60, 1)}, "model": "{save_model}"}}', flush=True)

    rec_env = build_env(
        env_id,
        max_steps,
        args.seed,
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
    )
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
    env.close()
    imageio.mimsave(video_out, frames, fps=args.fps, format="FFMPEG")
    print(
        f'{{"event": "recorded", "video": "{video_out}", "episodes": {args.eval_episodes}, '
        f'"success_rate": {float(np.mean(successes)):.3f}}}',
        flush=True,
    )


if __name__ == "__main__":
    main()
