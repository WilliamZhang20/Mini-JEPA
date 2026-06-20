"""Proper Adroit controller: RL (SAC/TQC) on the FROZEN JEPA latent.

Adroit is non-goal / dense-reward, so there is no HER. We wrap the env so its
observation is the JEPA world-model latent (JEPALatentObsWrapper) and train a
standard off-policy actor-critic on it. This is the deep-learning control half
of the JEPA agent for Tier 3 — it optimizes real environment reward in the
learned latent space, rather than behaviour-cloning a (weak) teacher.

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/train_adroit_controller.py \
        --task adroit_door \
        --jepa-model-path runs/adroit_door_explore/checkpoints/adroit_door_explore_jepa_model.pt \
        --total-steps 2000000 --train-seconds 28800
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import make_env
from jepa_robotics.sb3_jepa import JEPAConcatExtractor, JEPAEncoderExtractor, JEPALatentObsWrapper
from jepa_robotics.tasks import resolve_task


def build_env(env_id, max_steps, seed, jepa_model_path, wrapper_device, encoder_mode,
              render_mode=None, width=None, height=None):
    """frozen -> wrap the env obs with the frozen JEPA latent.
    finetune -> raw env (the trainable JEPA encoder lives inside the policy)."""
    env = make_env(env_id, seed=seed, max_episode_steps=max_steps,
                   render_mode=render_mode, width=width, height=height)
    if encoder_mode == "frozen":
        return JEPALatentObsWrapper(env, jepa_model_path, device=wrapper_device)
    return env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="adroit_door")
    p.add_argument("--env-id", default=None)
    p.add_argument("--jepa-model-path", type=Path, required=True)
    p.add_argument("--encoder-mode", default="finetune",
                   choices=["frozen", "finetune", "none", "concat"],
                   help="frozen: env-wrapped frozen latent. finetune: trainable JEPA encoder in the "
                        "policy (warm-started). none: NON-JEPA reference — standard RL on the raw "
                        "observation. concat: raw obs + trainable JEPA latent (full observability + "
                        "world-model features) — the iteration after pure-latent control failed.")
    p.add_argument("--algo", default="sac", choices=["sac", "tqc"])
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--train-seconds", type=int, default=28_800)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--net-arch", default="256,256,256")
    p.add_argument("--n-critics", type=int, default=2)
    p.add_argument("--ent-coef", default="auto",
                   help="SAC/TQC entropy temperature. 'auto' tunes it (can collapse to ~0 and kill "
                        "exploration on hard sparse-success tasks); set a fixed float (e.g. 0.1) to "
                        "sustain exploration.")
    p.add_argument("--ent-coef-final", type=float, default=None,
                   help="If set (with a fixed --ent-coef), linearly ANNEAL the entropy coef from "
                        "--ent-coef to this value over --ent-coef-anneal-steps (explore->exploit, to "
                        "stabilize transient successes that fixed-entropy can't lock in).")
    p.add_argument("--ent-coef-anneal-steps", type=int, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--wrapper-device", default="cpu",
                   help="Device for the frozen JEPA encoder inside the env wrapper (cpu avoids per-step H2D).")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--save-model", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import imageio.v2 as imageio
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback

    task = resolve_task(args.task, args.env_id)
    env_id = args.env_id or task.env_id
    max_steps = task.max_episode_steps
    save_model = args.save_model or Path(f"runs/{task.slug}/checkpoints/{task.slug}_jepa_latent_ctrl.zip")
    video_out = args.video_out or Path(f"runs/{task.slug}/videos/{task.slug}_jepa_latent_ctrl.mp4")
    save_model.parent.mkdir(parents=True, exist_ok=True)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    env = build_env(env_id, max_steps, args.seed, args.jepa_model_path, args.wrapper_device, args.encoder_mode)

    policy_kwargs = dict(net_arch=net_arch)
    if args.encoder_mode in ("finetune", "concat"):
        extractor = JEPAEncoderExtractor if args.encoder_mode == "finetune" else JEPAConcatExtractor
        policy_kwargs.update(
            features_extractor_class=extractor,
            features_extractor_kwargs=dict(model_path=str(args.jepa_model_path), freeze=False),
        )
    if args.algo == "tqc":
        from sb3_contrib import TQC as Algo
        policy_kwargs["n_critics"] = args.n_critics
        algo_kwargs = dict(policy_kwargs=policy_kwargs)
    else:
        from stable_baselines3 import SAC as Algo
        algo_kwargs = dict(policy_kwargs=policy_kwargs)

    if args.resume and save_model.exists():
        print(f'{{"event": "resume", "model": "{save_model}"}}', flush=True)
        model = Algo.load(str(save_model), env=env, device=args.device)
    else:
        ent_coef = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
        model = Algo("MlpPolicy", env, learning_rate=args.learning_rate, batch_size=args.batch_size,
                     gamma=args.gamma, buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                     ent_coef=ent_coef, verbose=1, device=args.device, seed=args.seed, **algo_kwargs)

    class TimeBudget(BaseCallback):
        def __init__(self, s): super().__init__(); self.s = s; self.t = time.time()
        def _on_step(self):
            if time.time() - self.t > self.s:
                print(f'{{"event": "time_budget_reached", "seconds": {self.s}}}', flush=True); return False
            return True

    class EntropyAnneal(BaseCallback):
        """Linearly anneal a fixed entropy coef start->final over anneal_steps (explore->exploit)."""
        def __init__(self, start, final, steps):
            super().__init__(); self.start = start; self.final = final; self.steps = steps
        def _on_step(self):
            frac = min(1.0, self.num_timesteps / max(1, self.steps))
            val = self.start + frac * (self.final - self.start)
            self.model.ent_coef_tensor = torch.as_tensor(val, device=self.model.device)
            return True

    cb_list = [TimeBudget(args.train_seconds)]
    if args.ent_coef_final is not None and args.ent_coef != "auto":
        anneal_steps = args.ent_coef_anneal_steps or args.total_steps
        cb_list.append(EntropyAnneal(float(args.ent_coef), args.ent_coef_final, anneal_steps))
        print(f'{{"event": "entropy_anneal", "from": {float(args.ent_coef)}, '
              f'"to": {args.ent_coef_final}, "steps": {anneal_steps}}}', flush=True)

    eval_env = build_env(env_id, max_steps, args.seed + 10_000, args.jepa_model_path, args.wrapper_device, args.encoder_mode)
    callbacks = CallbackList([
        *cb_list,
        CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(save_model.parent),
                           name_prefix=save_model.stem),
        EvalCallback(eval_env, best_model_save_path=str(save_model.parent / f"{save_model.stem}_best"),
                     log_path=str(save_model.parent / f"{save_model.stem}_eval"),
                     eval_freq=args.eval_freq, n_eval_episodes=min(args.eval_episodes, 10),
                     deterministic=True, render=False, verbose=1),
    ])
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=callbacks,
                reset_num_timesteps=not args.resume, log_interval=20)
    model.save(str(save_model))
    print(f'{{"event": "trained", "minutes": {round((time.time()-t0)/60,1)}, "model": "{save_model}"}}', flush=True)

    # record
    rec = build_env(env_id, max_steps, args.seed + 222, args.jepa_model_path, args.wrapper_device, args.encoder_mode,
                    render_mode="rgb_array", width=640, height=480)
    frames, succ = [], []
    for ep in range(args.eval_episodes):
        obs, _ = rec.reset(seed=args.seed + 1000 + ep)
        f = rec.render();  frames.append(f) if f is not None else None
        term = trunc = False; info = {}
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = rec.step(action)
            f = rec.render();  frames.append(f) if f is not None else None
        succ.append(float(info.get("is_success", info.get("success", 0.0))))
    rec.close(); env.close(); eval_env.close()
    imageio.mimsave(video_out, frames, fps=30, format="FFMPEG")
    print(f'{{"event": "recorded", "video": "{video_out}", "episodes": {args.eval_episodes}, '
          f'"success_rate": {float(np.mean(succ)):.3f}}}', flush=True)


if __name__ == "__main__":
    main()
