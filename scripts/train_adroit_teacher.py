"""RL teacher (data engine) for Adroit Door — the Tier-3 task that breaks our
scripted-expert + BC data pipeline.

AdroitHandDoor-v1 is a 28-DoF anthropomorphic hand with a *flat* 39-D observation
(no achieved/desired goal) and a *dense* reward — there is no simple geometric
controller for finger coordination, and no Minari offline dataset is installed.
So we learn the data source: train an off-policy RL teacher (SAC, MlpPolicy) on
the raw observation and dense reward, then roll it out to produce trajectories
that feed the JEPA world model and a behaviour-cloned latent policy.

Usage:
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/train_adroit_teacher.py \
        --task adroit_door --total-steps 1000000 --train-seconds 14400 \
        --save-model runs/adroit_door/checkpoints/adroit_door_teacher.zip \
        --export-episodes runs/adroit_door/data/adroit_door_teacher_episodes.npz \
        --export-steps 200000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.tasks import resolve_task


def export_teacher_episodes(model, env_id, max_steps, n_steps, seed, deterministic_frac=0.7):
    """Roll the teacher out (mostly greedy, some stochastic for coverage) and save
    flattened (states, actions) per episode in the JEPA ``Episode`` npz layout."""
    rng = np.random.default_rng(seed)
    states_all, actions_all, ep_starts = [], [], []
    env = make_env(env_id, seed=seed, max_episode_steps=max_steps)
    total = 0
    successes = []
    while total < n_steps:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        deterministic = rng.random() < deterministic_frac
        ep_states = [flatten_obs(obs)]
        ep_actions = []
        term = trunc = False
        info = {}
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            ep_actions.append(np.asarray(action, dtype=np.float32))
            ep_states.append(flatten_obs(obs))
            total += 1
        successes.append(float(info.get("is_success", info.get("success", 0.0))))
        ep_starts.append(len(actions_all))
        states_all.append(np.stack(ep_states).astype(np.float32))
        actions_all.append(np.stack(ep_actions).astype(np.float32))
    env.close()
    return states_all, actions_all, float(np.mean(successes))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="adroit_door")
    p.add_argument("--env-id", default=None)
    p.add_argument("--algo", default="sac", choices=["sac", "tqc"])
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--train-seconds", type=int, default=14_400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--net-arch", default="256,256,256")
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--save-model", type=Path, default=None)
    p.add_argument("--export-episodes", type=Path, default=None,
                   help="After training, roll the teacher out and save trajectories for the JEPA WM.")
    p.add_argument("--export-steps", type=int, default=200_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback

    task = resolve_task(args.task, args.env_id)
    env_id = args.env_id or task.env_id
    max_steps = task.max_episode_steps
    save_model = args.save_model or Path(f"runs/{task.slug}/checkpoints/{task.slug}_teacher.zip")
    save_model.parent.mkdir(parents=True, exist_ok=True)

    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    env = make_env(env_id, seed=args.seed, max_episode_steps=max_steps)

    if args.algo == "tqc":
        from sb3_contrib import TQC as Algo
    else:
        from stable_baselines3 import SAC as Algo

    if args.resume and save_model.exists():
        print(f'{{"event": "resume", "model": "{save_model}"}}', flush=True)
        model = Algo.load(str(save_model), env=env, device=args.device)
    else:
        model = Algo(
            "MlpPolicy", env,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1, device=args.device, seed=args.seed,
        )

    class TimeBudget(BaseCallback):
        def __init__(self, seconds): super().__init__(); self.seconds = seconds; self.start = time.time()
        def _on_step(self):
            if time.time() - self.start > self.seconds:
                print(f'{{"event": "time_budget_reached", "seconds": {self.seconds}}}', flush=True)
                return False
            return True

    eval_env = make_env(env_id, seed=args.seed + 10_000, max_episode_steps=max_steps)
    callbacks = CallbackList([
        TimeBudget(args.train_seconds),
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

    if args.export_episodes is not None:
        states, actions, succ = export_teacher_episodes(
            model, env_id, max_steps, args.export_steps, args.seed + 5)
        args.export_episodes.parent.mkdir(parents=True, exist_ok=True)
        # ragged episodes -> object arrays
        np.savez(
            args.export_episodes,
            states=np.array(states, dtype=object),
            actions=np.array(actions, dtype=object),
        )
        print(f'{{"event": "exported_episodes", "path": "{args.export_episodes}", '
              f'"episodes": {len(states)}, "rollout_success": {succ:.3f}}}', flush=True)


if __name__ == "__main__":
    main()
