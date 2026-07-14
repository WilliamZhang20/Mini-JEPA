"""Dense-reward RL recovery SPECIALIST for the AntMaze walker — a data source,
not a runtime controller.

Random/OU flailing essentially cannot self-right the Ant (33 recoveries in 88k
flipped steps), so the recovery flow cannot be mined from undirected trials.
Self-righting is, however, a short-horizon dense-reward skill — the easiest kind
of RL problem — so we train a small SAC agent whose episodes START in flipped
states (restored via set_state from a bank mined from our own rollout npz
files) with reward = uprightness gain. The trained specialist is then rolled
out from held-out flipped states and its successful recoveries are saved in the
standard episode npz format for distillation into the SSL recovery flow
(train_recovery_walker.py). The runtime controller stays SSL; RL only
manufactures the trajectories the demos never contained (same pattern as the
HandManipulate data-source plan).

qpos/qvel reconstruction from stored 31-dim states:
qpos = [xy (achieved dims 27:29), z+quat+joints (obs dims 0:13)]; qvel = obs 13:27.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

from jepa_robotics.data import Episode, load_episodes_npz, save_episodes_npz
from jepa_robotics.envs import make_env
from jepa_robotics.tasks import resolve_task
from scripts.train_recovery_walker import uprightness


def state_to_qpos_qvel(s: np.ndarray):
    qpos = np.concatenate([s[27:29], s[0:13]]).astype(np.float64)
    qvel = s[13:27].astype(np.float64)
    return qpos, qvel


def build_flipped_bank(npz_paths, *, max_u=0.2, max_states=6000, seed=0):
    rng = np.random.default_rng(seed)
    bank = []
    for path in npz_paths:
        for ep in load_episodes_npz(path):
            u = uprightness(ep.states)
            idx = np.flatnonzero(u < max_u)
            bank.extend(ep.states[i] for i in idx[:: max(1, len(idx) // 50)])
    bank = np.asarray(bank, np.float32)
    if len(bank) > max_states:
        bank = bank[rng.choice(len(bank), max_states, replace=False)]
    return bank


class FlippedAntRecoveryEnv(gym.Env):
    """Ant self-righting: episodes start in a flipped state drawn from the bank."""

    def __init__(self, env_id, bank, *, horizon=150, success_u=0.85, seed=0):
        self.maze = make_env(env_id, seed=seed, max_episode_steps=1_000_000)
        self.ant = self.maze.unwrapped.ant_env
        self.bank = bank
        self.horizon = horizon
        self.success_u = success_u
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (27,), np.float64)
        self.action_space = self.maze.action_space
        self._t = 0
        self._u = 0.0

    def _obs_u(self, obs_dict):
        o = np.asarray(obs_dict["observation"], np.float64)
        u = 1.0 - 2.0 * float(o[2] ** 2 + o[3] ** 2)
        return o, u

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict, _ = self.maze.reset(seed=int(self.rng.integers(1 << 30)))
        s = self.bank[int(self.rng.integers(len(self.bank)))]
        qpos, qvel = state_to_qpos_qvel(s)
        self.ant.set_state(qpos, qvel)
        obs_dict, _, _, _, _ = self.maze.step(np.zeros(self.action_space.shape, np.float32))
        o, self._u = self._obs_u(obs_dict)
        self._t = 0
        return o, {}

    def step(self, action):
        obs_dict, _, _, _, _ = self.maze.step(action)
        o, u = self._obs_u(obs_dict)
        reward = 20.0 * (u - self._u) - 0.05
        self._u = u
        self._t += 1
        terminated = u > self.success_u
        if terminated:
            reward += 5.0
        truncated = self._t >= self.horizon
        return o, reward, terminated, truncated, {"is_success": terminated}


def collect_distill_episodes(model, env_id, bank, n_episodes, *, horizon=150, success_u=0.85,
                             goal_radius=4.0, seed=123):
    """Roll the specialist from flipped bank states; save SUCCESSFUL recoveries as
    31-dim-state episodes (desired goal relabeled to a random nearby xy, mimicking
    live subgoal conditioning)."""
    rng = np.random.default_rng(seed)
    env = FlippedAntRecoveryEnv(env_id, bank, horizon=horizon, success_u=success_u, seed=seed)
    episodes, n_succ = [], 0
    for _ in range(n_episodes):
        o, _ = env.reset()
        states, actions = [], []
        done = trunc = False
        succ = False
        while not (done or trunc):
            xy = env.ant.data.qpos[:2].astype(np.float32)
            ang = rng.uniform(0, 2 * np.pi)
            desired = xy + goal_radius * np.array([np.cos(ang), np.sin(ang)], np.float32)
            states.append(np.concatenate([o.astype(np.float32), xy, desired]))
            a, _ = model.predict(o, deterministic=False)
            o, _, done, trunc, info = env.step(a)
            actions.append(np.asarray(a, np.float32))
            succ = bool(info.get("is_success", False))
        xy = env.ant.data.qpos[:2].astype(np.float32)
        states.append(np.concatenate([o.astype(np.float32), xy, xy]))
        if succ:
            n_succ += 1
            episodes.append(Episode(states=np.asarray(states, np.float32),
                                    actions=np.asarray(actions, np.float32)))
    env.maze.close()
    return episodes, n_succ


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episodes-npz", type=Path, nargs="+", required=True,
                   help="Rollout npz files to mine flipped start states from.")
    p.add_argument("--save-model", type=Path, required=True)
    p.add_argument("--distill-out", type=Path, required=True,
                   help="Where to save the specialist's successful recovery episodes (npz).")
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--distill-episodes", type=int, default=800)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--success-u", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    task = resolve_task(args.task, None)
    bank = build_flipped_bank(args.episodes_npz, seed=args.seed)
    print(json.dumps({"event": "flipped_bank", "states": len(bank)}), flush=True)

    env = Monitor(FlippedAntRecoveryEnv(task.env_id, bank, horizon=args.horizon,
                                        success_u=args.success_u, seed=args.seed))
    model = SAC("MlpPolicy", env, verbose=1, seed=args.seed, device=args.device,
                batch_size=256, learning_rate=3e-4, buffer_size=400_000,
                learning_starts=2000, policy_kwargs=dict(net_arch=[256, 256]))
    model.learn(total_timesteps=args.total_steps, log_interval=25)
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.save_model))
    print(json.dumps({"event": "rl_saved", "path": str(args.save_model)}), flush=True)

    episodes, n_succ = collect_distill_episodes(model, task.env_id, bank, args.distill_episodes,
                                                horizon=args.horizon, success_u=args.success_u,
                                                seed=args.seed + 999)
    args.distill_out.parent.mkdir(parents=True, exist_ok=True)
    save_episodes_npz(args.distill_out, episodes)
    print(json.dumps({"event": "distill_saved", "path": str(args.distill_out),
                      "episodes": len(episodes),
                      "success_rate": round(n_succ / max(1, args.distill_episodes), 3)}), flush=True)


if __name__ == "__main__":
    main()
