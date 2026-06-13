"""Record a max-performance RL reference (pretrained SB3/sb3-contrib agent).

Downloads a pretrained checkpoint from the Hugging Face Hub (or loads a local
.zip), runs it on the task env, and writes a high-res MP4 plus a success rate.
Used to establish a strong reference benchmark on hard tasks like FetchSlide,
where the scripted controller's open-loop strike tops out far below RL+HER.

    PYTHONNOUSERSITE=1 MUJOCO_GL=egl python scripts/record_sb3_reference.py \
        --task fetch_slide --hf-repo sb3/tqc-FetchSlide-v1 \
        --hf-filename tqc-FetchSlide-v1.zip --episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa_robotics.envs import make_env
from jepa_robotics.evaluate import SB3Policy
from jepa_robotics.tasks import resolve_task


class TimeFeatureFloat32(gymnasium.Wrapper):
    """RL-Zoo compatibility wrapper for pretrained Fetch checkpoints.

    The RL-Zoo Fetch models were trained with a ``TimeFeatureWrapper`` (appends
    normalized remaining time to ``observation``) and float32 Dict observations.
    This reproduces that so a v1-trained policy runs on the gymnasium env.
    """

    def __init__(self, env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self._t = 0
        obs_space = env.observation_space
        o = obs_space["observation"]
        low = np.concatenate([o.low.astype(np.float32), [0.0]])
        high = np.concatenate([o.high.astype(np.float32), [1.0]])
        self.observation_space = gymnasium.spaces.Dict({
            "observation": gymnasium.spaces.Box(low, high, dtype=np.float32),
            "achieved_goal": gymnasium.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "desired_goal": gymnasium.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        })

    def _wrap(self, obs):
        t_feat = max(0.0, 1.0 - self._t / self.max_steps)
        return {
            "observation": np.append(obs["observation"].astype(np.float32), np.float32(t_feat)),
            "achieved_goal": obs["achieved_goal"].astype(np.float32),
            "desired_goal": obs["desired_goal"].astype(np.float32),
        }

    def reset(self, **kwargs):
        self._t = 0
        obs, info = self.env.reset(**kwargs)
        return self._wrap(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        return self._wrap(obs), reward, terminated, truncated, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_slide")
    p.add_argument("--env-id", default=None, help="Override env id (e.g. FetchSlide-v1 to match a v1-trained model).")
    p.add_argument("--hf-repo", default=None, help="Hugging Face repo id of the SB3 checkpoint.")
    p.add_argument("--hf-filename", default="best_model.zip")
    p.add_argument("--sb3-path", type=Path, default=None, help="Local .zip checkpoint (overrides --hf-repo).")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--time-feature", action="store_true",
                   help="Wrap env with the RL-Zoo TimeFeature+float32 adapter (needed for zoo Fetch models).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Episode length / time-feature horizon (defaults to the task's; zoo Fetch models used 50).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    task = resolve_task(args.task, None)
    if args.sb3_path is not None:
        sb3_path = args.sb3_path
    elif args.hf_repo is not None:
        from huggingface_hub import hf_hub_download

        sb3_path = Path(hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename))
    else:
        raise SystemExit("Provide --hf-repo or --sb3-path.")

    out = args.out or Path(f"runs/{task.slug}/videos/reference_{task.slug}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or task.max_episode_steps
    env_id = args.env_id or task.env_id
    env = make_env(
        env_id, seed=args.seed, max_episode_steps=max_steps,
        render_mode="rgb_array", width=args.width, height=args.height,
    )
    if args.time_feature:
        env = TimeFeatureFloat32(env, max_steps=max_steps)
    # Replay buffer is nulled at load, so no env is needed at load time; the
    # wrapped env above supplies the right obs format at predict time.
    policy = SB3Policy(sb3_path, name="reference")
    frames, successes = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            action = policy.act(obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        successes.append(float(info.get("is_success", 0.0)))
    env.close()

    imageio.mimsave(out, frames, fps=args.fps, format="FFMPEG")
    print(
        f"wrote {out}  policy={policy.name}  episodes={args.episodes}  "
        f"success={np.mean(successes):.2f}  frames={len(frames)}"
    )


if __name__ == "__main__":
    main()
