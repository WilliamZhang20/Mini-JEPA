"""Evaluate flat-task inverse prior with demo-future retrieval and JEPA ranking."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.futures import DemoLockedFutureIndex, NearestFutureIndex
from jepa_robotics.algos.priors import InversePrior, append_emphasis


class FlatInversePolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        prior,
        ckpt,
        future_index,
        device,
        possession_prior=None,
        possession_ckpt=None,
        possession_switch_threshold: float = 0.06,
        possession_hysteresis: float = 0.0,
        possession_enter_delay: int = 0,
        candidates: int,
        noise_std: float,
        exec_k: int,
        target_horizon: int | None,
        latent_weight: float,
        state_weight: float,
        action_delta_weight: float,
        action_scale: float,
    ) -> None:
        self.name = f"flat_inverse_jepa_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.prior = prior
        self.ckpt = ckpt
        self.future_index = future_index
        self.device = device
        self.possession_prior = possession_prior
        self.possession_ckpt = possession_ckpt
        self.possession_switch_threshold = possession_switch_threshold
        self.possession_hysteresis = possession_hysteresis
        self.possession_enter_delay = possession_enter_delay
        self._in_possession_mode = False
        self._possession_replans = 0
        self.candidates = candidates
        self.noise_std = noise_std
        self.exec_k = exec_k
        self.target_horizon = target_horizon
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.action_delta_weight = action_delta_weight
        self.action_scale = action_scale
        self.cached: list[np.ndarray] = []
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.cached = []
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        self._in_possession_mode = False
        self._possession_replans = 0
        if hasattr(self.future_index, "reset"):
            self.future_index.reset()

    @torch.no_grad()
    def _plan(self, obs, env):
        raw = flatten_obs(obs)
        prior, ckpt = self.prior, self.ckpt
        if self.possession_prior is not None:
            palm_ball = float(np.linalg.norm(raw[30:33]))
            enter = self.possession_switch_threshold
            leave = self.possession_switch_threshold + self.possession_hysteresis
            if self._in_possession_mode:
                self._in_possession_mode = palm_ball < leave
            else:
                self._in_possession_mode = palm_ball < enter
            if self._in_possession_mode:
                self._possession_replans += 1
                # Hold the reach specialist (which just secured the grasp) for
                # the first enter_delay replans into possession before handing
                # off to the transport specialist, so a marginal fresh grasp is
                # not immediately broken by a transport action.
                if self._possession_replans > self.possession_enter_delay:
                    prior, ckpt = self.possession_prior, self.possession_ckpt
            else:
                self._possession_replans = 0
        target_state = self.future_index.query(raw, self.normalizer)
        s = torch.from_numpy(self.normalizer.encode(raw)).unsqueeze(0).to(self.device)
        tgt = torch.from_numpy(self.normalizer.encode(target_state)).unsqueeze(0).to(self.device)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(tgt)
        horizons = list(ckpt.get("future_horizons", [int(ckpt["H"])]))
        target_h = int(self.target_horizon or max(horizons))
        h_token = torch.tensor([[float(target_h) / float(max(horizons))]], dtype=z.dtype, device=self.device)
        parts = [z, z_goal, h_token]
        if bool(ckpt.get("concat_raw", False)):
            parts.extend([s, tgt])
        append_emphasis(parts, s, ckpt)
        cond = torch.cat(parts, dim=-1)
        base = prior(cond).view(1, int(ckpt["H"]), int(ckpt["action_dim"])) * self.action_scale
        if self.candidates > 1 and self.noise_std > 0:
            actions = base.repeat(self.candidates, 1, 1) + torch.randn(self.candidates, *base.shape[1:], device=self.device) * self.noise_std
            actions[0] = base[0]
        else:
            actions = base.repeat(self.candidates, 1, 1)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        actions = actions.clamp(low, high)
        if actions.shape[0] == 1 and self.state_weight <= 0 and self.action_delta_weight <= 0:
            return actions[0].detach().cpu().numpy().astype(np.float32)
        traj_z = self.wm.predict_rollout(z.repeat(actions.shape[0], 1), actions, int(self.ckpt["H"]))
        scores = self.latent_weight * torch.sum(
            (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        if self.state_weight > 0:
            pred = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            target_t = torch.as_tensor(target_state, dtype=pred.dtype, device=self.device).view(1, 1, -1)
            dist = torch.linalg.norm(pred - target_t, dim=-1)
            scores = scores + self.state_weight * (dist[:, -1] + 0.25 * dist.mean(dim=1))
        if self.action_delta_weight > 0:
            prev = torch.as_tensor(self.prev_action, dtype=torch.float32, device=self.device).view(1, 1, -1)
            delta = torch.cat([actions[:, :1] - prev, actions[:, 1:] - actions[:, :-1]], dim=1)
            scores = scores + self.action_delta_weight * delta.square().mean(dim=(1, 2))
        best = int(torch.argmin(scores).detach().cpu())
        return actions[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs, env):
        if not self.cached:
            plan = self._plan(obs, env)
            k = max(1, min(self.exec_k, len(plan)))
            self.cached = [plan[i].copy() for i in range(k)]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
        self.prev_action = action
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--inverse-path", type=Path, required=True)
    p.add_argument("--inverse-possession-path", type=Path, default=None,
                   help="Optional specialist inverse used while the possession predicate holds (palm-ball norm below the switch threshold); --inverse-path then serves reach/regrasp.")
    p.add_argument("--possession-switch-threshold", type=float, default=0.06)
    p.add_argument("--possession-hysteresis", type=float, default=0.0,
                   help="Extra palm-ball slack before switching back OUT of the possession specialist, preventing specialist dithering at the grasp boundary.")
    p.add_argument("--possession-enter-delay", type=int, default=0,
                   help="Keep the reach specialist for this many replans after first entering possession before handing off to the transport specialist, so a marginal fresh grasp is secured before a transport action is issued.")
    p.add_argument("--log-episodes", action="store_true",
                   help="Print a per-episode diagnostic row: grasp achieved, drops after possession, final ball-target distance.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.0)
    p.add_argument("--action-delta-weight", type=float, default=0.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--future-index", choices=["nearest", "demo_locked"], default="nearest")
    p.add_argument("--future-episodes-npz", type=Path, default=None,
                   help="Episode npz supplying full demo trajectories for the demo_locked future index.")
    p.add_argument("--future-locality-weight", type=float, default=0.0)
    p.add_argument("--future-possession-gate", action="store_true",
                   help="demo_locked: while palm-ball distance (obs dims 30:33) exceeds the threshold, cap the future target at the demo's first-possession frame so transport futures are only requested once the ball is actually held.")
    p.add_argument("--future-possession-threshold", type=float, default=0.06)
    p.add_argument("--future-geom-weight", type=float, default=1.0,
                   help="demo_locked: upweight task-geometry dims (30:39) in the matching distance so demo (re-)locking tracks grasp geometry.")
    p.add_argument("--future-relock-margin", type=float, default=0.0,
                   help="demo_locked: re-lock to another demo when its weighted match beats the locked demo by this margin (recovery after drops).")
    p.add_argument("--torch-seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video-out", type=Path, default=None,
                   help="Save an mp4 of evaluated episodes.")
    p.add_argument("--video-episodes", type=int, default=1,
                   help="Number of consecutive episodes to concatenate into the video.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    ckpt = torch.load(args.inverse_path, map_location=dev, weights_only=False)
    prior = InversePrior(int(ckpt["cond_dim"]), int(ckpt["chunk_dim"]), int(ckpt["hidden"]), int(ckpt["n_blocks"])).to(dev)
    prior.load_state_dict(ckpt["state_dict"])
    prior.eval()
    possession_prior = None
    possession_ckpt = None
    if args.inverse_possession_path is not None:
        possession_ckpt = torch.load(args.inverse_possession_path, map_location=dev, weights_only=False)
        possession_prior = InversePrior(
            int(possession_ckpt["cond_dim"]),
            int(possession_ckpt["chunk_dim"]),
            int(possession_ckpt["hidden"]),
            int(possession_ckpt["n_blocks"]),
        ).to(dev)
        possession_prior.load_state_dict(possession_ckpt["state_dict"])
        possession_prior.eval()
    if args.future_index == "demo_locked":
        if args.future_episodes_npz is None:
            raise ValueError("--future-index demo_locked requires --future-episodes-npz")
        horizons = list(ckpt.get("future_horizons", [int(ckpt["H"])]))
        future_index = DemoLockedFutureIndex(
            [ep.states for ep in load_episodes_npz(args.future_episodes_npz)],
            norm,
            horizon=int(args.target_horizon or max(horizons)),
            locality_weight=args.future_locality_weight,
            predicate_dims=(30, 33) if args.future_possession_gate else None,
            predicate_threshold=args.future_possession_threshold,
            geom_dims=(30, 39),
            geom_weight=args.future_geom_weight,
            relock_margin=args.future_relock_margin,
        )
    else:
        future_index = NearestFutureIndex(np.asarray(ckpt["bank_states"]), np.asarray(ckpt["bank_futures"]), norm)
    render = args.video_out is not None
    env = make_env(task.env_id, seed=args.seed,
                   max_episode_steps=args.max_episode_steps or task.max_episode_steps,
                   render_mode="rgb_array" if render else None,
                   width=args.width if render else None, height=args.height if render else None)
    video_frames = []
    policy = FlatInversePolicy(
        wm=wm, normalizer=norm, spec=spec, prior=prior, ckpt=ckpt, future_index=future_index,
        device=dev, possession_prior=possession_prior, possession_ckpt=possession_ckpt,
        possession_switch_threshold=args.possession_switch_threshold,
        candidates=args.candidates, noise_std=args.noise_std,
        possession_hysteresis=args.possession_hysteresis,
        possession_enter_delay=args.possession_enter_delay,
        exec_k=args.exec_k, target_horizon=args.target_horizon,
        latent_weight=args.latent_weight, state_weight=args.state_weight,
        action_delta_weight=args.action_delta_weight, action_scale=args.action_scale,
    )
    successes = []
    episode_returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()
        ep_states = [flatten_obs(obs).copy()]
        term = trunc = False
        info = {}
        episode_return = 0.0
        capture = render and ep < args.video_episodes
        frames = []
        if capture:
            f = env.render()
            if f is not None:
                frames.append(f)
        while not (term or trunc):
            obs, reward, term, trunc, info = env.step(policy.act(obs, env))
            episode_return += float(reward)
            ep_states.append(flatten_obs(obs).copy())
            if capture:
                f = env.render()
                if f is not None:
                    frames.append(f)
        success = float(info.get("is_success", info.get("success", 0.0)))
        successes.append(success)
        episode_returns.append(episode_return)
        if capture:
            video_frames.extend(frames)
        if args.log_episodes:
            traj = np.asarray(ep_states, dtype=np.float32)
            palm_ball = np.linalg.norm(traj[:, 30:33], axis=-1)
            ball_target = np.linalg.norm(traj[:, 36:39], axis=-1)
            possessed = palm_ball < 0.06
            print(
                json.dumps(
                    {
                        "event": "episode_diag",
                        "episode": ep,
                        "success": success,
                        "grasped": bool(possessed.any()),
                        "possession_steps": int(possessed.sum()),
                        "drops_after_possession": int(np.sum(possessed[:-1] & ~possessed[1:])),
                        "min_palm_ball": float(palm_ball.min()),
                        "final_ball_target": float(ball_target[-1]),
                        "min_ball_target": float(ball_target.min()),
                    }
                ),
                flush=True,
            )
    env.close()
    if render and video_frames:
        import imageio.v2 as imageio
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video_out, video_frames, fps=args.fps, format="FFMPEG")
        print(json.dumps({"event": "video_saved", "path": str(args.video_out),
                          "episodes": min(args.video_episodes, args.episodes)}), flush=True)
    row = {
        "event": "flat_inverse_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "inverse_path": str(args.inverse_path),
        "model_config": cfg,
        "policy": policy.name,
        "episodes": float(args.episodes),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0,
        "candidates": int(args.candidates),
        "noise_std": float(args.noise_std),
        "exec_k": int(args.exec_k),
        "target_horizon": args.target_horizon,
        "future_index": args.future_index,
        "future_episodes_npz": str(args.future_episodes_npz) if args.future_episodes_npz is not None else None,
        "future_possession_gate": bool(args.future_possession_gate),
        "future_possession_threshold": float(args.future_possession_threshold),
        "future_geom_weight": float(args.future_geom_weight),
        "future_relock_margin": float(args.future_relock_margin),
        "inverse_possession_path": str(args.inverse_possession_path) if args.inverse_possession_path is not None else None,
        "possession_switch_threshold": float(args.possession_switch_threshold),
        "possession_hysteresis": float(args.possession_hysteresis),
        "torch_seed": args.torch_seed,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
