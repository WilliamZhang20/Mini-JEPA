"""Collect contact-balanced, reward-free dexterous play.

Candidate chunks come from a state-conditioned play prior. A frozen JEPA world
model predicts their outcomes, and a count-based intrinsic score favors
under-visited combinations of:

* dominant object-rotation axis/sign,
* rotation magnitude,
* tactile contact density,
* object/contact latent novelty.

No desired goal, environment reward, success label, demonstration flag, policy
gradient, or task-specific recovery rule enters collection. Every trajectory is
saved, including drops and ineffective attempts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.algos.priors import make_ddpm, sample_action_chunks
from jepa_robotics.data import (
    Episode,
    load_episodes_npz,
    save_episodes_npz,
)
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import DexterousFlowPrior
from jepa_robotics.tasks import resolve_task


def quaternion_bins(
    current: torch.Tensor, future: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed dominant-axis bin (0..5) and angle bin (0..2)."""
    current = F.normalize(current, dim=-1)
    future = F.normalize(future, dim=-1)
    aw, av = future[:, :1], future[:, 1:]
    bw, bv = current[:, :1], -current[:, 1:]
    rel_w = aw * bw - (av * bv).sum(-1, keepdim=True)
    rel_v = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
    sign = torch.where(rel_w < 0, -1.0, 1.0)
    rel_w, rel_v = rel_w * sign, rel_v * sign
    angle = 2.0 * torch.atan2(
        torch.linalg.vector_norm(rel_v, dim=-1),
        rel_w[:, 0].clamp_min(1e-7),
    )
    dominant = rel_v.abs().argmax(-1)
    component = rel_v.gather(1, dominant[:, None])[:, 0]
    axis = 2 * dominant + (component < 0).long()
    angle_deg = torch.rad2deg(angle)
    magnitude = torch.bucketize(
        angle_deg,
        torch.tensor([3.0, 8.0], device=angle_deg.device),
    )
    return axis, magnitude


class IntrinsicPlayExplorer:
    def __init__(
        self,
        model,
        normalizer,
        spec,
        config,
        flow,
        flow_config,
        device,
        low,
        high,
        *,
        candidates: int,
        sample_steps: int,
        execute: int,
        slew_limit: float,
        archive_size: int,
        seed: int,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.spec = spec
        self.config = config
        self.flow = flow
        self.flow_config = flow_config
        self.device = device
        self.low, self.high = low, high
        self.candidates = candidates
        self.sample_steps = sample_steps
        self.execute = execute
        self.slew_limit = slew_limit
        self.archive_size = archive_size
        self.rng = np.random.default_rng(seed)
        self.context_len = int(model.context_len)
        self.horizon = int(flow_config["chunk"])
        self.action_dim = spec.action_dim
        self.ddpm = make_ddpm(int(flow_config["flow_steps"]), device)
        self.object_lo = spec.obs_dim
        self.object_hi = spec.obs_dim + 7
        contact_dims = config.get("contact_dims")
        if not contact_dims:
            raise ValueError("Intrinsic tactile play requires contact_dims")
        self.contact_lo, self.contact_hi = (int(v) for v in contact_dims)
        self.contact_mean = torch.as_tensor(
            normalizer.mean[self.contact_lo:self.contact_hi],
            dtype=torch.float32,
            device=device,
        )
        self.contact_std = torch.as_tensor(
            normalizer.std[self.contact_lo:self.contact_hi],
            dtype=torch.float32,
            device=device,
        )
        self.increment_mean = torch.as_tensor(
            flow_config["increment_mean"], dtype=torch.float32, device=device
        )
        self.increment_std = torch.as_tensor(
            flow_config["increment_std"], dtype=torch.float32, device=device
        )
        # signed axis x magnitude x tactile-density mode
        self.counts = np.zeros((6, 3, 4), dtype=np.int64)
        self.archive: list[torch.Tensor] = []
        self.total_plans = 0
        self.reset()

    def reset(self) -> None:
        self.state_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.previous = np.zeros(self.action_dim, dtype=np.float32)
        self.buffer: list[np.ndarray] = []
        self.chunk_start_pose: np.ndarray | None = None

    def _observe(self, raw: np.ndarray) -> None:
        if self.state_history:
            self.action_history.append(self.previous.copy())
        self.state_history.append(self.normalizer.encode(raw))
        self.state_history = self.state_history[-self.context_len :]
        self.action_history = self.action_history[
            -(self.context_len - 1) :
        ]

    def _context(self):
        states = [self.state_history[0]] * (
            self.context_len - len(self.state_history)
        ) + self.state_history
        actions = [np.zeros(self.action_dim, dtype=np.float32)] * (
            self.context_len - 1 - len(self.action_history)
        ) + self.action_history
        state = torch.from_numpy(np.stack(states)).unsqueeze(0).to(self.device)
        action = torch.from_numpy(np.stack(actions)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.model.encode(
                state.reshape(self.context_len, -1)
            ).reshape(1, self.context_len, -1)
        return latent, action

    def _contact_mode(self, contact: torch.Tensor) -> torch.Tensor:
        active = (contact.abs() > 1e-6).sum(-1)
        return torch.bucketize(
            active,
            torch.tensor([1, 5, 13], device=active.device),
        )

    def _latent_feature(self, latent: torch.Tensor) -> torch.Tensor:
        slot_dim = self.model.slot_dim
        object_pose = latent[..., :7]
        contact_slot = latent[..., slot_dim : 2 * slot_dim]
        return F.normalize(torch.cat([object_pose, contact_slot], dim=-1), dim=-1)

    def _update_actual(self, raw: np.ndarray, latent: torch.Tensor) -> None:
        feature = self._latent_feature(latent[:, -1]).detach()
        if len(self.archive) < self.archive_size:
            self.archive.append(feature[0].cpu())
        else:
            index = int(self.rng.integers(self.total_plans + 1))
            if index < self.archive_size:
                self.archive[index] = feature[0].cpu()
        if self.chunk_start_pose is None:
            return
        current_q = torch.from_numpy(
            self.chunk_start_pose[3:]
        ).to(self.device).view(1, 4)
        future_q = torch.from_numpy(
            raw[self.object_lo + 3:self.object_hi]
        ).to(self.device).view(1, 4)
        axis, magnitude = quaternion_bins(current_q, future_q)
        contact = torch.from_numpy(
            raw[self.contact_lo:self.contact_hi]
        ).to(self.device).view(1, -1)
        mode = self._contact_mode(contact)
        self.counts[int(axis), int(magnitude), int(mode)] += 1

    @torch.no_grad()
    def _plan(self, raw: np.ndarray) -> np.ndarray:
        latent_history, action_history = self._context()
        self._update_actual(raw, latent_history)
        condition = torch.cat(
            [latent_history.flatten(1), action_history.flatten(1)], dim=-1
        ).expand(self.candidates, -1)
        sampled = sample_action_chunks(
            self.flow,
            self.ddpm,
            condition,
            int(self.flow_config["chunk_dim"]),
            self.device,
            objective="flow",
            flow_steps=self.sample_steps,
        ).reshape(self.candidates, self.horizon, self.action_dim)
        increments = sampled * self.increment_std + self.increment_mean
        previous = torch.from_numpy(self.previous).to(self.device).view(1, 1, -1)
        actions = previous + torch.cumsum(increments, dim=1)
        if self.slew_limit > 0:
            limited, live = [], previous[:, 0].expand(self.candidates, -1)
            for step in range(self.horizon):
                live = live + (actions[:, step] - live).clamp(
                    -self.slew_limit, self.slew_limit
                )
                limited.append(live)
            actions = torch.stack(limited, dim=1)
        actions = actions.clamp(self.low, self.high)
        rollout = self.model.predict_rollout_context(
            latent_history.expand(self.candidates, -1, -1),
            action_history.expand(self.candidates, -1, -1),
            actions,
            self.horizon,
        )
        endpoint = rollout[:, -1]
        predicted_object_norm = self.model.predict_object(endpoint)
        object_mean = torch.as_tensor(
            self.normalizer.mean[self.object_lo:self.object_hi],
            dtype=torch.float32,
            device=self.device,
        )
        object_std = torch.as_tensor(
            self.normalizer.std[self.object_lo:self.object_hi],
            dtype=torch.float32,
            device=self.device,
        )
        predicted_object = predicted_object_norm * object_std + object_mean
        current_q = torch.from_numpy(
            raw[self.object_lo + 3:self.object_hi]
        ).to(self.device).view(1, 4).expand(self.candidates, -1)
        axis, magnitude = quaternion_bins(current_q, predicted_object[:, 3:])

        predicted_contact_norm = self.model.contact_consistency(endpoint)
        predicted_contact = (
            predicted_contact_norm * self.contact_std + self.contact_mean
        )
        contact_mode = self._contact_mode(predicted_contact)
        count_tensor = torch.as_tensor(self.counts, device=self.device)
        visits = count_tensor[axis, magnitude, contact_mode].float()
        rarity = torch.rsqrt(visits + 1.0)

        feature = self._latent_feature(endpoint)
        if self.archive:
            archive = torch.stack(self.archive).to(self.device)
            novelty = torch.cdist(feature, archive).amin(dim=1)
        else:
            novelty = torch.ones(self.candidates, device=self.device)
        slot_dim = self.model.slot_dim
        contact_change = torch.linalg.vector_norm(
            endpoint[:, slot_dim:2 * slot_dim]
            - latent_history[:, -1, slot_dim:2 * slot_dim],
            dim=-1,
        ) / np.sqrt(slot_dim)
        command_delta = torch.diff(
            torch.cat(
                [previous.expand(self.candidates, -1, -1), actions], dim=1
            ),
            dim=1,
        ).square().mean(dim=(1, 2))
        score = (
            2.0 * rarity
            + 0.35 * novelty
            + 0.15 * contact_change
            + 0.10 * magnitude.float()
            - 0.05 * command_delta
        )
        chosen = int(torch.argmax(score))
        self.chunk_start_pose = raw[self.object_lo:self.object_hi].copy()
        self.total_plans += 1
        return actions[chosen].cpu().numpy()

    def act(self, raw: np.ndarray) -> np.ndarray:
        self._observe(raw)
        if not self.buffer:
            plan = self._plan(raw)
            self.buffer = [
                plan[i].copy()
                for i in range(min(self.execute, len(plan)))
            ]
        action = self.buffer.pop(0).astype(np.float32)
        self.previous = action.copy()
        return action

    def finish(self, raw: np.ndarray) -> None:
        if self.state_history:
            self._observe(raw)
            latent, _ = self._context()
            self._update_actual(raw, latent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--play-prior-path", type=Path, required=True)
    parser.add_argument("--base-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--slew-limit", type=float, default=0.2)
    parser.add_argument("--archive-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    model, normalizer, spec, config = load_jepa_artifact(
        args.model_path, device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    artifact = torch.load(
        args.play_prior_path, map_location=device, weights_only=False
    )
    flow_config = artifact["config"]
    flow = DexterousFlowPrior(
        int(flow_config["chunk_dim"]),
        int(flow_config["condition_dim"]),
        hidden=int(flow_config["hidden"]),
        n_blocks=int(flow_config["blocks"]),
        heads=int(flow_config["heads"]),
        action_dim=int(flow_config["action_dim"]),
    ).to(device)
    flow.load_state_dict(artifact.get("ema", artifact["state_dict"]))
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    task = resolve_task(args.task, None)
    env = make_env(
        task.env_id, seed=args.seed, max_episode_steps=args.episode_length
    )
    live_spec = obs_spec_from_env(env)
    if live_spec != spec:
        raise ValueError("Checkpoint observation spec does not match environment")
    explorer = IntrinsicPlayExplorer(
        model,
        normalizer,
        spec,
        config,
        flow,
        flow_config,
        device,
        torch.as_tensor(env.action_space.low, device=device),
        torch.as_tensor(env.action_space.high, device=device),
        candidates=args.candidates,
        sample_steps=args.sample_steps,
        execute=args.execute,
        slew_limit=args.slew_limit,
        archive_size=args.archive_size,
        seed=args.seed,
    )

    collected: list[Episode] = []
    action_delta_sq: list[float] = []
    total = 0
    episode_id = 0
    while total < args.num_steps:
        observation, _ = env.reset(seed=args.seed + episode_id)
        explorer.reset()
        raw = flatten_obs(observation)
        states, actions = [raw.copy()], []
        previous = np.zeros(spec.action_dim, dtype=np.float32)
        terminated = truncated = False
        while (
            not (terminated or truncated)
            and total < args.num_steps
        ):
            action = explorer.act(raw)
            action_delta_sq.append(float(np.mean((action - previous) ** 2)))
            previous = action
            observation, _, terminated, truncated, _ = env.step(action)
            raw = flatten_obs(observation)
            states.append(raw.copy())
            actions.append(action.copy())
            total += 1
        explorer.finish(raw)
        if len(actions) >= 2:
            collected.append(
                Episode(
                    states=np.asarray(states, dtype=np.float32),
                    actions=np.asarray(actions, dtype=np.float32),
                )
            )
        episode_id += 1
        if episode_id % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "intrinsic_play_collect",
                        "steps": total,
                        "episodes": len(collected),
                        "occupied_bins": int(np.count_nonzero(explorer.counts)),
                        "total_bins": int(explorer.counts.size),
                    }
                ),
                flush=True,
            )
    env.close()
    base = load_episodes_npz(args.base_npz)
    combined = base + collected
    save_episodes_npz(args.out, combined, spec)
    print(
        json.dumps(
            {
                "event": "intrinsic_play_saved",
                "new_episodes": len(collected),
                "new_steps": total,
                "base_episodes": len(base),
                "combined_episodes": len(combined),
                "occupied_bins": int(np.count_nonzero(explorer.counts)),
                "total_bins": int(explorer.counts.size),
                "bin_counts": explorer.counts.tolist(),
                "action_delta_rms": float(
                    np.sqrt(np.mean(action_delta_sq))
                ),
                "out": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
