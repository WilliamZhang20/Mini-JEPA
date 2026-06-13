from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import imageio.v2 as imageio
import numpy as np
import torch
from torch.nn import functional as F

from .data import Normalizer, scripted_action
from .envs import (
    ObsSpec,
    flatten_obs,
    goal_reach_distance,
    goal_state_from_state,
    make_env,
    obs_spec_from_env,
)
from .models import ActionConditionedJEPA
from .tasks import resolve_task, task_dir


def patch_numpy_pickle_aliases() -> None:
    """Register ``numpy._core`` module aliases so older SB3 pickles unpickle on newer NumPy."""
    try:
        import numpy.core as numpy_core
    except ImportError:
        return

    import importlib
    import sys

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("numeric", "multiarray", "umath", "fromnumeric", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def install_gym_compat_shim() -> None:
    """Alias the legacy ``gym`` package to ``gymnasium`` so old SB3-zoo pickles load.

    RL-Zoo checkpoints trained on the ``-v1`` Fetch envs were pickled with OpenAI
    ``gym`` and reference ``gym.spaces.*`` classes. We map those module paths onto
    ``gymnasium`` (which has the same submodule layout) so unpickling resolves.
    """
    import importlib
    import sys

    try:
        import gymnasium
    except ImportError:
        return
    sys.modules.setdefault("gym", gymnasium)
    for sub in ("spaces", "spaces.box", "spaces.dict", "spaces.discrete",
                "spaces.multi_discrete", "spaces.multi_binary", "spaces.tuple",
                "spaces.space", "spaces.utils"):
        try:
            sys.modules.setdefault(f"gym.{sub}", importlib.import_module(f"gymnasium.{sub}"))
        except ImportError:
            continue


class Policy(Protocol):
    """Structural interface for an evaluation policy: a name plus an ``act(obs, env)`` method."""

    name: str

    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        ...


@dataclass
class RandomPolicy:
    """Baseline policy that samples uniformly from the environment's action space."""

    name: str = "random"

    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        return env.action_space.sample().astype(np.float32)


@dataclass
class ScriptedGoalPolicy:
    """Hand-coded geometric controller (reach/push/pick_place) used as a teacher and baseline."""

    action_dim: int
    controller: str
    gain: float = 12.0
    name: str = "scripted"

    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        rng = np.random.default_rng(0)
        action = scripted_action(obs, self.action_dim, self.gain, self.controller, env, rng)
        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


class SB3Policy:
    """Wraps a Stable-Baselines3 / sb3-contrib checkpoint (TQC/DDPG/SAC/TD3/PPO) as a policy.

    ``env`` must be supplied for HER checkpoints (e.g. the RL-Zoo Fetch models),
    whose ``HerReplayBuffer`` requires an environment at load time.
    """

    def __init__(self, path: Path, name: str = "sb3", env=None) -> None:
        from stable_baselines3 import DDPG, PPO, SAC, TD3

        patch_numpy_pickle_aliases()
        install_gym_compat_shim()
        self.name = name
        # Override pickled schedules/buffers that don't survive version/Python
        # changes; none of them are needed for deterministic inference.
        custom_objects = {
            "action_noise": None,
            "replay_buffer": None,
            # Drop the (HER) replay buffer entirely: not needed for inference and
            # its saved kwargs (e.g. ``online_sampling``) are stale across versions.
            "replay_buffer_class": None,
            "replay_buffer_kwargs": {},
            "_last_obs": None,
            "_last_episode_starts": None,
            "_vec_normalize_env": None,
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "exploration_schedule": lambda _: 0.0,
        }
        algos = [DDPG, SAC, TD3, PPO]
        try:
            # TQC (sb3-contrib) is the RL-Zoo algorithm behind the strongest
            # pretrained Fetch checkpoints; try it first when available.
            from sb3_contrib import TQC

            algos.insert(0, TQC)
        except ImportError:
            pass
        errors = []
        for cls in algos:
            try:
                self.model = cls.load(path, env=env, custom_objects=custom_objects)
                self.name = f"{name}_{cls.__name__.lower()}"
                return
            except Exception as exc:  # pragma: no cover - depends on external checkpoints.
                errors.append(f"{cls.__name__}: {exc}")
        raise RuntimeError(f"Could not load SB3 checkpoint {path}: {'; '.join(errors)}")

    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class JEPAMPCPolicy:
    """Model-predictive controller that plans action sequences by scoring rollouts of the JEPA world model."""

    def __init__(
        self,
        *,
        model: ActionConditionedJEPA,
        normalizer: Normalizer,
        spec: ObsSpec,
        device: torch.device,
        candidates: int,
        horizon: int,
        seed: int,
        method: str = "random",
        score_mode: str = "latent",
        cem_iters: int = 3,
        elite_frac: float = 0.1,
        action_std: float = 0.7,
        manip_reach_weight: float = 0.4,
        manip_path_weight: float = 0.3,
        manip_align_weight: float = 0.0,
        manip_grasp_weight: float = 0.0,
        action_l2_weight: float = 0.0,
        action_delta_weight: float = 0.0,
        execute_smoothing: float = 0.0,
        warm_start_cem: bool = True,
        grad_iters: int = 30,
        grad_lr: float = 0.08,
        scripted_proposal_fraction: float = 0.0,
        teacher_correction_fraction: float = 0.0,
        teacher_correction_threshold: float = 0.0,
        scripted_gain: float = 12.0,
        scripted_controller: str = "reach",
        policy_net=None,
        policy_proposal_fraction: float = 0.0,
    ) -> None:
        self.name = f"jepa_mpc_{method}_{score_mode}"
        if action_l2_weight > 0.0 or action_delta_weight > 0.0 or execute_smoothing > 0.0:
            self.name = f"{self.name}_smooth"
        if scripted_proposal_fraction > 0.0:
            proposal_pct = int(round(scripted_proposal_fraction * 100))
            self.name = f"{self.name}_proposal{proposal_pct}"
        if teacher_correction_fraction > 0.0:
            teacher_pct = int(round(teacher_correction_fraction * 100))
            if np.isinf(teacher_correction_threshold):
                self.name = f"{self.name}_teacher{teacher_pct}"
            else:
                threshold_mm = int(round(teacher_correction_threshold * 1000))
                self.name = f"{self.name}_teacher{teacher_pct}_{threshold_mm}mm"
        self.model = model
        self.normalizer = normalizer
        self.spec = spec
        self.device = device
        self.candidates = candidates
        self.horizon = horizon
        self.method = method
        self.score_mode = score_mode
        self.cem_iters = cem_iters
        self.elite_frac = elite_frac
        self.action_std = action_std
        self.manip_reach_weight = manip_reach_weight
        self.manip_path_weight = manip_path_weight
        self.manip_align_weight = manip_align_weight
        self.manip_grasp_weight = manip_grasp_weight
        self.action_l2_weight = action_l2_weight
        self.action_delta_weight = action_delta_weight
        self.execute_smoothing = execute_smoothing
        self.warm_start_cem = warm_start_cem
        self.grad_iters = grad_iters
        self.grad_lr = grad_lr
        self.scripted_proposal_fraction = scripted_proposal_fraction
        self.teacher_correction_fraction = float(np.clip(teacher_correction_fraction, 0.0, 1.0))
        self.teacher_correction_threshold = float(teacher_correction_threshold)
        self.scripted_gain = scripted_gain
        self.scripted_controller = scripted_controller
        self.policy_net = policy_net
        self.policy_proposal_fraction = float(np.clip(policy_proposal_fraction, 0.0, 1.0))
        if policy_net is not None and policy_proposal_fraction > 0.0:
            self.name = f"{self.name}_policy{int(round(policy_proposal_fraction * 100))}"
        self.rng = np.random.default_rng(seed)
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.prev_plan: np.ndarray | None = None

    def _manip_scores(
        self,
        raw_state: np.ndarray,
        z: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Staged grasp-aware manipulation cost over the whole predicted trajectory.

        Rolls the latent dynamics out for the full horizon, decodes every
        intermediate state, and combines several dense sub-costs that together
        encode the reach -> align -> grasp -> lift -> transport phases of a pick.

        Why this is needed: the object-to-goal distance is *flat* until the
        object is actually grasped (the block does not move on its own), so a
        planner that scores only that distance gets no gradient and never
        discovers the grasp. Each sub-cost below is dense in a different phase:

        * ``align``  - gripper x/y over the object (so a descent can grasp it),
        * ``reach``  - full 3D gripper->object distance,
        * ``grasp``  - *close the fingers when the gripper is on the object*;
          this is the catalyst term. Without it the gripper command oscillates
          and the block is never picked up.
        * ``path``/``terminal`` - object-to-goal distance (the real objective),
          which becomes informative only once the grasp makes the object movable.
        """
        traj_z = self.model.predict_rollout(z, action_tensor, self.horizon)
        pred_state = self.normalizer.decode_tensor(self.model.state_probe(traj_z))

        grip = pred_state[..., :3]
        obj_start = self.spec.obs_dim
        obj = pred_state[..., obj_start : obj_start + self.spec.goal_dim]
        desired_start = self.spec.obs_dim + self.spec.goal_dim
        desired = torch.as_tensor(
            raw_state[desired_start : desired_start + self.spec.goal_dim],
            dtype=pred_state.dtype,
            device=pred_state.device,
        ).view(1, 1, -1)

        gd = min(3, self.spec.goal_dim)
        obj_to_goal = torch.linalg.norm(obj - desired, dim=-1)                 # [B, H]
        grip_to_obj = torch.linalg.norm(grip[..., :gd] - obj[..., :gd], dim=-1)  # [B, H]
        align_xy = torch.linalg.norm(grip[..., :2] - obj[..., :2], dim=-1)       # [B, H]

        terminal = obj_to_goal[:, -1]
        path = obj_to_goal.mean(dim=1)
        reach = grip_to_obj.mean(dim=1)
        align = align_xy.mean(dim=1)
        scores = (
            terminal
            + self.manip_path_weight * path
            + self.manip_reach_weight * reach
            + self.manip_align_weight * align
        )

        # Grasp catalyst: when the gripper is on the object, reward closing the
        # fingers. Gripper command > 0 opens, < 0 closes, so we penalise an open
        # command weighted by how close the gripper is to the object.
        if self.manip_grasp_weight > 0.0 and action_tensor.shape[-1] >= 4:
            nearness = torch.exp(-grip_to_obj / 0.04)                          # [B, H], ~1 on the object
            open_cmd = torch.clamp(action_tensor[..., 3], min=-1.0)            # [B, H]
            grasp = (nearness * (open_cmd + 1.0)).mean(dim=1)                  # 0 when closed on object
            scores = scores + self.manip_grasp_weight * grasp

        scores = scores + self._action_regularizers(action_tensor)
        return scores

    def _action_regularizers(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Per-candidate L2 magnitude and step-to-step delta penalties that encourage smooth plans."""
        reg = torch.zeros(action_tensor.shape[0], dtype=action_tensor.dtype, device=action_tensor.device)
        if self.action_l2_weight > 0.0:
            reg = reg + self.action_l2_weight * torch.mean(action_tensor.square(), dim=(1, 2))
        if self.action_delta_weight > 0.0:
            prev = torch.as_tensor(
                self.prev_action,
                dtype=action_tensor.dtype,
                device=action_tensor.device,
            ).view(1, 1, -1)
            first_delta = action_tensor[:, :1] - prev
            seq_delta = action_tensor[:, 1:] - action_tensor[:, :-1]
            if seq_delta.numel() == 0:
                delta_cost = torch.mean(first_delta.square(), dim=(1, 2))
            else:
                delta_cost = torch.cat([first_delta, seq_delta], dim=1).square().mean(dim=(1, 2))
            reg = reg + self.action_delta_weight * delta_cost
        return reg

    def _score_action_tensor(
        self,
        obs: dict[str, np.ndarray],
        action_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate action sequence (lower is better) via the world model and the chosen score mode."""
        raw_state = flatten_obs(obs)
        state = torch.from_numpy(self.normalizer.encode(raw_state)).unsqueeze(0).to(self.device)
        z = self.model.encode(state).repeat(action_tensor.shape[0], 1)

        if self.score_mode == "manip":
            return self._manip_scores(raw_state, z, action_tensor)

        pred_z = self.model.predict(z, action_tensor, self.horizon)

        scores = torch.zeros(action_tensor.shape[0], dtype=pred_z.dtype, device=self.device)
        if self.score_mode in ("latent", "combined"):
            goal_state = goal_state_from_state(raw_state, self.spec)
            goal_norm = torch.from_numpy(self.normalizer.encode(goal_state)).unsqueeze(0).to(self.device)
            goal_z = self.model.encode_target(goal_norm)
            latent_scores = torch.sum(
                (F.normalize(pred_z, dim=-1) - F.normalize(goal_z, dim=-1)) ** 2,
                dim=-1,
            )
            scores = scores + latent_scores
        if self.score_mode in ("state", "combined"):
            pred_state_norm = self.model.state_probe(pred_z)
            pred_state = self.normalizer.decode_tensor(pred_state_norm)
            achieved = pred_state[:, self.spec.obs_dim : self.spec.obs_dim + self.spec.goal_dim]
            desired_start = self.spec.obs_dim + self.spec.goal_dim
            desired = torch.as_tensor(
                raw_state[desired_start : desired_start + self.spec.goal_dim],
                dtype=pred_state.dtype,
                device=pred_state.device,
            ).unsqueeze(0)
            state_scores = torch.linalg.norm(achieved - desired, dim=-1)
            scores = scores + state_scores
        if self.action_l2_weight > 0.0:
            scores = scores + self.action_l2_weight * torch.mean(action_tensor.square(), dim=(1, 2))
        if self.action_delta_weight > 0.0:
            prev = torch.as_tensor(
                self.prev_action,
                dtype=action_tensor.dtype,
                device=action_tensor.device,
            ).view(1, 1, -1)
            first_delta = action_tensor[:, :1] - prev
            seq_delta = action_tensor[:, 1:] - action_tensor[:, :-1]
            if seq_delta.numel() == 0:
                delta_cost = torch.mean(first_delta.square(), dim=(1, 2))
            else:
                delta_cost = torch.cat([first_delta, seq_delta], dim=1).square().mean(dim=(1, 2))
            scores = scores + self.action_delta_weight * delta_cost
        return scores

    def _score_sequences(
        self,
        obs: dict[str, np.ndarray],
        action_seq: np.ndarray,
    ) -> torch.Tensor:
        """Convenience wrapper that scores a NumPy batch of action sequences."""
        return self._score_action_tensor(obs, torch.from_numpy(action_seq).to(self.device))

    def _sample_uniform(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        """Draw candidate action sequences i.i.d. uniformly over the action range, then inject proposals."""
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        action_seq = self.rng.uniform(
            low, high, size=(self.candidates, self.horizon, self.spec.action_dim)
        ).astype(np.float32)
        self._inject_policy_proposals(obs, env, action_seq)
        self._inject_scripted_proposals(obs, env, action_seq)
        return action_seq

    def _inject_scripted_proposals(self, obs, env, action_seq: np.ndarray) -> None:
        """Overwrite a fraction of candidates with the scripted teacher's action (plus jitter)."""
        proposal_count = int(round(action_seq.shape[0] * self.scripted_proposal_fraction))
        if proposal_count <= 0:
            return
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        scripted = scripted_action(
            obs,
            self.spec.action_dim,
            self.scripted_gain,
            self.scripted_controller,
            env,
            self.rng,
        )
        scripted = np.clip(scripted, low, high).astype(np.float32)
        action_seq[:proposal_count] = scripted
        noise = self.rng.normal(
            0.0, 0.1, size=(proposal_count, self.horizon, self.spec.action_dim)
        ).astype(np.float32)
        action_seq[:proposal_count] = np.clip(action_seq[:proposal_count] + noise, low, high)

    @torch.no_grad()
    def _policy_rollout(self, obs: dict[str, np.ndarray], env) -> np.ndarray | None:
        """Open-loop action sequence from the learned policy + world model.

        Encodes the current observation, then alternates ``a = policy(z)`` and a
        one-step latent prediction to imagine a full-horizon plan. This is the
        proposal the sampling planner refines; rolling the policy *through the
        world model* keeps the proposal self-consistent with the dynamics.
        """
        if self.policy_net is None:
            return None
        raw_state = flatten_obs(obs)
        z = self.model.encode(
            torch.from_numpy(self.normalizer.encode(raw_state)).unsqueeze(0).to(self.device)
        )
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        actions = []
        for _ in range(self.horizon):
            a = self.policy_net(z)
            actions.append(a)
            z = self.model.predict_rollout(z, a.unsqueeze(1), 1)[:, -1]
        seq = torch.cat(actions, dim=0).cpu().numpy().astype(np.float32)
        return np.clip(seq, low, high)

    def _inject_policy_proposals(self, obs, env, action_seq: np.ndarray) -> None:
        """Seed a fraction of candidates with the learned-policy rollout (one clean copy, rest jittered)."""
        if self.policy_net is None or self.policy_proposal_fraction <= 0.0:
            return
        count = int(round(action_seq.shape[0] * self.policy_proposal_fraction))
        if count <= 0:
            return
        seq = self._policy_rollout(obs, env)
        if seq is None:
            return
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        action_seq[:count] = seq
        # keep one clean copy of the proposal, jitter the rest for local search
        if count > 1:
            noise = self.rng.normal(0.0, 0.1, size=(count - 1,) + seq.shape).astype(np.float32)
            action_seq[1:count] = np.clip(action_seq[1:count] + noise, low, high)

    def _teacher_action(self, obs, env) -> np.ndarray:
        """Clipped scripted-controller action used for teacher correction/blending."""
        action = scripted_action(
            obs,
            self.spec.action_dim,
            self.scripted_gain,
            self.scripted_controller,
            env,
            self.rng,
        )
        return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

    def _teacher_correction_active(self, obs) -> bool:
        """Whether the scripted teacher should correct the plan now (enabled and within the goal-distance threshold)."""
        if self.teacher_correction_fraction <= 0.0 or not self.spec.is_goal_env:
            return False
        if np.isinf(self.teacher_correction_threshold):
            return True
        distance = goal_reach_distance(flatten_obs(obs), self.spec)
        return distance <= self.teacher_correction_threshold

    def _sample_cem(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        """Cross-Entropy Method planning: iteratively refit a Gaussian over sequences to its elite samples.

        Returns the single best-scoring sequence found across all iterations.
        """
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        policy_seq = self._policy_rollout(obs, env)
        if policy_seq is not None:
            # warm-start the search distribution at the learned proposal
            mean = policy_seq.copy()
        elif self.warm_start_cem and self.prev_plan is not None:
            mean = np.zeros((self.horizon, self.spec.action_dim), dtype=np.float32)
            mean[:-1] = self.prev_plan[1:]
            mean[-1] = self.prev_plan[-1]
        else:
            mean = np.zeros((self.horizon, self.spec.action_dim), dtype=np.float32)
        std = np.full_like(mean, self.action_std)
        elite_count = max(1, int(round(self.candidates * self.elite_frac)))
        best_seq = None
        best_score = float("inf")

        for _ in range(self.cem_iters):
            samples = self.rng.normal(mean, std, size=(self.candidates, self.horizon, self.spec.action_dim))
            samples = np.clip(samples.astype(np.float32), low, high)
            self._inject_policy_proposals(obs, env, samples)
            self._inject_scripted_proposals(obs, env, samples)
            scores = self._score_sequences(obs, samples).detach().cpu().numpy()
            order = np.argsort(scores)
            if float(scores[order[0]]) < best_score:
                best_score = float(scores[order[0]])
                best_seq = samples[order[0]].copy()
            elites = samples[order[:elite_count]]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 0.05)

        if best_seq is None:
            return self._sample_uniform(obs, env)
        return best_seq[None, :, :]

    def _sample_grad(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        """Gradient-based planning: optimize actions through the differentiable world model with Adam.

        Actions are reparameterized through ``tanh`` to stay in bounds; returns
        the best-scoring optimized sequence.
        """
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        center = (high + low) * 0.5
        half = torch.clamp((high - low) * 0.5, min=1e-6)
        candidate_count = max(1, self.candidates)

        if self.prev_plan is not None:
            init = np.zeros((self.horizon, self.spec.action_dim), dtype=np.float32)
            init[:-1] = self.prev_plan[1:]
            init[-1] = self.prev_plan[-1]
            init = np.repeat(init[None], candidate_count, axis=0)
            init += self.rng.normal(0.0, 0.15, size=init.shape).astype(np.float32)
        else:
            init = self.rng.normal(
                0.0,
                self.action_std,
                size=(candidate_count, self.horizon, self.spec.action_dim),
            ).astype(np.float32)
        init = np.clip(init, env.action_space.low, env.action_space.high)
        normalized = (torch.from_numpy(init).to(self.device) - center) / half
        u = torch.atanh(torch.clamp(normalized, -0.999, 0.999)).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=self.grad_lr)

        with torch.enable_grad():
            for _ in range(self.grad_iters):
                optimizer.zero_grad(set_to_none=True)
                action_tensor = center + half * torch.tanh(u)
                scores = self._score_action_tensor(obs, action_tensor)
                loss = scores.mean()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            action_tensor = center + half * torch.tanh(u)
            scores = self._score_action_tensor(obs, action_tensor)
            best = int(torch.argmin(scores).detach().cpu())
            return action_tensor[best : best + 1].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        """Plan with the selected method, optionally blend in the teacher, and execute the first action (receding horizon)."""
        teacher_active = self._teacher_correction_active(obs)
        if teacher_active and self.teacher_correction_fraction >= 0.999:
            action = self._teacher_action(obs, env)
            self.prev_action = action
            self.prev_plan = np.repeat(action[None, :], self.horizon, axis=0)
            return action

        if self.method == "cem":
            action_seq = self._sample_cem(obs, env)
        elif self.method == "grad":
            action_seq = self._sample_grad(obs, env)
        else:
            action_seq = self._sample_uniform(obs, env)
        scores = self._score_sequences(obs, action_seq)
        best = int(torch.argmin(scores).cpu())
        chosen_plan = action_seq[best].copy()
        raw_action = chosen_plan[0].copy()
        if teacher_active:
            teacher_action = self._teacher_action(obs, env)
            raw_action = (
                (1.0 - self.teacher_correction_fraction) * raw_action
                + self.teacher_correction_fraction * teacher_action
            )
        if self.execute_smoothing > 0.0:
            smoothing = float(np.clip(self.execute_smoothing, 0.0, 0.95))
            action = (1.0 - smoothing) * raw_action + smoothing * self.prev_action
        else:
            action = raw_action
        action = np.asarray(action, dtype=np.float32)
        self.prev_action = action
        self.prev_plan = chosen_plan
        return action


class LearnedPolicyOnly:
    """The learned action prior executed directly, with no MPC refinement.

    Serves as the baseline that isolates how much the world-model planner adds
    on top of the behaviour-cloned policy.
    """

    def __init__(self, *, model, policy_net, normalizer, spec, device, name="jepa_policy") -> None:
        self.model = model
        self.policy_net = policy_net
        self.normalizer = normalizer
        self.spec = spec
        self.device = device
        self.name = name

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        raw_state = flatten_obs(obs)
        z = self.model.encode(
            torch.from_numpy(self.normalizer.encode(raw_state)).unsqueeze(0).to(self.device)
        )
        action = self.policy_net(z)[0].cpu().numpy().astype(np.float32)
        return np.clip(action, env.action_space.low, env.action_space.high)


def load_policy_artifact(path: Path, device: torch.device):
    """Load a saved GoalConditionedPolicy checkpoint into eval mode; returns ``(policy, config)``."""
    from .models import GoalConditionedPolicy

    artifact = torch.load(path, map_location=device, weights_only=False)
    cfg = artifact["config"]
    policy = GoalConditionedPolicy(
        latent_dim=int(cfg["latent_dim"]),
        action_dim=int(cfg["action_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    policy.load_state_dict(artifact["policy"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, cfg


def _remap_legacy_state_dict(state_dict: dict, predictor_mode: str) -> dict:
    """Map pre-``transition_depth`` recurrent checkpoints onto the current keys.

    The recurrent predictor used to hold a single ``transition`` MLP; it is now a
    ``transition_blocks`` ModuleList so the depth can be configured. Old
    checkpoints (depth 1) store ``transition.net.*``, which corresponds exactly
    to ``transition_blocks.0.net.*``. Only the ``recurrent`` predictor changed -
    the ``rollout`` predictor still uses a plain ``transition`` and is untouched.
    """
    if predictor_mode != "recurrent":
        return state_dict
    if any(k.startswith("transition_blocks.") for k in state_dict):
        return state_dict  # already in the new layout
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("transition.net."):
            remapped["transition_blocks.0." + key[len("transition.") :]] = value
        else:
            remapped[key] = value
    return remapped


def load_jepa_artifact(path: Path, device: torch.device):
    """Rebuild a trained JEPA world model from a checkpoint; returns ``(model, normalizer, spec, config)``."""
    artifact = torch.load(path, map_location=device, weights_only=False)
    spec = ObsSpec(**artifact["spec"])
    config = artifact["config"]
    normalizer = Normalizer(
        mean=np.asarray(artifact["normalizer"]["mean"], dtype=np.float32),
        std=np.asarray(artifact["normalizer"]["std"], dtype=np.float32),
    )
    model = ActionConditionedJEPA(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        max_horizon=int(config["max_horizon"]),
        predictor_mode=str(config.get("predictor_mode", "direct")),
        residual_prediction=bool(config.get("residual_prediction", False)),
        transition_depth=int(config.get("transition_depth", 1)),
    ).to(device)
    state_dict = _remap_legacy_state_dict(
        artifact["model"], str(config.get("predictor_mode", "direct"))
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, normalizer, spec, config


def rollout_policy(
    env,
    policy: Policy,
    *,
    episodes: int,
    seed: int,
    video_path: Path | None = None,
    fps: int = 30,
) -> dict[str, float | str]:
    """Run a policy for ``episodes`` episodes and return aggregate metrics, optionally recording a video."""
    successes = []
    final_distances = []
    episode_lengths = []
    action_norms = []
    action_deltas = []
    frames = []

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        if video_path is not None and episode_idx == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        terminated = truncated = False
        final_info = {}
        steps = 0
        prev_action = None

        while not (terminated or truncated):
            action = policy.act(obs, env)
            action_norms.append(float(np.linalg.norm(action)))
            if prev_action is not None:
                action_deltas.append(float(np.linalg.norm(action - prev_action)))
            prev_action = np.array(action, copy=True)
            obs, _, terminated, truncated, final_info = env.step(action)
            steps += 1
            if video_path is not None and episode_idx == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        successes.append(float(final_info.get("is_success", 0.0)))
        if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
            achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
            desired = np.asarray(obs["desired_goal"], dtype=np.float32)
            final_distances.append(float(np.linalg.norm(achieved - desired)))
        else:
            final_distances.append(float("nan"))
        episode_lengths.append(float(steps))

    metrics: dict[str, float | str] = {
        "policy": policy.name,
        "episodes": float(episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "mean_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
    }
    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=fps, format="FFMPEG")
        metrics["video_path"] = str(video_path)
    return metrics


def make_argparser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the cross-evaluation script."""
    parser = argparse.ArgumentParser(description="Cross-evaluate JEPA against classic baselines.")
    parser.add_argument(
        "--task",
        default=None,
        choices=["fetch_reach", "fetch_pick_place", "fetch_push", "fetch_slide", "adroit_door"],
    )
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--sb3-path", type=Path, default=None, help="Optional pre-trained SB3 .zip checkpoint.")
    parser.add_argument("--hf-sb3-repo", default=None, help="Optional Hugging Face repo id for an SB3 checkpoint.")
    parser.add_argument("--hf-sb3-filename", default="best_model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--mpc-candidates", type=int, default=128)
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--mpc-method", choices=["random", "cem", "grad"], default="random")
    parser.add_argument("--mpc-score", choices=["latent", "state", "combined", "manip"], default="latent")
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--action-std", type=float, default=0.7)
    parser.add_argument("--manip-reach-weight", type=float, default=0.4)
    parser.add_argument("--manip-path-weight", type=float, default=0.3)
    parser.add_argument("--manip-align-weight", type=float, default=0.0)
    parser.add_argument("--manip-grasp-weight", type=float, default=0.0)
    parser.add_argument("--policy-path", type=Path, default=None,
                        help="Learned action-prior artifact; used as the MPC proposal and as a stand-alone baseline.")
    parser.add_argument("--policy-proposal-fraction", type=float, default=0.5,
                        help="Fraction of planner candidates seeded from the learned policy rollout.")
    parser.add_argument("--action-l2-weight", type=float, default=0.0)
    parser.add_argument("--action-delta-weight", type=float, default=0.0)
    parser.add_argument("--execute-smoothing", type=float, default=0.0)
    parser.add_argument("--no-warm-start-cem", action="store_true")
    parser.add_argument("--grad-iters", type=int, default=30)
    parser.add_argument("--grad-lr", type=float, default=0.08)
    parser.add_argument("--scripted-gain", type=float, default=12.0)
    parser.add_argument("--jepa-scripted-proposal-fraction", type=float, default=0.0)
    parser.add_argument(
        "--teacher-correction-fraction",
        type=float,
        default=0.0,
        help="Blend fraction for scripted teacher correction when within the threshold.",
    )
    parser.add_argument(
        "--teacher-correction-threshold",
        type=float,
        default=0.0,
        help="Goal distance threshold for teacher correction. Use inf to match the scripted controller.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument(
        "--video-policy",
        default="jepa_mpc",
        help="Policy name to record, `jepa_mpc` prefix, `all`, or `none`.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=None, help="Recording width in px (env default 480).")
    parser.add_argument("--height", type=int, default=None, help="Recording height in px (env default 480).")
    return parser


def main() -> None:
    """Load the model, evaluate every baseline and the JEPA-MPC policy on a task, and write JSONL results."""
    args = make_argparser().parse_args()
    task = resolve_task(args.task, args.env_id)
    args.task = task.name
    args.env_id = task.env_id
    if args.max_episode_steps is None:
        args.max_episode_steps = task.max_episode_steps
    out_dir = task_dir(args.output_root, task)
    if args.out is None:
        args.out = out_dir / "eval_results" / f"{task.slug}_cross_eval.jsonl"
    if args.video_dir is None:
        args.video_dir = out_dir / "videos"
    if args.model_path is None:
        args.model_path = out_dir / "checkpoints" / f"{task.slug}_jepa_model.pt"
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, normalizer, spec, config = load_jepa_artifact(args.model_path, device)
    env = make_env(args.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    env_spec = obs_spec_from_env(env)
    if env_spec != spec:
        raise ValueError(f"Model spec {spec} does not match env spec {env_spec}.")

    sb3_path = args.sb3_path
    if args.hf_sb3_repo is not None:
        from huggingface_hub import hf_hub_download

        sb3_path = Path(hf_hub_download(repo_id=args.hf_sb3_repo, filename=args.hf_sb3_filename))

    policy_net = None
    if args.policy_path is not None:
        policy_net, _ = load_policy_artifact(args.policy_path, device)

    policies: list[Policy] = [RandomPolicy()]
    if spec.is_goal_env:
        policies.append(
            ScriptedGoalPolicy(
                action_dim=spec.action_dim,
                controller=task.controller,
                gain=args.scripted_gain,
            )
        )
    if policy_net is not None:
        policies.append(
            LearnedPolicyOnly(
                model=model, policy_net=policy_net, normalizer=normalizer, spec=spec, device=device
            )
        )
    policies.append(
        JEPAMPCPolicy(
            model=model,
            normalizer=normalizer,
            spec=spec,
            device=device,
            candidates=args.mpc_candidates,
            horizon=args.mpc_horizon,
            seed=args.seed + 10_000,
            method=args.mpc_method,
            score_mode=args.mpc_score,
            cem_iters=args.cem_iters,
            elite_frac=args.elite_frac,
            action_std=args.action_std,
            manip_reach_weight=args.manip_reach_weight,
            manip_path_weight=args.manip_path_weight,
            manip_align_weight=args.manip_align_weight,
            manip_grasp_weight=args.manip_grasp_weight,
            action_l2_weight=args.action_l2_weight,
            action_delta_weight=args.action_delta_weight,
            execute_smoothing=args.execute_smoothing,
            warm_start_cem=not args.no_warm_start_cem,
            grad_iters=args.grad_iters,
            grad_lr=args.grad_lr,
            scripted_proposal_fraction=args.jepa_scripted_proposal_fraction,
            teacher_correction_fraction=args.teacher_correction_fraction,
            teacher_correction_threshold=args.teacher_correction_threshold,
            scripted_gain=args.scripted_gain,
            scripted_controller=task.controller,
            policy_net=policy_net,
            policy_proposal_fraction=args.policy_proposal_fraction,
        )
    )
    sb3_error = None
    if sb3_path is not None:
        try:
            policies.append(SB3Policy(sb3_path))
        except Exception as exc:
            sb3_error = str(exc)
            print(
                json.dumps(
                    {
                        "event": "sb3_load_failed",
                        "path": str(sb3_path),
                        "error": sb3_error,
                    }
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in policies:
        env.close()
        wants_video = (
            args.video_policy == "all"
            or args.video_policy == policy.name
            or (args.video_policy == "jepa_mpc" and policy.name.startswith("jepa_mpc"))
        )
        render_mode = "rgb_array" if wants_video else None
        eval_env = make_env(
            args.env_id,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            render_mode=render_mode,
            width=args.width if wants_video else None,
            height=args.height if wants_video else None,
        )
        video_path = args.video_dir / f"{policy.name}_{task.slug}.mp4" if wants_video else None
        metrics = rollout_policy(
            eval_env,
            policy,
            episodes=args.episodes,
            seed=args.seed,
            video_path=video_path,
            fps=args.fps,
        )
        eval_env.close()
        row = {
            "event": "cross_eval",
            "env_id": args.env_id,
            "device": str(device),
            "model_path": str(args.model_path),
            "model_config": config,
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, default=str))

    with args.out.open("w", encoding="utf-8") as f:
        if sb3_error is not None:
            f.write(
                json.dumps(
                    {
                        "event": "sb3_load_failed",
                        "path": str(sb3_path),
                        "error": sb3_error,
                    }
                )
                + "\n"
            )
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps({"event": "saved_eval", "path": str(args.out)}))


if __name__ == "__main__":
    main()
