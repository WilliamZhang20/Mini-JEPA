"""Evaluate flow proposals refined by a residual diffusion/flow model."""
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

from jepa_robotics.algos.contact import ContactTraceCVAE, possession_trust_barrier
from jepa_robotics.algos.futures import DemoLockedFutureIndex
from jepa_robotics.algos.priors import EpsNet, make_ddpm, sample_action_chunks
from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import MLP
from jepa_robotics.tasks import resolve_task
from scripts.eval_flat_future_inverse import NearestFutureIndex


def parse_dims(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


class RelocatePredicateFutureIndex(NearestFutureIndex):
    """Future lookup with Relocate-specific self-supervised geometry predicates.

    Nearest-state future lookup can keep asking for local demo futures even when
    the hand is in the wrong contact mode. Relocate exposes geometry in the flat
    observation, so the high level can choose futures that first reduce
    palm-to-ball distance and then reduce ball-to-target distance while
    maintaining palm-ball contact.
    """

    def __init__(
        self,
        states: np.ndarray,
        futures: np.ndarray,
        normalizer,
        *,
        k: int,
        local_weight: float,
        contact_threshold: float,
        contact_weight: float,
        target_weight: float,
        maintain_weight: float,
    ) -> None:
        super().__init__(states, futures, normalizer)
        self.k = k
        self.local_weight = local_weight
        self.contact_threshold = contact_threshold
        self.contact_weight = contact_weight
        self.target_weight = target_weight
        self.maintain_weight = maintain_weight
        self.future_palm_ball = np.linalg.norm(self.futures[:, 30:33], axis=-1)
        self.future_ball_target = np.linalg.norm(self.futures[:, 36:39], axis=-1)

    def query(self, state: np.ndarray, normalizer) -> np.ndarray:
        x = normalizer.encode(state).astype(np.float32)
        k = min(max(1, self.k), len(self.states))
        if self.tree is not None:
            dist, idx = self.tree.query(x, k=k)
            idx = np.atleast_1d(idx).astype(np.int64)
            dist = np.atleast_1d(dist).astype(np.float32)
        else:
            all_dist = np.linalg.norm(self.norm_states - x[None], axis=1)
            idx = np.argpartition(all_dist, k - 1)[:k]
            dist = all_dist[idx]
        palm_ball = float(np.linalg.norm(state[30:33]))
        if palm_ball > self.contact_threshold:
            pred_score = self.contact_weight * self.future_palm_ball[idx] + 0.25 * self.target_weight * self.future_ball_target[idx]
        else:
            pred_score = self.target_weight * self.future_ball_target[idx] + self.maintain_weight * self.future_palm_ball[idx]
        score = self.local_weight * dist + pred_score
        return self.futures[int(idx[int(np.argmin(score))])]


class FlowResidualPolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        flow,
        flow_ckpt,
        refiner,
        refiner_ckpt,
        relocate_probe,
        relocate_contact_dynamics,
        relocate_contact_dynamics_ckpt,
        relocate_contact_vae,
        relocate_contact_vae_ckpt,
        contact_vae_samples: int,
        contact_vae_uncertainty_weight: float,
        future_index,
        device,
        candidates: int,
        exec_k: int,
        flow_steps: int,
        refine_steps: int,
        target_horizon: int | None,
        latent_weight: float,
        state_weight: float,
        state_dim_weight: float,
        state_dims: list[int],
        action_l2_weight: float,
        action_delta_weight: float,
        relocate_contact_weight: float,
        relocate_target_weight: float,
        relocate_stage_score: bool,
        possession_barrier: float,
        possession_contact_threshold: float,
        possession_keep_threshold: float,
        possession_target_threshold: float,
        possession_target_slack: float,
        stage_contact_threshold: float,
        stage_target_threshold: float,
        stage_reach_weight: float,
        stage_transport_weight: float,
        stage_place_weight: float,
        stage_trust_weight: float,
        residual_scale: float,
        action_scale: float,
        warm_start_frac: float,
        warm_start_tau: float,
        grasp_commit_k: int,
        grasp_commit_low: float,
        grasp_commit_high: float,
        flow_init_noise_scale: float,
        refine_init_noise_scale: float,
        include_unrefined: bool,
        jepa_grad_steps: int,
        jepa_grad_lr: float,
        jepa_anchor_weight: float,
        jepa_smooth_weight: float,
    ) -> None:
        self.name = f"flow_residual_jepa_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.flow = flow
        self.flow_ckpt = flow_ckpt
        self.refiner = refiner
        self.refiner_ckpt = refiner_ckpt
        self.relocate_probe = relocate_probe
        self.relocate_contact_dynamics = relocate_contact_dynamics
        self.relocate_contact_dynamics_ckpt = relocate_contact_dynamics_ckpt
        self.relocate_contact_vae = relocate_contact_vae
        self.relocate_contact_vae_ckpt = relocate_contact_vae_ckpt
        self.contact_vae_samples = contact_vae_samples
        self.contact_vae_uncertainty_weight = contact_vae_uncertainty_weight
        self.future_index = future_index
        self.device = device
        self.candidates = candidates
        self.exec_k = exec_k
        self.flow_steps = flow_steps
        self.refine_steps = refine_steps
        self.target_horizon = target_horizon
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.state_dim_weight = state_dim_weight
        self.state_dims = state_dims
        self.action_l2_weight = action_l2_weight
        self.action_delta_weight = action_delta_weight
        self.relocate_contact_weight = relocate_contact_weight
        self.relocate_target_weight = relocate_target_weight
        self.relocate_stage_score = relocate_stage_score
        self.possession_barrier = possession_barrier
        self.possession_contact_threshold = possession_contact_threshold
        self.possession_keep_threshold = possession_keep_threshold
        self.possession_target_threshold = possession_target_threshold
        self.possession_target_slack = possession_target_slack
        self.stage_contact_threshold = stage_contact_threshold
        self.stage_target_threshold = stage_target_threshold
        self.stage_reach_weight = stage_reach_weight
        self.stage_transport_weight = stage_transport_weight
        self.stage_place_weight = stage_place_weight
        self.stage_trust_weight = stage_trust_weight
        self.residual_scale = residual_scale
        self.action_scale = action_scale
        self.warm_start_frac = warm_start_frac
        self.warm_start_tau = warm_start_tau
        self.grasp_commit_k = grasp_commit_k
        self.grasp_commit_low = grasp_commit_low
        self.grasp_commit_high = grasp_commit_high
        self.flow_init_noise_scale = flow_init_noise_scale
        self.refine_init_noise_scale = refine_init_noise_scale
        self.include_unrefined = include_unrefined
        self.jepa_grad_steps = jepa_grad_steps
        self.jepa_grad_lr = jepa_grad_lr
        self.jepa_anchor_weight = jepa_anchor_weight
        self.jepa_smooth_weight = jepa_smooth_weight
        self.flow_ddpm = make_ddpm(int(flow_ckpt["diffusion_steps"]), device)
        self.refiner_ddpm = make_ddpm(int(refiner_ckpt["diffusion_steps"]), device)
        self.cached: list[np.ndarray] = []
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.prev_plan: torch.Tensor | None = None
        self.last_exec_len = 0

    def reset(self) -> None:
        self.cached = []
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.prev_plan = None
        self.last_exec_len = 0
        if hasattr(self.future_index, "reset"):
            self.future_index.reset()

    def _stage_contact_scores(self, contact: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw_state = getattr(self, "_current_raw_state", None)
        if raw_state is None:
            return torch.zeros(contact.shape[0], dtype=contact.dtype, device=contact.device)
        palm_ball = float(torch.linalg.norm(raw_state[30:33]).detach().cpu())
        ball_target = float(torch.linalg.norm(raw_state[36:39]).detach().cpu())
        contact_violation = torch.relu(contact - self.stage_contact_threshold)
        contact_regress = torch.relu(contact - (palm_ball + 0.01))
        target_regress = torch.relu(target - (ball_target + 0.01))
        if palm_ball > self.stage_contact_threshold:
            # Reach/grasp: first make contact reachable; object-to-target is mostly
            # uninformative until the hand causally controls the object.
            return self.stage_reach_weight * (contact[:, -1] + 0.5 * contact.mean(dim=1)) + self.stage_trust_weight * contact_regress.mean(dim=1)
        if ball_target > self.stage_target_threshold:
            # Transport: reduce object-target distance while staying inside the
            # learned contact trust region.
            return self.stage_transport_weight * (
                target[:, -1]
                + 0.25 * target.mean(dim=1)
                + 0.75 * contact_violation.mean(dim=1)
            ) + self.stage_trust_weight * (contact_regress.mean(dim=1) + target_regress.mean(dim=1))
        # Place/settle: target proximity dominates, but large contact loss can
        # still indicate that the ball was dropped before placement.
        return self.stage_place_weight * (
            target[:, -1]
            + 0.25 * target.mean(dim=1)
            + 0.25 * contact_violation.mean(dim=1)
        )

    @torch.no_grad()
    def _condition(self, raw: np.ndarray, target_state: np.ndarray):
        s_np = self.normalizer.encode(raw).astype(np.float32)
        tgt_np = self.normalizer.encode(target_state).astype(np.float32)
        s = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
        tgt = torch.from_numpy(tgt_np).unsqueeze(0).to(self.device)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(tgt)
        horizons = list(self.flow_ckpt.get("future_horizons", [int(self.flow_ckpt["H"])]))
        target_h = int(self.target_horizon or max(horizons))
        h_token = torch.tensor([[float(target_h) / float(max(horizons))]], dtype=z.dtype, device=self.device)
        parts = [z, z_goal, h_token]
        if bool(self.flow_ckpt.get("concat_raw", False)):
            parts.extend([s, tgt])
        return z, z_goal, torch.cat(parts, dim=-1)

    def _candidate_scores(
        self,
        *,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor,
        target_state: np.ndarray,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H = actions.shape[1]
        contact_scoring = (
            self.relocate_contact_weight > 0
            or self.relocate_target_weight > 0
            or self.relocate_stage_score
            or self.possession_barrier > 0
        )
        needs_rollout = (
            self.latent_weight > 0
            or self.state_weight > 0
            or self.state_dim_weight > 0
            or (self.relocate_probe is not None and contact_scoring)
            or (self.relocate_probe is None and self.relocate_contact_dynamics is None and self.relocate_contact_vae is None and contact_scoring)
        )
        if needs_rollout:
            traj_z = self.wm.predict_rollout(z.repeat(actions.shape[0], 1), actions, H)
        else:
            traj_z = None
        scores = torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
        if self.latent_weight > 0:
            scores = scores + self.latent_weight * torch.sum(
                (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
                dim=-1,
            )
        contact_trace = None
        target_trace = None
        need_pred_state = (
            self.state_weight > 0
            or self.state_dim_weight > 0
            or (self.relocate_probe is None and self.relocate_contact_dynamics is None and self.relocate_contact_vae is None and contact_scoring)
        )
        if need_pred_state:
            pred = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            target_t = torch.as_tensor(target_state, dtype=pred.dtype, device=self.device).view(1, 1, -1)
            if self.state_weight > 0:
                dist = torch.linalg.norm(pred - target_t, dim=-1)
                scores = scores + self.state_weight * (dist[:, -1] + 0.25 * dist.mean(dim=1))
            if self.state_dim_weight > 0 and self.state_dims:
                dims = torch.as_tensor(self.state_dims, dtype=torch.long, device=self.device)
                dim_err = (pred[:, -1, dims] - target_t[:, :, dims].squeeze(1)).square().mean(dim=-1)
                scores = scores + self.state_dim_weight * dim_err
            if self.relocate_contact_weight > 0:
                palm_to_ball = torch.linalg.norm(pred[:, :, 30:33], dim=-1)
                contact_trace = palm_to_ball
                scores = scores + self.relocate_contact_weight * (palm_to_ball[:, -1] + 0.25 * palm_to_ball.mean(dim=1))
            if self.relocate_target_weight > 0:
                ball_to_target = torch.linalg.norm(pred[:, :, 36:39], dim=-1)
                target_trace = ball_to_target
                scores = scores + self.relocate_target_weight * (ball_to_target[:, -1] + 0.25 * ball_to_target.mean(dim=1))
            if contact_scoring and (contact_trace is None or target_trace is None):
                contact_trace = torch.linalg.norm(pred[:, :, 30:33], dim=-1)
                target_trace = torch.linalg.norm(pred[:, :, 36:39], dim=-1)
        if self.relocate_probe is not None and contact_scoring:
            probe_pred = torch.clamp(self.relocate_probe(traj_z.reshape(-1, traj_z.shape[-1])), min=0.0).view(traj_z.shape[0], traj_z.shape[1], 2)
            contact_trace = probe_pred[:, :, 0]
            target_trace = probe_pred[:, :, 1]
            if self.relocate_contact_weight > 0:
                contact = probe_pred[:, :, 0]
                scores = scores + self.relocate_contact_weight * (contact[:, -1] + 0.25 * contact.mean(dim=1))
            if self.relocate_target_weight > 0:
                target = probe_pred[:, :, 1]
                scores = scores + self.relocate_target_weight * (target[:, -1] + 0.25 * target.mean(dim=1))
        if self.relocate_contact_dynamics is not None and contact_scoring:
            raw_cond = getattr(self, "_current_raw_cond", None)
            if raw_cond is None:
                raw_cond = torch.zeros(1, int(self.relocate_contact_dynamics_ckpt["state_dim"]), device=self.device)
            raw_cond = raw_cond.repeat(actions.shape[0], 1)
            dyn_cond = torch.cat([z.repeat(actions.shape[0], 1), raw_cond, actions.flatten(1)], dim=-1)
            dyn = torch.clamp(self.relocate_contact_dynamics(dyn_cond), min=0.0).view(actions.shape[0], actions.shape[1], 2)
            contact_trace = dyn[:, :, 0]
            target_trace = dyn[:, :, 1]
            if self.relocate_contact_weight > 0:
                contact = dyn[:, :, 0]
                scores = scores + self.relocate_contact_weight * (contact[:, -1] + 0.25 * contact.mean(dim=1))
            if self.relocate_target_weight > 0:
                target = dyn[:, :, 1]
                scores = scores + self.relocate_target_weight * (target[:, -1] + 0.25 * target.mean(dim=1))
        if self.relocate_contact_vae is not None and contact_scoring:
            raw_cond = getattr(self, "_current_raw_cond", None)
            if raw_cond is None:
                raw_cond = torch.zeros(1, int(self.relocate_contact_vae_ckpt["state_dim"]), device=self.device)
            raw_cond = raw_cond.repeat(actions.shape[0], 1)
            vae_cond = torch.cat([z.repeat(actions.shape[0], 1), raw_cond, actions.flatten(1)], dim=-1)
            traces = self.relocate_contact_vae.sample(vae_cond, n=self.contact_vae_samples).view(
                actions.shape[0], self.contact_vae_samples, actions.shape[1], 2
            )
            mean_trace = traces.mean(dim=1)
            std_trace = traces.std(dim=1, unbiased=False)
            contact_trace = mean_trace[:, :, 0]
            target_trace = mean_trace[:, :, 1]
            if self.relocate_contact_weight > 0:
                contact = mean_trace[:, :, 0]
                contact_unc = std_trace[:, :, 0]
                scores = scores + self.relocate_contact_weight * (contact[:, -1] + 0.25 * contact.mean(dim=1))
                scores = scores + self.contact_vae_uncertainty_weight * self.relocate_contact_weight * contact_unc.mean(dim=1)
            if self.relocate_target_weight > 0:
                target = mean_trace[:, :, 1]
                target_unc = std_trace[:, :, 1]
                scores = scores + self.relocate_target_weight * (target[:, -1] + 0.25 * target.mean(dim=1))
                scores = scores + self.contact_vae_uncertainty_weight * self.relocate_target_weight * target_unc.mean(dim=1)
        if self.relocate_stage_score and contact_trace is not None and target_trace is not None:
            scores = scores + self._stage_contact_scores(contact_trace, target_trace)
        if self.possession_barrier > 0 and contact_trace is not None and target_trace is not None:
            raw_state = getattr(self, "_current_raw_state", None)
            if raw_state is not None:
                scores = scores + possession_trust_barrier(
                    contact_trace,
                    target_trace,
                    palm_ball=float(torch.linalg.norm(raw_state[30:33]).detach().cpu()),
                    ball_target=float(torch.linalg.norm(raw_state[36:39]).detach().cpu()),
                    contact_threshold=self.possession_contact_threshold,
                    keep_threshold=self.possession_keep_threshold,
                    target_threshold=self.possession_target_threshold,
                    target_slack=self.possession_target_slack,
                    barrier=self.possession_barrier,
                )
        if self.action_l2_weight > 0:
            scores = scores + self.action_l2_weight * actions.square().mean(dim=(1, 2))
        if self.action_delta_weight > 0:
            prev = torch.as_tensor(self.prev_action, dtype=torch.float32, device=self.device).view(1, 1, -1)
            delta = torch.cat([actions[:, :1] - prev, actions[:, 1:] - actions[:, :-1]], dim=1)
            scores = scores + self.action_delta_weight * delta.square().mean(dim=(1, 2))
        if anchor is not None and self.jepa_anchor_weight > 0:
            scores = scores + self.jepa_anchor_weight * (actions - anchor).square().mean(dim=(1, 2))
        if self.jepa_smooth_weight > 0:
            delta = actions[:, 1:] - actions[:, :-1]
            scores = scores + self.jepa_smooth_weight * delta.square().mean(dim=(1, 2))
        return scores

    def _grad_refine_actions(
        self,
        *,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor,
        target_state: np.ndarray,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        if self.jepa_grad_steps <= 0:
            return actions
        center = (high + low).view(1, 1, -1) * 0.5
        scale = (high - low).view(1, 1, -1) * 0.5
        init = torch.clamp((actions - center) / torch.clamp(scale, min=1e-6), -0.999, 0.999)
        u = torch.atanh(init).detach().requires_grad_(True)
        anchor = actions.detach()
        opt = torch.optim.Adam([u], lr=self.jepa_grad_lr)
        for _ in range(self.jepa_grad_steps):
            opt.zero_grad(set_to_none=True)
            a = center + scale * torch.tanh(u)
            loss = self._candidate_scores(z=z, z_goal=z_goal, actions=a, target_state=target_state, anchor=anchor).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 10.0)
            opt.step()
        return (center + scale * torch.tanh(u.detach())).clamp(low, high)

    def _plan(self, obs, env):
        with torch.no_grad():
            raw = flatten_obs(obs)
            target_state = self.future_index.query(raw, self.normalizer)
            self._current_raw_cond = torch.from_numpy(self.normalizer.encode(raw).astype(np.float32)).view(1, -1).to(self.device)
            self._current_raw_state = torch.from_numpy(raw.astype(np.float32)).to(self.device)
            z, z_goal, cond1 = self._condition(raw, target_state)
            objective = str(self.flow_ckpt.get("objective", "flow"))
            anchor_flat = None
            n_warm = 0
            if self.warm_start_frac > 0 and self.prev_plan is not None and objective == "flow":
                shift = min(max(0, self.last_exec_len), self.prev_plan.shape[0] - 1)
                anchor = torch.cat(
                    [self.prev_plan[shift:], self.prev_plan[-1:].expand(shift, -1)], dim=0
                )
                anchor_flat = anchor.reshape(1, -1)
                n_warm = int(round(self.candidates * self.warm_start_frac))
            n_fresh = max(0, self.candidates - n_warm)
            parts = []
            if n_fresh > 0:
                parts.append(
                    sample_action_chunks(
                        self.flow,
                        self.flow_ddpm,
                        cond1.repeat(n_fresh, 1),
                        int(self.flow_ckpt["chunk_dim"]),
                        self.device,
                        objective=objective,
                        flow_steps=self.flow_steps,
                        init_noise_scale=self.flow_init_noise_scale,
                    )
                )
            if n_warm > 0:
                parts.append(
                    sample_action_chunks(
                        self.flow,
                        self.flow_ddpm,
                        cond1.repeat(n_warm, 1),
                        int(self.flow_ckpt["chunk_dim"]),
                        self.device,
                        objective=objective,
                        flow_steps=self.flow_steps,
                        init_noise_scale=self.flow_init_noise_scale,
                        warm_init=anchor_flat.repeat(n_warm, 1),
                        warm_tau=self.warm_start_tau,
                    )
                )
            if anchor_flat is not None:
                parts.append(anchor_flat)
            proposals = torch.cat(parts, dim=0)
            cond = cond1.repeat(proposals.shape[0], 1)
            if self.residual_scale == 0.0:
                residual = torch.zeros_like(proposals)
            else:
                ref_cond = torch.cat([cond, proposals], dim=-1)
                residual = sample_action_chunks(
                    self.refiner,
                    self.refiner_ddpm,
                    ref_cond,
                    int(self.refiner_ckpt["chunk_dim"]),
                    self.device,
                    objective=str(self.refiner_ckpt.get("objective", "diffusion")),
                    flow_steps=self.refine_steps,
                    init_noise_scale=self.refine_init_noise_scale,
                )
        H = int(self.flow_ckpt["H"])
        A = int(self.flow_ckpt["action_dim"])
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        chunks = proposals + self.residual_scale * residual
        if self.include_unrefined:
            chunks = torch.cat([proposals, chunks], dim=0)
        actions = (chunks.view(chunks.shape[0], H, A) * self.action_scale).clamp(low, high)
        actions = self._grad_refine_actions(z=z, z_goal=z_goal, actions=actions, target_state=target_state, low=low, high=high)
        if (
            actions.shape[0] == 1
            and self.state_weight <= 0
            and self.state_dim_weight <= 0
            and self.action_l2_weight <= 0
            and self.action_delta_weight <= 0
            and self.jepa_grad_steps <= 0
        ):
            self.prev_plan = (actions[0] / max(self.action_scale, 1e-6)).detach()
            return actions[0].detach().cpu().numpy().astype(np.float32)
        with torch.no_grad():
            scores = self._candidate_scores(z=z, z_goal=z_goal, actions=actions, target_state=target_state)
        best = int(torch.argmin(scores).detach().cpu())
        self.prev_plan = (actions[best] / max(self.action_scale, 1e-6)).detach()
        return actions[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs, env):
        if not self.cached:
            plan = self._plan(obs, env)
            k = self.exec_k
            if self.grasp_commit_k > 0:
                raw = flatten_obs(obs)
                palm_ball = float(np.linalg.norm(raw[30:33]))
                ball_target = float(np.linalg.norm(raw[36:39]))
                if (
                    self.grasp_commit_low <= palm_ball <= self.grasp_commit_high
                    and ball_target > self.possession_target_threshold
                ):
                    # Commit through the contact mode switch: execute the grasp
                    # segment without replanning so closure is not dithered away
                    # by fresh proposal noise.
                    k = self.grasp_commit_k
            k = max(1, min(k, len(plan)))
            self.last_exec_len = k
            self.cached = [plan[i].copy() for i in range(k)]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
        self.prev_action = action
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--flow-path", type=Path, required=True)
    p.add_argument("--refiner-path", type=Path, required=True)
    p.add_argument("--relocate-probe-path", type=Path, default=None,
                   help="Optional self-supervised Relocate probe predicting palm-ball and ball-target distances from latent states.")
    p.add_argument("--relocate-contact-dynamics-path", type=Path, default=None,
                   help="Optional action-conditioned contact dynamics head predicting palm-ball and ball-target traces from z/raw/action chunk.")
    p.add_argument("--relocate-contact-vae-path", type=Path, default=None,
                   help="Optional conditional VAE over contact traces from z/raw/action chunk.")
    p.add_argument("--contact-vae-samples", type=int, default=8)
    p.add_argument("--contact-vae-uncertainty-weight", type=float, default=0.0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--refine-steps", type=int, default=16)
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.0)
    p.add_argument("--state-dims", type=parse_dims, default=[])
    p.add_argument("--state-dim-weight", type=float, default=0.0)
    p.add_argument("--action-l2-weight", type=float, default=0.0)
    p.add_argument("--action-delta-weight", type=float, default=0.0)
    p.add_argument("--relocate-contact-weight", type=float, default=0.0,
                   help="Adroit Relocate decoded-state score: palm-to-ball distance from obs dims 30:33.")
    p.add_argument("--relocate-target-weight", type=float, default=0.0,
                   help="Adroit Relocate decoded-state score: ball-to-target distance from obs dims 36:39.")
    p.add_argument("--relocate-stage-score", action="store_true",
                   help="Use staged Relocate contact-trust scoring: reach contact, transport while maintaining contact, then place.")
    p.add_argument("--possession-barrier", type=float, default=0.0,
                   help="Mode-aware trust-region barrier scale on predicted contact traces: bar predicted possession loss during transport instead of weakly penalizing it.")
    p.add_argument("--possession-contact-threshold", type=float, default=0.06)
    p.add_argument("--possession-keep-threshold", type=float, default=0.09)
    p.add_argument("--possession-target-threshold", type=float, default=0.06)
    p.add_argument("--possession-target-slack", type=float, default=0.10)
    p.add_argument("--torch-seed", type=int, default=None,
                   help="Seed torch RNG so stochastic flow sampling is reproducible across eval runs.")
    p.add_argument("--stage-contact-threshold", type=float, default=0.06)
    p.add_argument("--stage-target-threshold", type=float, default=0.06)
    p.add_argument("--stage-reach-weight", type=float, default=0.2)
    p.add_argument("--stage-transport-weight", type=float, default=0.2)
    p.add_argument("--stage-place-weight", type=float, default=0.2)
    p.add_argument("--stage-trust-weight", type=float, default=0.1)
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--warm-start-frac", type=float, default=0.0,
                   help="Fraction of candidates warm-started from the shifted previous plan (rectified-flow re-noising) for receding-horizon temporal coherence.")
    p.add_argument("--warm-start-tau", type=float, default=0.5,
                   help="Flow time in (0,1) to start warm-started integration from; higher stays closer to the previous plan.")
    p.add_argument("--grasp-commit-k", type=int, default=0,
                   help="When palm-ball distance is inside the commit band, execute this many steps of the chosen chunk without replanning (contact mode-switch commitment).")
    p.add_argument("--grasp-commit-low", type=float, default=0.0)
    p.add_argument("--grasp-commit-high", type=float, default=0.12)
    p.add_argument("--flow-init-noise-scale", type=float, default=1.0)
    p.add_argument("--refine-init-noise-scale", type=float, default=1.0)
    p.add_argument("--include-unrefined", action="store_true",
                   help="Score original flow proposals together with refined proposals, so refinement cannot overwrite a good chunk blindly.")
    p.add_argument("--jepa-grad-steps", type=int, default=0,
                   help="Differentiate through the action-conditioned JEPA rollout to refine sampled chunks before scoring.")
    p.add_argument("--jepa-grad-lr", type=float, default=0.05)
    p.add_argument("--jepa-anchor-weight", type=float, default=1.0,
                   help="Penalty keeping JEPA-gradient-refined actions near the sampled flow/refined chunk.")
    p.add_argument("--jepa-smooth-weight", type=float, default=0.0)
    p.add_argument("--future-index", choices=["nearest", "relocate_predicate", "demo_locked"], default="nearest")
    p.add_argument("--future-episodes-npz", type=Path, default=None,
                   help="Episode npz supplying full demo trajectories for the demo_locked future index.")
    p.add_argument("--future-locality-weight", type=float, default=0.0,
                   help="demo_locked: soft penalty on re-localizing far from the previous progress index.")
    p.add_argument("--future-k", type=int, default=1024)
    p.add_argument("--future-local-weight", type=float, default=0.25)
    p.add_argument("--relocate-contact-threshold", type=float, default=0.06)
    p.add_argument("--future-contact-weight", type=float, default=1.0)
    p.add_argument("--future-target-weight", type=float, default=1.0)
    p.add_argument("--future-maintain-weight", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--collect-npz", type=Path, default=None,
                   help="Also save evaluated rollouts (states/actions per episode) as an npz episode file, e.g. for contact-dynamics recalibration on the planner's own action distribution.")
    p.add_argument("--collect-keep", choices=["all", "success"], default="all")
    p.add_argument("--log-episodes", action="store_true",
                   help="Print a per-episode diagnostic row: grasp achieved, drops after possession, final ball-target distance.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)
    flow_ckpt = torch.load(args.flow_path, map_location=dev, weights_only=False)
    refiner_ckpt = torch.load(args.refiner_path, map_location=dev, weights_only=False)
    flow = EpsNet(
        int(flow_ckpt["chunk_dim"]),
        int(flow_ckpt["cond_dim"]),
        int(flow_ckpt["hidden"]),
        n_blocks=int(flow_ckpt["n_blocks"]),
    ).to(dev)
    flow.load_state_dict(flow_ckpt["ema"])
    flow.eval()
    for param in flow.parameters():
        param.requires_grad_(False)
    refiner = EpsNet(
        int(refiner_ckpt["chunk_dim"]),
        int(refiner_ckpt["cond_dim"]),
        int(refiner_ckpt["hidden"]),
        n_blocks=int(refiner_ckpt["n_blocks"]),
    ).to(dev)
    refiner.load_state_dict(refiner_ckpt["ema"])
    refiner.eval()
    for param in refiner.parameters():
        param.requires_grad_(False)
    relocate_probe = None
    if args.relocate_probe_path is not None:
        probe_ckpt = torch.load(args.relocate_probe_path, map_location=dev, weights_only=False)
        relocate_probe = MLP([int(probe_ckpt["latent_dim"]), int(probe_ckpt["hidden"]), int(probe_ckpt["hidden"]), 2], layer_norm=True).to(dev)
        relocate_probe.load_state_dict(probe_ckpt["state_dict"])
        relocate_probe.eval()
        for param in relocate_probe.parameters():
            param.requires_grad_(False)
    relocate_contact_dynamics = None
    relocate_contact_dynamics_ckpt = None
    if args.relocate_contact_dynamics_path is not None:
        relocate_contact_dynamics_ckpt = torch.load(args.relocate_contact_dynamics_path, map_location=dev, weights_only=False)
        sizes = [int(relocate_contact_dynamics_ckpt["cond_dim"])] + [int(relocate_contact_dynamics_ckpt["hidden"])] * max(1, int(relocate_contact_dynamics_ckpt["n_blocks"])) + [int(relocate_contact_dynamics_ckpt["target_dim"])]
        relocate_contact_dynamics = MLP(sizes, layer_norm=True).to(dev)
        relocate_contact_dynamics.load_state_dict(relocate_contact_dynamics_ckpt["state_dict"])
        relocate_contact_dynamics.eval()
        for param in relocate_contact_dynamics.parameters():
            param.requires_grad_(False)
    relocate_contact_vae = None
    relocate_contact_vae_ckpt = None
    if args.relocate_contact_vae_path is not None:
        relocate_contact_vae_ckpt = torch.load(args.relocate_contact_vae_path, map_location=dev, weights_only=False)
        relocate_contact_vae = ContactTraceCVAE(
            int(relocate_contact_vae_ckpt["cond_dim"]),
            int(relocate_contact_vae_ckpt["trace_dim"]),
            int(relocate_contact_vae_ckpt["vae_latent_dim"]),
            int(relocate_contact_vae_ckpt["hidden"]),
            int(relocate_contact_vae_ckpt["n_blocks"]),
        ).to(dev)
        relocate_contact_vae.load_state_dict(relocate_contact_vae_ckpt["state_dict"])
        relocate_contact_vae.eval()
        for param in relocate_contact_vae.parameters():
            param.requires_grad_(False)
    if int(refiner_ckpt["base_cond_dim"]) != int(flow_ckpt["cond_dim"]):
        raise ValueError("refiner was trained for a different flow condition dimension")
    bank_states = np.asarray(flow_ckpt["bank_states"])
    bank_futures = np.asarray(flow_ckpt["bank_futures"])
    if args.future_index == "demo_locked":
        if args.future_episodes_npz is None:
            raise ValueError("--future-index demo_locked requires --future-episodes-npz")
        horizons = list(flow_ckpt.get("future_horizons", [int(flow_ckpt["H"])]))
        future_index = DemoLockedFutureIndex(
            [ep.states for ep in load_episodes_npz(args.future_episodes_npz)],
            norm,
            horizon=int(args.target_horizon or max(horizons)),
            locality_weight=args.future_locality_weight,
        )
    elif args.future_index == "relocate_predicate":
        future_index = RelocatePredicateFutureIndex(
            bank_states,
            bank_futures,
            norm,
            k=args.future_k,
            local_weight=args.future_local_weight,
            contact_threshold=args.relocate_contact_threshold,
            contact_weight=args.future_contact_weight,
            target_weight=args.future_target_weight,
            maintain_weight=args.future_maintain_weight,
        )
    else:
        future_index = NearestFutureIndex(bank_states, bank_futures, norm)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    policy = FlowResidualPolicy(
        wm=wm,
        normalizer=norm,
        spec=spec,
        flow=flow,
        flow_ckpt=flow_ckpt,
        refiner=refiner,
        refiner_ckpt=refiner_ckpt,
        relocate_probe=relocate_probe,
        relocate_contact_dynamics=relocate_contact_dynamics,
        relocate_contact_dynamics_ckpt=relocate_contact_dynamics_ckpt,
        relocate_contact_vae=relocate_contact_vae,
        relocate_contact_vae_ckpt=relocate_contact_vae_ckpt,
        contact_vae_samples=args.contact_vae_samples,
        contact_vae_uncertainty_weight=args.contact_vae_uncertainty_weight,
        future_index=future_index,
        device=dev,
        candidates=args.candidates,
        exec_k=args.exec_k,
        flow_steps=args.flow_steps,
        refine_steps=args.refine_steps,
        target_horizon=args.target_horizon,
        latent_weight=args.latent_weight,
        state_weight=args.state_weight,
        state_dim_weight=args.state_dim_weight,
        state_dims=args.state_dims,
        action_l2_weight=args.action_l2_weight,
        action_delta_weight=args.action_delta_weight,
        relocate_contact_weight=args.relocate_contact_weight,
        relocate_target_weight=args.relocate_target_weight,
        relocate_stage_score=args.relocate_stage_score,
        possession_barrier=args.possession_barrier,
        possession_contact_threshold=args.possession_contact_threshold,
        possession_keep_threshold=args.possession_keep_threshold,
        possession_target_threshold=args.possession_target_threshold,
        possession_target_slack=args.possession_target_slack,
        stage_contact_threshold=args.stage_contact_threshold,
        stage_target_threshold=args.stage_target_threshold,
        stage_reach_weight=args.stage_reach_weight,
        stage_transport_weight=args.stage_transport_weight,
        stage_place_weight=args.stage_place_weight,
        stage_trust_weight=args.stage_trust_weight,
        residual_scale=args.residual_scale,
        action_scale=args.action_scale,
        warm_start_frac=args.warm_start_frac,
        warm_start_tau=args.warm_start_tau,
        grasp_commit_k=args.grasp_commit_k,
        grasp_commit_low=args.grasp_commit_low,
        grasp_commit_high=args.grasp_commit_high,
        flow_init_noise_scale=args.flow_init_noise_scale,
        refine_init_noise_scale=args.refine_init_noise_scale,
        include_unrefined=args.include_unrefined,
        jepa_grad_steps=args.jepa_grad_steps,
        jepa_grad_lr=args.jepa_grad_lr,
        jepa_anchor_weight=args.jepa_anchor_weight,
        jepa_smooth_weight=args.jepa_smooth_weight,
    )
    successes = []
    collect_states: list[np.ndarray] = []
    collect_actions: list[np.ndarray] = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()
        ep_states = [flatten_obs(obs).copy()]
        ep_actions = []
        term = trunc = False
        info = {}
        while not (term or trunc):
            action = policy.act(obs, env)
            obs, _, term, trunc, info = env.step(action)
            ep_actions.append(np.asarray(action, dtype=np.float32).copy())
            ep_states.append(flatten_obs(obs).copy())
        success = float(info.get("is_success", info.get("success", 0.0)))
        successes.append(success)
        if args.log_episodes:
            traj = np.asarray(ep_states, dtype=np.float32)
            palm_ball = np.linalg.norm(traj[:, 30:33], axis=-1)
            ball_target = np.linalg.norm(traj[:, 36:39], axis=-1)
            possessed = palm_ball < 0.06
            drops = int(np.sum(possessed[:-1] & ~possessed[1:]))
            print(
                json.dumps(
                    {
                        "event": "episode_diag",
                        "episode": ep,
                        "success": success,
                        "grasped": bool(possessed.any()),
                        "possession_steps": int(possessed.sum()),
                        "drops_after_possession": drops,
                        "min_palm_ball": float(palm_ball.min()),
                        "final_ball_target": float(ball_target[-1]),
                        "min_ball_target": float(ball_target.min()),
                    }
                ),
                flush=True,
            )
        if args.collect_npz is not None and (args.collect_keep == "all" or success > 0):
            collect_states.append(np.asarray(ep_states, dtype=np.float32))
            collect_actions.append(np.asarray(ep_actions, dtype=np.float32))
    env.close()
    if args.collect_npz is not None:
        args.collect_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.collect_npz,
            states=np.array(collect_states, dtype=object),
            actions=np.array(collect_actions, dtype=object),
            rewards=np.array([np.ones(len(a), dtype=np.float32) for a in collect_actions], dtype=object),
        )
    row = {
        "event": "flow_residual_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "flow_path": str(args.flow_path),
        "refiner_path": str(args.refiner_path),
        "model_config": cfg,
        "policy": policy.name,
        "episodes": float(args.episodes),
        "success_rate": float(np.mean(successes)),
        "candidates": int(args.candidates),
        "exec_k": int(args.exec_k),
        "residual_scale": float(args.residual_scale),
        "include_unrefined": bool(args.include_unrefined),
        "relocate_contact_weight": float(args.relocate_contact_weight),
        "relocate_target_weight": float(args.relocate_target_weight),
        "relocate_stage_score": bool(args.relocate_stage_score),
        "possession_barrier": float(args.possession_barrier),
        "possession_contact_threshold": float(args.possession_contact_threshold),
        "possession_keep_threshold": float(args.possession_keep_threshold),
        "possession_target_threshold": float(args.possession_target_threshold),
        "possession_target_slack": float(args.possession_target_slack),
        "latent_weight": float(args.latent_weight),
        "state_weight": float(args.state_weight),
        "flow_steps": int(args.flow_steps),
        "action_delta_weight": float(args.action_delta_weight),
        "action_scale": float(args.action_scale),
        "flow_init_noise_scale": float(args.flow_init_noise_scale),
        "torch_seed": args.torch_seed,
        "warm_start_frac": float(args.warm_start_frac),
        "warm_start_tau": float(args.warm_start_tau),
        "grasp_commit_k": int(args.grasp_commit_k),
        "grasp_commit_low": float(args.grasp_commit_low),
        "grasp_commit_high": float(args.grasp_commit_high),
        "future_episodes_npz": str(args.future_episodes_npz) if args.future_episodes_npz is not None else None,
        "future_locality_weight": float(args.future_locality_weight),
        "stage_contact_threshold": float(args.stage_contact_threshold),
        "stage_target_threshold": float(args.stage_target_threshold),
        "stage_reach_weight": float(args.stage_reach_weight),
        "stage_transport_weight": float(args.stage_transport_weight),
        "stage_place_weight": float(args.stage_place_weight),
        "stage_trust_weight": float(args.stage_trust_weight),
        "relocate_probe_path": str(args.relocate_probe_path) if args.relocate_probe_path is not None else None,
        "relocate_contact_dynamics_path": str(args.relocate_contact_dynamics_path) if args.relocate_contact_dynamics_path is not None else None,
        "relocate_contact_vae_path": str(args.relocate_contact_vae_path) if args.relocate_contact_vae_path is not None else None,
        "contact_vae_samples": int(args.contact_vae_samples),
        "contact_vae_uncertainty_weight": float(args.contact_vae_uncertainty_weight),
        "jepa_grad_steps": int(args.jepa_grad_steps),
        "jepa_anchor_weight": float(args.jepa_anchor_weight),
        "future_index": args.future_index,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
