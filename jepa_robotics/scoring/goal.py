from __future__ import annotations

import torch
import torch.nn.functional as F

from ..envs import goal_state_from_state


class GoalScoringMixin:
    """Generic goal-reaching cost: latent / state / combined distance to goal.

    Used by reach and navigation (PointMaze) tasks where the objective is
    simply driving achieved_goal -> desired_goal.
    """

    def _goal_scores(self, raw_state, z, action_tensor):

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
