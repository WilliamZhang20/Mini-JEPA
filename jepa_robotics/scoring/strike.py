from __future__ import annotations

import torch


class StrikeScoringMixin:
    """FetchSlide ballistic-strike cost ('commit then watch')."""

    def _strike_scores(
        self,
        raw_state: np.ndarray,
        z: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Ballistic-strike cost for FetchSlide-style 'commit then watch' tasks.

        Unlike ``manip`` (built for picking, which keeps the gripper *on* the
        object), a slide strike requires the gripper to contact the puck once and
        then *leave* it while the puck coasts. So we:

        * reward reaching the puck *early* (min gripper->puck over the opening
          window) to initiate a strike, and
        * reward the puck getting close to the goal at the end *and* at its best
          point in the post-strike window (the world-model horizon is capped and
          the goal is out of reach, so the closest approach can occur before the
          final step),

        and crucially we add **no penalty for the gripper separating from the
        puck** after impact. Weights reuse the manip CLI knobs: terminal=1.0,
        late-progress=``manip_path_weight``, early-contact=``manip_reach_weight``.
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
        obj_to_goal = torch.linalg.norm(obj - desired, dim=-1)                  # [B, H]
        grip_to_obj = torch.linalg.norm(grip[..., :gd] - obj[..., :gd], dim=-1)  # [B, H]

        H = obj_to_goal.shape[1]
        early = max(1, H // 3)
        contact = grip_to_obj[:, :early].min(dim=1).values    # reach puck early to strike
        terminal = obj_to_goal[:, -1]                          # puck near goal at horizon end
        late_best = obj_to_goal[:, early:].min(dim=1).values   # closest approach post-strike

        scores = (
            terminal
            + self.manip_path_weight * late_best
            + self.manip_reach_weight * contact
        )
        scores = scores + self._action_regularizers(action_tensor)
        return scores
