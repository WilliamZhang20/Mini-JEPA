from __future__ import annotations

import torch


class ManipScoringMixin:
    """Fetch pick/push manipulation cost (reach->align->grasp->transport)."""

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
