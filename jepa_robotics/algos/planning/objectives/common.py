from __future__ import annotations

import torch


class CommonScoringMixin:
    """Action-magnitude / smoothness regularizers shared by every score mode."""

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
