from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalized prediction and target latents (cosine-style JEPA loss)."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(pred, target)


def variance_regularizer(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hinge penalty pushing each latent dimension's batch std toward 1, to prevent representation collapse."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_regularizer(z: torch.Tensor) -> torch.Tensor:
    """Penalize redundant latent dimensions by shrinking off-diagonal covariance.

    Together with the variance hinge, this is the VICReg/Barlow-style version of
    bounding low-energy latent volume: dimensions must stay active, but cannot
    all encode the same direction.
    """
    if z.shape[0] < 2:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / float(z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / z.shape[1]
