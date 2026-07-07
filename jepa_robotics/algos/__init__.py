"""Shared algorithm utilities for SSL latent-control policies.

Training and evaluation scripts should stay as thin CLI wrappers. Reusable
models, indexing logic, and planning helpers live here so Fetch, Adroit, Maze,
and Kitchen experiments share the same JEPA/SSL control primitives.
"""

from .hwm import LatentMacroPredictor, MacroActionEncoder, sample_macro_dataset
from .phase import PhaseFutureIndex, batch_phase_features, phase_features, phase_id
from .priors import EpsNet, InversePrior, make_ddpm, sinusoidal_embedding

__all__ = [
    "EpsNet",
    "InversePrior",
    "LatentMacroPredictor",
    "MacroActionEncoder",
    "PhaseFutureIndex",
    "batch_phase_features",
    "make_ddpm",
    "phase_features",
    "phase_id",
    "sample_macro_dataset",
    "sinusoidal_embedding",
]
