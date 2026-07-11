from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .envs import flatten_obs
from .evaluate import load_jepa_artifact


class JEPALatentObsWrapper(gym.ObservationWrapper):
    """Replace an env's observation with the frozen JEPA latent.

    For non-goal envs (Adroit) the HER ``JEPALatentExtractor`` path does not
    apply, so instead we wrap the *environment* so its observation IS the JEPA
    encoding of the (flattened, normalized) state. A standard SAC/TQC ``MlpPolicy``
    then learns a controller entirely in the world-model latent space — the
    deep-learning control half of the JEPA agent, on dense reward, no HER.
    """

    def __init__(self, env, model_path, device: str = "cpu") -> None:
        super().__init__(env)
        self._device = torch.device(device)
        model, normalizer, spec, config = load_jepa_artifact(
            resolve_jepa_model_path(model_path), self._device
        )
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._mean = torch.as_tensor(normalizer.mean, dtype=torch.float32, device=self._device)
        self._std = torch.as_tensor(normalizer.std, dtype=torch.float32, device=self._device)
        latent_dim = int(config["latent_dim"])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32
        )

    def observation(self, obs):
        flat = torch.from_numpy(flatten_obs(obs)).to(self._device)
        with torch.no_grad():
            z = self.model.encode(((flat - self._mean) / self._std).unsqueeze(0))[0]
        return z.cpu().numpy().astype(np.float32)


class JEPAEncoderExtractor(BaseFeaturesExtractor):
    """JEPA encoder as a *trainable* SB3 feature extractor (warm-start + fine-tune).

    The frozen-latent controller fails on contact-rich Adroit tasks because the
    encoder, trained on random exploratory data, never saw success states (door
    open, nail driven) and so represents them poorly. Here the encoder is
    initialized from the pretrained world model but left **trainable**, so the
    actor/critic gradients reshape it toward reward-relevant features while
    keeping the JEPA dynamics pretraining as a strong initialization. For flat
    (non-goal) Box observations.
    """

    def __init__(self, observation_space: spaces.Box, model_path, device: str = "cpu",
                 freeze: bool = False) -> None:
        model, normalizer, _spec, config = load_jepa_artifact(
            resolve_jepa_model_path(model_path), torch.device("cpu")
        )
        latent_dim = int(config["latent_dim"])
        super().__init__(observation_space, features_dim=latent_dim)
        self.encoder = model.encoder  # warm-started; trainable unless frozen
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.register_buffer("mean", torch.as_tensor(normalizer.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(normalizer.std, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = (observations.float() - self.mean.to(observations.device)) / self.std.to(observations.device)
        return self.encoder(x)


class JEPAConcatExtractor(BaseFeaturesExtractor):
    """Feature = concat[normalized raw obs, trainable JEPA latent].

    Iteration after pure-latent control failed on Adroit: the random-data encoder
    bottlenecks the policy (it never encoded success states), so a raw-obs
    reference solves Door while the latent controller stays at 0%. Concatenating
    the raw observation restores full observability (the policy *can* solve, like
    the reference) while keeping the warm-started, trainable JEPA features so the
    world-model representation can still contribute. For flat (non-goal) obs.
    """

    def __init__(self, observation_space: spaces.Box, model_path, device: str = "cpu",
                 freeze: bool = False) -> None:
        model, normalizer, _spec, config = load_jepa_artifact(
            resolve_jepa_model_path(model_path), torch.device("cpu")
        )
        latent_dim = int(config["latent_dim"])
        obs_dim = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=obs_dim + latent_dim)
        self.encoder = model.encoder
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.register_buffer("mean", torch.as_tensor(normalizer.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(normalizer.std, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = (observations.float() - self.mean.to(observations.device)) / self.std.to(observations.device)
        return torch.cat([x, self.encoder(x)], dim=-1)


def resolve_jepa_model_path(model_path: str | Path) -> Path:
    """Resolve stale embedded JEPA paths in saved SB3 feature extractors."""
    path = Path(model_path)
    if path.exists():
        return path

    env_fallback = os.environ.get("JEPA_MODEL_FALLBACK")
    candidates: list[Path] = []
    if env_fallback:
        candidates.append(Path(env_fallback))
    if path.name == "slide_vicreg_resume_20260611_130154_model.pt":
        candidates.append(path.with_name("slide_vicreg_resume_20260613_model.pt"))

    for candidate in candidates:
        if candidate.exists():
            print(
                f'{{"event": "jepa_model_path_fallback", "missing": "{path}", '
                f'"using": "{candidate}"}}',
                flush=True,
            )
            return candidate
    return path


class JEPALatentExtractor(BaseFeaturesExtractor):
    """Frozen JEPA encoder as an SB3 feature extractor for goal-conditioned RL.

    SB3/HER keeps the original Dict observation space. This extractor flattens
    ``observation``, ``achieved_goal`` and ``desired_goal`` the same way the JEPA
    code does, applies the saved normalizer, and feeds the result through the
    frozen JEPA encoder. The RL actor/critic are still trained with real
    environment rewards, so this removes the BC/scripted-reference ceiling while
    reusing the JEPA representation.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        model_path: str | Path,
        device: str = "auto",
        layer_norm: bool = False,
    ) -> None:
        load_device = torch.device("cpu")
        model, normalizer, _spec, config = load_jepa_artifact(resolve_jepa_model_path(model_path), load_device)
        features_dim = int(config["latent_dim"])
        super().__init__(observation_space, features_dim=features_dim)

        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.register_buffer("mean", torch.as_tensor(normalizer.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(normalizer.std, dtype=torch.float32))
        self.requested_device = device
        self.layer_norm = layer_norm

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            value = observations[key]
            parts.append(value.float().reshape(value.shape[0], -1))
        state = torch.cat(parts, dim=-1)
        state = (state - self.mean.to(state.device)) / self.std.to(state.device)
        with torch.no_grad():
            z = self.model.encode(state)
            if self.layer_norm:
                z = F.layer_norm(z, z.shape[-1:])
            return z

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["requested_device"] = "auto"
        return state
