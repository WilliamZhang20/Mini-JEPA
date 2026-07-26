"""Fit a context-aware JEPA predictor on reward-free dexterous play.

The representation comes from an existing DexterousJEPA checkpoint and is
frozen.  Only the action-conditioned dynamics is trained, solely by predicting
the target-encoder latent of a future observation:

    z[t-W+1:t] = frozen_encoder(states)
    z_hat[t+2] = predictor(z_history, past_actions, actions[t:t+2])
    loss        = latent_distance(z_hat[t+2], frozen_target(state[t+2]))

The default W=3 exposes velocity and acceleration.  A two-step rollout is used
with the first prediction detached, matching the last-gradient-only/TBPTT
recipe that recent JEPA-WM ablations find strongest for simulated control.  No
reward, success label, value, policy, inverse dynamics, or demonstration flag
enters the objective.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.models import ContextualDexterousJEPA, normalized_mse


def _copy_representation(target: ContextualDexterousJEPA, source) -> None:
    """Transfer the learned SSL representation and diagnostic probes."""
    target.encoder.load_state_dict(source.encoder.state_dict())
    target.target_encoder.load_state_dict(source.target_encoder.state_dict())
    target.state_probe.load_state_dict(source.state_probe.state_dict())
    target.distance_probe.load_state_dict(source.distance_probe.state_dict())
    if hasattr(source, "object_probe") and hasattr(target, "object_probe"):
        target.object_probe.load_state_dict(source.object_probe.state_dict())
    if hasattr(source, "contact_probe") and hasattr(target, "contact_probe"):
        target.contact_probe.load_state_dict(source.contact_probe.state_dict())


def _copy_contextual_dynamics(
    target: ContextualDexterousJEPA, source
) -> None:
    """Warm-start a second optimization stage from a contextual checkpoint."""
    if not hasattr(source, "context_len") or source.context_len != target.context_len:
        raise ValueError("Warm-start model has an incompatible context length")
    target.dyn_heads.load_state_dict(source.dyn_heads.state_dict())
    if hasattr(target, "action_decoder"):
        if not hasattr(source, "action_decoder"):
            raise ValueError("Warm-start model has no latent action decoder")
        target.action_decoder.load_state_dict(source.action_decoder.state_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--episodes-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=5e-5,
        help="Small representation learning rate when latent-difference action decoding is enabled.",
    )
    parser.add_argument("--ema", type=float, default=0.996)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--object-slot-weight", type=float, default=2.0)
    parser.add_argument(
        "--latent-loss",
        choices=["raw", "normalized"],
        default="raw",
        help="Raw L2 preserves rollout scale (recommended); normalized is retained only for ablation compatibility.",
    )
    parser.add_argument(
        "--ensemble-heads",
        type=int,
        default=1,
        help="Contextual dynamics heads. One is the efficient default; use >1 only when planning uses uncertainty.",
    )
    parser.add_argument(
        "--delta-action-weight",
        type=float,
        default=0.0,
        help="Enable Delta-JEPA latent-displacement action reconstruction with this loss weight.",
    )
    parser.add_argument("--action-decoder-depth", type=int, default=3)
    parser.add_argument(
        "--warm-start-dynamics",
        action="store_true",
        help="Also copy contextual dynamics/action-decoder weights from --base-model.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Keep a previously action-shaped encoder fixed during this stage.",
    )
    parser.add_argument(
        "--explicit-object-slot",
        action="store_true",
        help="Copy the normalized observed object pose into the leading object-slot coordinates.",
    )
    parser.add_argument(
        "--explicit-pose-weight",
        type=float,
        default=5.0,
        help="Extra latent L2 weight on the explicit normalized object-pose coordinates.",
    )
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1451)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.context_len < 2:
        raise ValueError("--context-len must be at least 2")
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be positive")
    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, normalizer, spec, config = load_jepa_artifact(
        args.base_model, device
    )
    token_groups = config.get("token_groups")
    contact_dims = config.get("contact_dims")
    pose_relation_dims = config.get("pose_relation_dims")
    object_dims = config.get("object_dims")
    model = ContextualDexterousJEPA(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        latent_dim=int(config["latent_dim"]),
        d_model=int(config.get("d_model", 256)),
        enc_depth=int(config.get("enc_depth", 4)),
        dyn_depth=int(config.get("dyn_depth", 4)),
        heads=int(config.get("heads", 8)),
        max_horizon=max(
            int(config.get("max_horizon", args.rollout_steps)),
            args.rollout_steps,
        ),
        ensemble_heads=args.ensemble_heads,
        contact_dims=tuple(contact_dims) if contact_dims else None,
        token_groups=(
            tuple(tuple(int(v) for v in group) for group in token_groups)
            if token_groups
            else None
        ),
        latent_slots=int(config.get("latent_slots", 1)),
        pose_relation_dims=(
            tuple(int(v) for v in pose_relation_dims)
            if pose_relation_dims
            else None
        ),
        state_mean=torch.as_tensor(normalizer.mean, dtype=torch.float32),
        state_std=torch.as_tensor(normalizer.std, dtype=torch.float32),
        object_probe_dims=(
            tuple(int(v) for v in object_dims)
            if config.get("object_probe", False) and object_dims
            else None
        ),
        context_len=args.context_len,
        latent_difference_actions=args.delta_action_weight > 0,
        action_decoder_depth=args.action_decoder_depth,
        explicit_object_slot=args.explicit_object_slot,
    ).to(device)
    _copy_representation(model, base)
    if args.warm_start_dynamics:
        _copy_contextual_dynamics(model, base)
    del base

    # Probes and the EMA target stay fixed with respect to backpropagation.
    # Plain contextual training also freezes the online encoder. Delta-JEPA
    # instead fine-tunes it slowly so latent displacements become action-aware.
    frozen_modules = [model.target_encoder, model.state_probe, model.distance_probe]
    if args.delta_action_weight <= 0 or args.freeze_encoder:
        frozen_modules.append(model.encoder)
    if hasattr(model, "object_probe"):
        frozen_modules.append(model.object_probe)
    if hasattr(model, "contact_probe"):
        frozen_modules.append(model.contact_probe)
    for module in frozen_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    episodes = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    first_t: list[int] = []
    last_t: list[int] = []
    for episode in episodes:
        if len(episode.actions) < args.context_len + args.rollout_steps:
            continue
        states.append(normalizer.encode(episode.states.astype(np.float32)))
        actions.append(episode.actions.astype(np.float32))
        # t is the current state index.  W-1 preceding actions connect the
        # context states; H following actions lead to target state t+H.
        first_t.append(args.context_len - 1)
        last_t.append(len(episode.actions) - args.rollout_steps)
    if not states:
        raise ValueError("No episode is long enough for the requested context/rollout")

    encoder_parameters = [
        p for p in model.encoder.parameters() if p.requires_grad
    ]
    encoder_ids = {id(p) for p in encoder_parameters}
    dynamics_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in encoder_ids
    ]
    trainable = encoder_parameters + dynamics_parameters
    parameter_groups = [{"params": dynamics_parameters, "lr": args.lr}]
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": args.encoder_lr}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay
    )
    latent_slots = int(config.get("latent_slots", 1))
    slot_dim = int(config["latent_dim"]) // latent_slots
    print(
        json.dumps(
            {
                "event": "dexterous_contextual_data",
                "episodes": len(states),
                "context_len": args.context_len,
                "rollout_steps": args.rollout_steps,
                "latent_slots": latent_slots,
                "trainable_parameters_M": round(
                    sum(p.numel() for p in trainable) / 1e6, 3
                ),
                "frozen_representation": str(args.base_model),
                "delta_action_weight": args.delta_action_weight,
                "encoder_trainable": bool(encoder_parameters),
                "warm_start_dynamics": args.warm_start_dynamics,
            }
        ),
        flush=True,
    )

    def sample_batch():
        state_history, action_history, future_actions, future_states = [], [], [], []
        for _ in range(args.batch_size):
            episode_id = int(rng.integers(len(states)))
            t = int(
                rng.integers(
                    first_t[episode_id], last_t[episode_id] + 1
                )
            )
            state_history.append(
                states[episode_id][t - args.context_len + 1 : t + 1]
            )
            action_history.append(
                actions[episode_id][t - args.context_len + 1 : t]
            )
            future_actions.append(
                actions[episode_id][t : t + args.rollout_steps]
            )
            future_states.append(
                states[episode_id][t + 1 : t + args.rollout_steps + 1]
            )
        return (
            torch.from_numpy(np.stack(state_history)).to(device),
            torch.from_numpy(np.stack(action_history)).to(device),
            torch.from_numpy(np.stack(future_actions)).to(device),
            torch.from_numpy(np.stack(future_states)).to(device),
        )

    model.train()
    for module in frozen_modules:
        module.eval()
    for step in range(1, args.steps + 1):
        state_history, action_history, future_actions, future_states = sample_batch()
        batch = state_history.shape[0]
        if args.delta_action_weight > 0:
            state_sequence = torch.cat([state_history, future_states], dim=1)
            z_sequence = model.encode(
                state_sequence.reshape(
                    batch * (args.context_len + args.rollout_steps), -1
                )
            ).reshape(batch, args.context_len + args.rollout_steps, -1)
            z_history = z_sequence[:, : args.context_len]
        else:
            with torch.no_grad():
                z_history = model.encode(
                    state_history.reshape(batch * args.context_len, -1)
                ).reshape(batch, args.context_len, -1)
        with torch.no_grad():
            target_state = future_states[:, -1]
            target = model.encode_target(target_state)
        head_rollouts = model.rollout_heads_context(
            z_history,
            action_history,
            future_actions,
            args.rollout_steps,
            detach_intermediate=True,
        )
        endpoint = head_rollouts[:, :, -1]
        latent_loss = (
            torch.nn.functional.mse_loss
            if args.latent_loss == "raw"
            else normalized_mse
        )
        loss = torch.stack(
            [latent_loss(prediction, target) for prediction in endpoint]
        ).mean()
        object_loss = torch.zeros((), device=device)
        if latent_slots > 1 and args.object_slot_weight > 0:
            target_object = target[:, :slot_dim]
            object_loss = torch.stack(
                [
                    latent_loss(prediction[:, :slot_dim], target_object)
                    for prediction in endpoint
                ]
            ).mean()
            loss = loss + args.object_slot_weight * object_loss
        explicit_pose_loss = torch.zeros((), device=device)
        if args.explicit_object_slot and args.explicit_pose_weight > 0:
            pose_width = int(object_dims[1] - object_dims[0])
            explicit_pose_loss = torch.stack(
                [
                    torch.nn.functional.mse_loss(
                        prediction[:, :pose_width], target[:, :pose_width]
                    )
                    for prediction in endpoint
                ]
            ).mean()
            loss = loss + args.explicit_pose_weight * explicit_pose_loss
        action_loss = torch.zeros((), device=device)
        if args.delta_action_weight > 0:
            latent_pairs = z_sequence[:, 1:] - z_sequence[:, :-1]
            action_targets = torch.cat(
                [action_history, future_actions], dim=1
            )
            action_prediction = model.action_decoder(
                latent_pairs.reshape(-1, model.latent_dim)
            ).reshape_as(action_targets)
            action_loss = torch.nn.functional.mse_loss(
                action_prediction, action_targets
            )
            loss = loss + args.delta_action_weight * action_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if encoder_parameters:
            model.update_target(args.ema)
        if step == 1 or step % 1000 == 0:
            print(
                json.dumps(
                    {
                        "event": "dexterous_contextual_train",
                        "step": step,
                        "loss": round(float(loss.detach()), 5),
                        "object_latent_loss": round(
                            float(object_loss.detach()), 5
                        ),
                        "delta_action_loss": round(
                            float(action_loss.detach()), 5
                        ),
                        "explicit_pose_loss": round(
                            float(explicit_pose_loss.detach()), 5
                        ),
                    }
                ),
                flush=True,
            )

    output_config = dict(config)
    output_config.update(
        {
            "arch": "dexterous_contextual",
            "context_len": args.context_len,
            "rollout_steps": args.rollout_steps,
            "base_model": str(args.base_model),
            "dynamics_object_slot_weight": args.object_slot_weight,
            "dynamics_objective": (
                "delta_jepa_latent_last_gradient_only"
                if args.delta_action_weight > 0
                else "frozen_jepa_latent_last_gradient_only"
            ),
            "ensemble_heads": args.ensemble_heads,
            "latent_difference_actions": args.delta_action_weight > 0,
            "action_decoder_depth": args.action_decoder_depth,
            "delta_action_weight": args.delta_action_weight,
            "encoder_lr": args.encoder_lr,
            "latent_loss": args.latent_loss,
            "warm_start_dynamics": args.warm_start_dynamics,
            "encoder_frozen": not bool(encoder_parameters),
            "explicit_object_slot": args.explicit_object_slot,
            "explicit_pose_weight": args.explicit_pose_weight,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "normalizer": {
                "mean": normalizer.mean,
                "std": normalizer.std,
            },
            "spec": spec.__dict__,
            "config": output_config,
        },
        args.out,
    )
    print(
        json.dumps(
            {
                "event": "dexterous_contextual_saved",
                "path": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
