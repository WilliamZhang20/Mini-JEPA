from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import Episode
from .envs import ObsSpec


FETCH_PICK_PHASES = ("approach", "pregrasp", "grasp", "lift", "place")
FETCH_PUSH_PHASES = ("approach", "push", "place")
FETCH_SLIDE_PHASES = ("approach", "strike", "coast")


def _fetch_parts(state: np.ndarray, spec: ObsSpec):
    obs = np.asarray(state[: spec.obs_dim], dtype=np.float32)
    grip = obs[:3]
    obj = state[spec.obs_dim : spec.obs_dim + spec.goal_dim]
    goal = state[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim]
    finger = obs[9:11] if spec.obs_dim >= 11 else np.array([0.05, 0.05], dtype=np.float32)
    return obs, grip, obj, goal, finger


def _encoded_demo_latents(episodes: list[Episode], normalizer, model, device, max_states: int):
    demo_states = np.concatenate([ep.states for ep in episodes], axis=0).astype(np.float32)
    if demo_states.shape[0] > max_states:
        idx = np.linspace(0, demo_states.shape[0] - 1, max_states).astype(np.int64)
        demo_states = demo_states[idx]
    with torch.no_grad():
        chunks = []
        states_t = torch.from_numpy(normalizer.encode(demo_states)).to(device)
        for i in range(0, states_t.shape[0], 8192):
            chunks.append(model.encode(states_t[i : i + 8192]).detach().cpu())
        demo_latents = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    return demo_states, demo_latents


def _artifact(kind: str, phases, templates, episodes, spec, normalizer, model, device, max_states):
    demo_states, demo_latents = _encoded_demo_latents(episodes, normalizer, model, device, max_states)
    return {
        "kind": kind,
        "phases": tuple(phases),
        "templates": templates,
        "demo_states": demo_states,
        "demo_latents": demo_latents,
        "latent_mean": demo_latents.mean(axis=0).astype(np.float32),
        "latent_std": (demo_latents.std(axis=0) + 1e-4).astype(np.float32),
        "spec": {
            "obs_dim": spec.obs_dim,
            "goal_dim": spec.goal_dim,
            "state_dim": spec.state_dim,
            "action_dim": spec.action_dim,
            "is_goal_env": spec.is_goal_env,
        },
    }


def _classify_fetch_pick_state(state: np.ndarray, spec: ObsSpec) -> str:
    _obs, grip, obj, goal, finger = _fetch_parts(state, spec)
    xy = float(np.linalg.norm(grip[:2] - obj[:2]))
    d3 = float(np.linalg.norm(grip - obj))
    dist_goal = float(np.linalg.norm(obj - goal))
    fingers_open = float(np.sum(finger)) > 0.075
    lifted = float(obj[2]) > 0.46

    if dist_goal < 0.045:
        return "place"
    if not fingers_open and lifted:
        return "lift"
    if d3 < 0.060 and not fingers_open:
        return "grasp"
    if xy < 0.035 and d3 < 0.090:
        return "pregrasp"
    return "approach"


def build_fetch_pick_subgoal_artifact(
    episodes: list[Episode],
    spec: ObsSpec,
    *,
    model,
    normalizer,
    device: torch.device,
    max_states: int = 200_000,
) -> dict:
    if spec.obs_dim < 25 or spec.goal_dim < 3:
        raise ValueError("Fetch pick/place subgoals require the 25-D Fetch object observation.")

    rows: dict[str, list[np.ndarray]] = {phase: [] for phase in FETCH_PICK_PHASES}
    total = 0
    for ep in episodes:
        for state in ep.states:
            rows[_classify_fetch_pick_state(state, spec)].append(np.asarray(state, dtype=np.float32))
            total += 1
            if total >= max_states:
                break
        if total >= max_states:
            break

    fallback_offsets = {
        "approach": np.array([0.0, 0.0, 0.060], dtype=np.float32),
        "pregrasp": np.array([0.0, 0.0, 0.010], dtype=np.float32),
        "grasp": np.array([0.0, 0.0, 0.005], dtype=np.float32),
        "lift": np.array([0.0, 0.0, 0.010], dtype=np.float32),
        "place": np.array([0.0, 0.0, 0.010], dtype=np.float32),
    }
    templates = {}
    for phase in FETCH_PICK_PHASES:
        states = rows[phase]
        if states:
            grips, objs, fingers = [], [], []
            for state in states:
                _obs, grip, obj, goal, finger = _fetch_parts(state, spec)
                grips.append(grip - obj)
                objs.append(obj - goal)
                fingers.append(finger)
            grip_obj_offset = np.median(np.stack(grips), axis=0).astype(np.float32)
            obj_goal_offset = np.median(np.stack(objs), axis=0).astype(np.float32)
            finger = np.median(np.stack(fingers), axis=0).astype(np.float32)
            count = len(states)
        else:
            grip_obj_offset = fallback_offsets[phase]
            obj_goal_offset = np.zeros(3, dtype=np.float32)
            finger = np.array([0.050, 0.050], dtype=np.float32)
            count = 0
        templates[phase] = {
            "grip_obj_offset": grip_obj_offset,
            "obj_goal_offset": obj_goal_offset,
            "finger": finger,
            "count": count,
        }
    return _artifact(
        "fetch_pick_latent_subgoals", FETCH_PICK_PHASES, templates,
        episodes, spec, normalizer, model, device, max_states,
    )


def _classify_fetch_push_state(state: np.ndarray, spec: ObsSpec) -> str:
    _obs, grip, obj, goal, _finger = _fetch_parts(state, spec)
    dist_goal = float(np.linalg.norm(obj[:2] - goal[:2]))
    if dist_goal < 0.045:
        return "place"
    to_goal = goal[:2] - obj[:2]
    d = to_goal / (float(np.linalg.norm(to_goal)) + 1e-6)
    rel = grip[:2] - obj[:2]
    s = float(np.dot(rel, d))
    lateral = float(np.linalg.norm(rel - s * d))
    low = float(abs(grip[2] - obj[2])) < 0.035
    return "push" if s < -0.020 and lateral < 0.050 and low else "approach"


def build_fetch_push_subgoal_artifact(
    episodes: list[Episode],
    spec: ObsSpec,
    *,
    model,
    normalizer,
    device: torch.device,
    max_states: int = 200_000,
) -> dict:
    if spec.obs_dim < 25 or spec.goal_dim < 3:
        raise ValueError("Fetch push subgoals require the 25-D Fetch object observation.")

    rows: dict[str, list[np.ndarray]] = {phase: [] for phase in FETCH_PUSH_PHASES}
    total = 0
    for ep in episodes:
        for state in ep.states:
            rows[_classify_fetch_push_state(state, spec)].append(np.asarray(state, dtype=np.float32))
            total += 1
            if total >= max_states:
                break
        if total >= max_states:
            break

    templates = {}
    for phase in FETCH_PUSH_PHASES:
        states = rows[phase]
        if states:
            grips, fingers = [], []
            for state in states:
                _obs, grip, obj, _goal, finger = _fetch_parts(state, spec)
                grips.append(grip - obj)
                fingers.append(finger)
            grip_obj_offset = np.median(np.stack(grips), axis=0).astype(np.float32)
            finger = np.median(np.stack(fingers), axis=0).astype(np.float32)
            count = len(states)
        else:
            grip_obj_offset = np.array([-0.070, 0.0, 0.0], dtype=np.float32)
            finger = np.array([0.025, 0.025], dtype=np.float32)
            count = 0
        templates[phase] = {
            "grip_obj_offset": grip_obj_offset,
            "finger": finger,
            "count": count,
        }
    return _artifact(
        "fetch_push_latent_subgoals", FETCH_PUSH_PHASES, templates,
        episodes, spec, normalizer, model, device, max_states,
    )


def _classify_fetch_slide_state(state: np.ndarray, spec: ObsSpec) -> str:
    _obs, grip, obj, goal, _finger = _fetch_parts(state, spec)
    dist = float(np.linalg.norm(obj[:2] - goal[:2]))
    if dist < 0.050:
        return "coast"
    to_goal = goal[:2] - obj[:2]
    d = to_goal / (float(np.linalg.norm(to_goal)) + 1e-6)
    rel = grip[:2] - obj[:2]
    s = float(np.dot(rel, d))
    lateral = float(np.linalg.norm(rel - s * d))
    low = abs(float(grip[2] - obj[2])) < 0.040
    return "strike" if s < 0.020 and lateral < 0.045 and low else "approach"


def build_fetch_slide_subgoal_artifact(
    episodes: list[Episode],
    spec: ObsSpec,
    *,
    model,
    normalizer,
    device: torch.device,
    max_states: int = 200_000,
) -> dict:
    if spec.obs_dim < 25 or spec.goal_dim < 3:
        raise ValueError("Fetch slide subgoals require the 25-D Fetch object observation.")

    rows: dict[str, list[np.ndarray]] = {phase: [] for phase in FETCH_SLIDE_PHASES}
    total = 0
    for ep in episodes:
        for state in ep.states:
            rows[_classify_fetch_slide_state(state, spec)].append(np.asarray(state, dtype=np.float32))
            total += 1
            if total >= max_states:
                break
        if total >= max_states:
            break

    templates = {}
    for phase in FETCH_SLIDE_PHASES:
        states = rows[phase]
        if states:
            grips, fingers = [], []
            for state in states:
                _obs, grip, obj, _goal, finger = _fetch_parts(state, spec)
                grips.append(grip - obj)
                fingers.append(finger)
            grip_obj_offset = np.median(np.stack(grips), axis=0).astype(np.float32)
            finger = np.median(np.stack(fingers), axis=0).astype(np.float32)
            count = len(states)
        else:
            grip_obj_offset = np.array([-0.075, 0.0, 0.0], dtype=np.float32)
            finger = np.array([0.025, 0.025], dtype=np.float32)
            count = 0
        templates[phase] = {
            "grip_obj_offset": grip_obj_offset,
            "finger": finger,
            "count": count,
        }
    return _artifact(
        "fetch_slide_latent_subgoals", FETCH_SLIDE_PHASES, templates,
        episodes, spec, normalizer, model, device, max_states,
    )


def save_subgoal_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_subgoal_artifact(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def make_fetch_pick_target_state(raw_state: np.ndarray, spec: ObsSpec, artifact: dict) -> tuple[str, np.ndarray]:
    target = np.array(raw_state, copy=True).astype(np.float32)
    _obs, grip, obj, goal, finger = _fetch_parts(raw_state, spec)
    xy = float(np.linalg.norm(grip[:2] - obj[:2]))
    d3 = float(np.linalg.norm(grip - obj))
    fingers_open = float(np.sum(finger)) > 0.075
    lifted = float(obj[2]) > 0.46
    dist_goal = float(np.linalg.norm(obj - goal))

    if dist_goal < 0.045:
        phase = "place"
    elif not lifted and (xy > 0.030 or grip[2] < obj[2] + 0.045):
        phase = "approach"
    elif fingers_open and d3 > 0.040:
        phase = "pregrasp"
    elif fingers_open:
        phase = "grasp"
    elif not lifted:
        phase = "lift"
    else:
        phase = "place"

    tmpl = artifact["templates"][phase]
    offset = np.asarray(tmpl["grip_obj_offset"], dtype=np.float32)
    finger_target = np.asarray(tmpl["finger"], dtype=np.float32)
    if phase in ("approach", "pregrasp"):
        target_obj = obj.copy()
        target_grip = obj + offset
        finger_target = np.maximum(finger_target, np.array([0.045, 0.045], dtype=np.float32))
    elif phase == "grasp":
        target_obj = obj.copy()
        target_grip = obj + offset
        finger_target = np.minimum(finger_target, np.array([0.026, 0.026], dtype=np.float32))
    elif phase == "lift":
        target_obj = obj.copy()
        target_obj[2] = max(float(obj[2] + 0.10), float(goal[2] + 0.03), 0.52)
        target_grip = target_obj + offset
        finger_target = np.minimum(finger_target, np.array([0.026, 0.026], dtype=np.float32))
    else:
        delta = goal - obj
        norm = float(np.linalg.norm(delta))
        step = delta if norm <= 0.10 else delta / (norm + 1e-6) * 0.10
        target_obj = obj + step
        if norm <= 0.10:
            target_obj = goal.copy()
        target_grip = target_obj + offset
        finger_target = np.minimum(finger_target, np.array([0.026, 0.026], dtype=np.float32))

    target[:3] = target_grip
    target[3:6] = target_obj
    target[6:9] = target_obj - target_grip
    target[9:11] = finger_target
    target[14:25] = 0.0
    target[spec.obs_dim : spec.obs_dim + spec.goal_dim] = target_obj
    target[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim] = goal
    return phase, target


def make_fetch_push_target_state(raw_state: np.ndarray, spec: ObsSpec, artifact: dict) -> tuple[str, np.ndarray]:
    target = np.array(raw_state, copy=True).astype(np.float32)
    _obs, grip, obj, goal, _finger = _fetch_parts(raw_state, spec)
    delta = goal[:2] - obj[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 0.045:
        phase = "place"
    else:
        d = delta / (dist + 1e-6)
        rel = grip[:2] - obj[:2]
        s = float(np.dot(rel, d))
        lateral = float(np.linalg.norm(rel - s * d))
        low = abs(float(grip[2] - obj[2])) < 0.035
        phase = "push" if s < -0.020 and lateral < 0.050 and low else "approach"

    if dist > 1e-6:
        d3 = np.array([delta[0] / dist, delta[1] / dist, 0.0], dtype=np.float32)
    else:
        d3 = np.zeros(3, dtype=np.float32)
    if phase == "approach":
        target_obj = obj.copy()
        target_grip = obj - 0.075 * d3
        target_grip[2] = obj[2] + 0.015
    elif phase == "push":
        step = min(0.08, 0.7 * dist)
        target_obj = obj.copy()
        target_obj[:2] = obj[:2] + step * d3[:2]
        target_grip = target_obj - 0.055 * d3
        target_grip[2] = obj[2] + 0.005
    else:
        target_obj = goal.copy()
        target_grip = target_obj - 0.055 * d3
        target_grip[2] = target_obj[2] + 0.005

    finger_target = np.minimum(
        np.asarray(artifact["templates"][phase]["finger"], dtype=np.float32),
        np.array([0.026, 0.026], dtype=np.float32),
    )
    target[:3] = target_grip
    target[3:6] = target_obj
    target[6:9] = target_obj - target_grip
    target[9:11] = finger_target
    target[14:25] = 0.0
    target[spec.obs_dim : spec.obs_dim + spec.goal_dim] = target_obj
    target[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim] = goal
    return phase, target


def make_fetch_slide_target_state(raw_state: np.ndarray, spec: ObsSpec, artifact: dict) -> tuple[str, np.ndarray]:
    target = np.array(raw_state, copy=True).astype(np.float32)
    _obs, grip, obj, goal, _finger = _fetch_parts(raw_state, spec)
    delta = goal[:2] - obj[:2]
    dist = float(np.linalg.norm(delta))
    if dist < 0.050:
        phase = "coast"
    else:
        d = delta / (dist + 1e-6)
        rel = grip[:2] - obj[:2]
        s = float(np.dot(rel, d))
        lateral = float(np.linalg.norm(rel - s * d))
        low = abs(float(grip[2] - obj[2])) < 0.040
        phase = "strike" if s < 0.020 and lateral < 0.045 and low else "approach"

    if dist > 1e-6:
        d3 = np.array([delta[0] / dist, delta[1] / dist, 0.0], dtype=np.float32)
    else:
        d3 = np.zeros(3, dtype=np.float32)
    if phase == "approach":
        target_obj = obj.copy()
        target_grip = obj - 0.075 * d3
        target_grip[2] = obj[2] + 0.010
    elif phase == "strike":
        step = min(0.18, max(0.08, 0.35 * dist))
        target_obj = obj.copy()
        target_obj[:2] = obj[:2] + step * d3[:2]
        target_grip = target_obj + 0.020 * d3
        target_grip[2] = obj[2] + 0.005
    else:
        target_obj = goal.copy()
        target_grip = obj + np.asarray(artifact["templates"][phase]["grip_obj_offset"], dtype=np.float32)

    target[:3] = target_grip
    target[3:6] = target_obj
    target[6:9] = target_obj - target_grip
    if spec.obs_dim >= 11:
        target[9:11] = np.array([0.025, 0.025], dtype=np.float32)
    target[14:25] = 0.0
    target[spec.obs_dim : spec.obs_dim + spec.goal_dim] = target_obj
    target[spec.obs_dim + spec.goal_dim : spec.obs_dim + 2 * spec.goal_dim] = goal
    return phase, target


def make_latent_subgoal_target_state(raw_state: np.ndarray, spec: ObsSpec, artifact: dict) -> tuple[str, np.ndarray]:
    kind = artifact.get("kind")
    if kind == "fetch_push_latent_subgoals":
        return make_fetch_push_target_state(raw_state, spec, artifact)
    if kind == "fetch_pick_latent_subgoals":
        return make_fetch_pick_target_state(raw_state, spec, artifact)
    if kind == "fetch_slide_latent_subgoals":
        return make_fetch_slide_target_state(raw_state, spec, artifact)
    raise ValueError(f"Unsupported latent subgoal artifact kind: {kind!r}")
