"""Evaluate flow-proposed action chunks selected by the JEPA world model.

Runtime loop:

    encode current obs -> z_t
    encode goal/demo future obs -> z_goal
    sample N chunks from flow(a_chunk | z_t, z_goal)
    roll chunks through JEPA dynamics
    select lowest latent reachability cost
    execute first exec-k actions
"""
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

from jepa_robotics.envs import flatten_obs, goal_state_from_state, make_env
from jepa_robotics.evaluate import load_jepa_artifact, rollout_policy
from jepa_robotics.subgoals import load_subgoal_artifact, make_latent_subgoal_target_state
from jepa_robotics.tasks import resolve_task
from scripts.eval_diffusion_policy import sample_chunk
from scripts.train_fetch_flow_prior import fetch_geometry_features
from scripts.train_diffusion_policy import EpsNet, make_ddpm


class FlowJEPAChunkPolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        flow,
        flow_ckpt: dict,
        device,
        candidates: int,
        exec_k: int,
        flow_steps: int,
        goal_mode: str,
        subgoal_artifact: dict | None,
        action_l2_weight: float,
        action_delta_weight: float,
        target_horizon: int | None,
        latent_weight: float,
        state_weight: float,
        final_goal_weight: float,
    ) -> None:
        self.name = f"jepa_flow_chunk_select_{goal_mode}_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.flow = flow
        self.flow_ckpt = flow_ckpt
        self.device = device
        self.candidates = candidates
        self.exec_k = exec_k
        self.flow_steps = flow_steps
        self.goal_mode = goal_mode
        self.subgoal_artifact = subgoal_artifact
        self.action_l2_weight = action_l2_weight
        self.action_delta_weight = action_delta_weight
        self.target_horizon = target_horizon
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.final_goal_weight = final_goal_weight
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.cached: list[np.ndarray] = []
        self.ddpm = make_ddpm(int(flow_ckpt["diffusion_steps"]), device)

    def reset(self) -> None:
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        self.cached = []

    def _target_state(self, raw_state: np.ndarray) -> np.ndarray:
        if self.goal_mode == "local":
            if self.subgoal_artifact is None:
                raise ValueError("goal_mode=local requires a subgoal artifact")
            return make_latent_subgoal_target_state(raw_state, self.spec, self.subgoal_artifact)[1]
        return goal_state_from_state(raw_state, self.spec)

    @torch.no_grad()
    def _plan(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        raw_state = flatten_obs(obs)
        target_state = self._target_state(raw_state)
        state = torch.from_numpy(self.normalizer.encode(raw_state)).unsqueeze(0).to(self.device)
        target = torch.from_numpy(self.normalizer.encode(target_state)).unsqueeze(0).to(self.device)
        z = self.wm.encode(state)
        z_goal = self.wm.encode_target(target)
        future_horizons = list(self.flow_ckpt.get("future_horizons", [int(self.flow_ckpt["H"])]))
        target_h = int(self.target_horizon or min(future_horizons, key=lambda h: abs(h - int(self.flow_ckpt["H"]))))
        h_token = torch.tensor(
            [[float(target_h) / float(max(future_horizons))]],
            dtype=z.dtype,
            device=self.device,
        )
        cond = torch.cat([z, z_goal, h_token], dim=-1)
        if bool(self.flow_ckpt.get("concat_geometry", False)):
            geom = fetch_geometry_features(raw_state, target_state, self.spec)
            mean = np.asarray(self.flow_ckpt["geom_mean"], dtype=np.float32)
            std = np.asarray(self.flow_ckpt["geom_std"], dtype=np.float32)
            geom = (geom - mean) / np.maximum(std, 1e-6)
            geom_t = torch.from_numpy(geom).unsqueeze(0).to(self.device)
            cond = torch.cat([cond, geom_t], dim=-1)
        cond = cond.repeat(self.candidates, 1)
        chunks = sample_chunk(
            self.flow,
            self.ddpm,
            cond,
            int(self.flow_ckpt["chunk_dim"]),
            self.device,
            objective="flow",
            flow_steps=self.flow_steps,
        )
        H = int(self.flow_ckpt["H"])
        A = int(self.flow_ckpt["action_dim"])
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        action_tensor = chunks.view(self.candidates, H, A).clamp(low, high)
        traj_z = self.wm.predict_rollout(z.repeat(self.candidates, 1), action_tensor, H)
        latent_score = torch.sum(
            (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        scores = self.latent_weight * latent_score
        if self.state_weight > 0.0 or self.final_goal_weight > 0.0:
            pred_state = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            pred_achieved = pred_state[..., self.spec.obs_dim : self.spec.obs_dim + self.spec.goal_dim]
            target_achieved = torch.as_tensor(
                target_state[self.spec.obs_dim : self.spec.obs_dim + self.spec.goal_dim],
                dtype=pred_state.dtype,
                device=self.device,
            ).view(1, 1, -1)
            local_dist = torch.linalg.norm(pred_achieved - target_achieved, dim=-1)
            scores = scores + self.state_weight * (local_dist[:, -1] + 0.25 * local_dist.mean(dim=1))
            final_goal = torch.as_tensor(
                raw_state[self.spec.obs_dim + self.spec.goal_dim : self.spec.obs_dim + 2 * self.spec.goal_dim],
                dtype=pred_state.dtype,
                device=self.device,
            ).view(1, 1, -1)
            final_dist = torch.linalg.norm(pred_achieved - final_goal, dim=-1)
            scores = scores + self.final_goal_weight * (final_dist[:, -1] + 0.25 * final_dist.mean(dim=1))
        if self.action_l2_weight > 0.0:
            scores = scores + self.action_l2_weight * action_tensor.square().mean(dim=(1, 2))
        if self.action_delta_weight > 0.0:
            prev = torch.as_tensor(self.prev_action, dtype=torch.float32, device=self.device).view(1, 1, -1)
            first_delta = action_tensor[:, :1] - prev
            seq_delta = action_tensor[:, 1:] - action_tensor[:, :-1]
            delta = torch.cat([first_delta, seq_delta], dim=1).square().mean(dim=(1, 2))
            scores = scores + self.action_delta_weight * delta
        best = int(torch.argmin(scores).detach().cpu())
        return action_tensor[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs: dict[str, np.ndarray], env) -> np.ndarray:
        if not self.cached:
            plan = self._plan(obs, env)
            k = max(1, min(self.exec_k, len(plan)))
            self.cached = [plan[i].copy() for i in range(k)]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
        self.prev_action = action
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fetch_push")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--flow-path", type=Path, required=True)
    p.add_argument("--subgoal-path", type=Path, default=None)
    p.add_argument("--goal-mode", choices=["final", "local"], default="local")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--action-l2-weight", type=float, default=0.0)
    p.add_argument("--action-delta-weight", type=float, default=0.0)
    p.add_argument("--target-horizon", type=int, default=None,
                   help="Future-horizon token used for the runtime target. Defaults near policy chunk H.")
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.0,
                   help="Weight for decoded achieved_goal distance to the local target.")
    p.add_argument("--final-goal-weight", type=float, default=0.0,
                   help="Weight for decoded achieved_goal distance to the episode final goal.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    task = resolve_task(args.task, None)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    wm, normalizer, spec, cfg = load_jepa_artifact(args.model_path, device)
    flow_ckpt = torch.load(args.flow_path, map_location=device, weights_only=False)
    flow = EpsNet(
        int(flow_ckpt["chunk_dim"]),
        int(flow_ckpt["cond_dim"]),
        int(flow_ckpt["hidden"]),
        n_blocks=int(flow_ckpt["n_blocks"]),
    ).to(device)
    flow.load_state_dict(flow_ckpt["ema"])
    flow.eval()
    subgoal_artifact = load_subgoal_artifact(args.subgoal_path) if args.subgoal_path is not None else None
    if args.goal_mode == "local" and subgoal_artifact is None:
        raise ValueError("--goal-mode local requires --subgoal-path")

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    policy = FlowJEPAChunkPolicy(
        wm=wm,
        normalizer=normalizer,
        spec=spec,
        flow=flow,
        flow_ckpt=flow_ckpt,
        device=device,
        candidates=args.candidates,
        exec_k=args.exec_k,
        flow_steps=args.flow_steps,
        goal_mode=args.goal_mode,
        subgoal_artifact=subgoal_artifact,
        action_l2_weight=args.action_l2_weight,
        action_delta_weight=args.action_delta_weight,
        target_horizon=args.target_horizon,
        latent_weight=args.latent_weight,
        state_weight=args.state_weight,
        final_goal_weight=args.final_goal_weight,
    )
    metrics = rollout_policy(env, policy, episodes=args.episodes, seed=args.seed)
    env.close()
    row = {
        "event": "flow_jepa_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "flow_path": str(args.flow_path),
        "model_config": cfg,
        **metrics,
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
