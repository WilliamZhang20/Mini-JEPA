"""Evaluate Relocate contact-trace-conditioned inverse chunks."""
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

from jepa_robotics.algos.priors import InversePrior
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


class ContactTraceIndex:
    def __init__(self, states: np.ndarray, futures: np.ndarray, traces: np.ndarray, normalizer) -> None:
        self.states = states.astype(np.float32)
        self.futures = futures.astype(np.float32)
        self.traces = traces.astype(np.float32)
        self.norm_states = normalizer.encode(self.states).astype(np.float32)
        try:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(self.norm_states)
        except Exception:
            self.tree = None

    def query(self, state: np.ndarray, normalizer, k: int, trace_weight: float) -> tuple[np.ndarray, np.ndarray]:
        x = normalizer.encode(state).astype(np.float32)
        kk = min(max(1, k), len(self.states))
        if self.tree is not None:
            dist, idx = self.tree.query(x, k=kk)
            dist = np.atleast_1d(dist).astype(np.float32)
            idx = np.atleast_1d(idx).astype(np.int64)
        else:
            all_dist = np.linalg.norm(self.norm_states - x[None], axis=1)
            idx = np.argpartition(all_dist, kk - 1)[:kk]
            dist = all_dist[idx]
        if trace_weight > 0:
            current = np.array([np.linalg.norm(state[30:33]), np.linalg.norm(state[36:39])], dtype=np.float32)
            trace0 = self.traces[idx, :2]
            score = dist + trace_weight * np.linalg.norm(trace0 - current[None], axis=-1)
            best = int(idx[int(np.argmin(score))])
        else:
            best = int(idx[int(np.argmin(dist))])
        return self.futures[best], self.traces[best]


class ContactTraceInversePolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        prior,
        ckpt,
        index,
        device,
        candidates: int,
        noise_std: float,
        exec_k: int,
        latent_weight: float,
        state_weight: float,
        action_delta_weight: float,
        trace_k: int,
        trace_weight: float,
    ) -> None:
        self.name = f"relocate_trace_inverse_n{candidates}"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.prior = prior
        self.ckpt = ckpt
        self.index = index
        self.device = device
        self.candidates = candidates
        self.noise_std = noise_std
        self.exec_k = exec_k
        self.latent_weight = latent_weight
        self.state_weight = state_weight
        self.action_delta_weight = action_delta_weight
        self.trace_k = trace_k
        self.trace_weight = trace_weight
        self.cached: list[np.ndarray] = []
        self.prev_action = np.zeros(spec.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.cached = []
        self.prev_action = np.zeros(self.spec.action_dim, dtype=np.float32)

    @torch.no_grad()
    def _plan(self, obs, env):
        raw = flatten_obs(obs)
        target_state, trace = self.index.query(raw, self.normalizer, self.trace_k, self.trace_weight)
        s = torch.from_numpy(self.normalizer.encode(raw)).unsqueeze(0).to(self.device)
        tgt = torch.from_numpy(self.normalizer.encode(target_state)).unsqueeze(0).to(self.device)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(tgt)
        h_token = torch.ones(1, 1, dtype=z.dtype, device=self.device)
        trace_t = torch.from_numpy(trace).view(1, -1).to(self.device)
        cond = torch.cat([z, z_goal, h_token, s, tgt, trace_t], dim=-1)
        base = self.prior(cond).view(1, int(self.ckpt["H"]), int(self.ckpt["action_dim"]))
        if self.candidates > 1 and self.noise_std > 0:
            actions = base.repeat(self.candidates, 1, 1) + torch.randn(self.candidates, *base.shape[1:], device=self.device) * self.noise_std
            actions[0] = base[0]
        else:
            actions = base.repeat(self.candidates, 1, 1)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        actions = actions.clamp(low, high)
        if actions.shape[0] == 1 and self.state_weight <= 0 and self.action_delta_weight <= 0:
            return actions[0].detach().cpu().numpy().astype(np.float32)
        traj_z = self.wm.predict_rollout(z.repeat(actions.shape[0], 1), actions, int(self.ckpt["H"]))
        scores = self.latent_weight * torch.sum(
            (F.normalize(traj_z[:, -1], dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        if self.state_weight > 0:
            pred = self.normalizer.decode_tensor(self.wm.state_probe(traj_z))
            target_t = torch.as_tensor(target_state, dtype=pred.dtype, device=self.device).view(1, 1, -1)
            dist = torch.linalg.norm(pred - target_t, dim=-1)
            scores = scores + self.state_weight * (dist[:, -1] + 0.25 * dist.mean(dim=1))
        if self.action_delta_weight > 0:
            prev = torch.as_tensor(self.prev_action, dtype=torch.float32, device=self.device).view(1, 1, -1)
            delta = torch.cat([actions[:, :1] - prev, actions[:, 1:] - actions[:, :-1]], dim=1)
            scores = scores + self.action_delta_weight * delta.square().mean(dim=(1, 2))
        best = int(torch.argmin(scores).detach().cpu())
        return actions[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs, env):
        if not self.cached:
            plan = self._plan(obs, env)
            k = max(1, min(self.exec_k, len(plan)))
            self.cached = [plan[i].copy() for i in range(k)]
        action = np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
        self.prev_action = action
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="adroit_relocate")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--inverse-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--exec-k", type=int, default=1)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--state-weight", type=float, default=0.1)
    p.add_argument("--action-delta-weight", type=float, default=0.001)
    p.add_argument("--trace-k", type=int, default=1)
    p.add_argument("--trace-weight", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    ckpt = torch.load(args.inverse_path, map_location=dev, weights_only=False)
    prior = InversePrior(int(ckpt["cond_dim"]), int(ckpt["chunk_dim"]), int(ckpt["hidden"]), int(ckpt["n_blocks"])).to(dev)
    prior.load_state_dict(ckpt["state_dict"])
    prior.eval()
    index = ContactTraceIndex(np.asarray(ckpt["bank_states"]), np.asarray(ckpt["bank_futures"]), np.asarray(ckpt["bank_traces"]), norm)
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    policy = ContactTraceInversePolicy(
        wm=wm,
        normalizer=norm,
        spec=spec,
        prior=prior,
        ckpt=ckpt,
        index=index,
        device=dev,
        candidates=args.candidates,
        noise_std=args.noise_std,
        exec_k=args.exec_k,
        latent_weight=args.latent_weight,
        state_weight=args.state_weight,
        action_delta_weight=args.action_delta_weight,
        trace_k=args.trace_k,
        trace_weight=args.trace_weight,
    )
    successes = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()
        term = trunc = False
        info = {}
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(policy.act(obs, env))
        successes.append(float(info.get("is_success", info.get("success", 0.0))))
    env.close()
    row = {
        "event": "relocate_trace_inverse_eval",
        "task": task.name,
        "model_path": str(args.model_path),
        "inverse_path": str(args.inverse_path),
        "model_config": cfg,
        "policy": policy.name,
        "episodes": float(args.episodes),
        "success_rate": float(np.mean(successes)),
    }
    print(json.dumps(row, default=str), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
