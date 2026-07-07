"""Evaluate same-latent HWM planning with a future-conditioned inverse low level."""
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

from jepa_robotics.algos.hwm import LatentMacroPredictor, MacroActionEncoder
from jepa_robotics.algos.priors import InversePrior
from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


class StartGoalIndex:
    def __init__(self, starts: np.ndarray, finals: np.ndarray, normalizer) -> None:
        self.starts = starts.astype(np.float32)
        self.finals = finals.astype(np.float32)
        self.norm_starts = normalizer.encode(self.starts).astype(np.float32)
        try:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(self.norm_starts)
        except Exception:
            self.tree = None

    def query(self, state: np.ndarray, normalizer) -> np.ndarray:
        x = normalizer.encode(state).astype(np.float32)
        if self.tree is not None:
            _dist, idx = self.tree.query(x, k=1)
            return self.finals[int(idx)]
        idx = int(np.argmin(np.linalg.norm(self.norm_starts - x[None], axis=1)))
        return self.finals[idx]


@torch.no_grad()
def cem_plan_macro(predictor, z, z_goal, cfg, device, *, horizon, samples, iters, elite_frac, goal_weight, subgoal_weight):
    macro_dim = int(cfg["macro_dim"])
    mean0 = np.asarray(cfg["macro_mean"], dtype=np.float32)
    std0 = np.maximum(np.asarray(cfg["macro_std"], dtype=np.float32), 0.05)
    mean = np.tile(mean0, (horizon, 1))
    std = np.tile(std0 * 2.0, (horizon, 1))
    best_seq = mean.copy()
    best_score = float("inf")
    for _ in range(iters):
        macros_np = np.random.normal(mean, std, size=(samples, horizon, macro_dim)).astype(np.float32)
        macros = torch.from_numpy(macros_np).to(device)
        z_roll = z.repeat(samples, 1)
        subgoal = None
        for k in range(horizon):
            z_roll = predictor(z_roll, macros[:, k])
            if k == 0:
                subgoal = z_roll
        score = goal_weight * torch.sum(
            (F.normalize(z_roll, dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
            dim=-1,
        )
        if subgoal_weight > 0 and subgoal is not None:
            score = score + subgoal_weight * torch.sum(
                (F.normalize(subgoal, dim=-1) - F.normalize(z_goal, dim=-1)) ** 2,
                dim=-1,
            )
        score_np = score.detach().cpu().numpy()
        order = np.argsort(score_np)
        if float(score_np[order[0]]) < best_score:
            best_score = float(score_np[order[0]])
            best_seq = macros_np[order[0]].copy()
        elite = macros_np[order[: max(1, int(samples * elite_frac))]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.03)
    macro_first = torch.from_numpy(best_seq[:1]).to(device)
    z_next = predictor(z, macro_first)
    return z_next


class LatentHWMPolicy:
    def __init__(
        self,
        *,
        wm,
        normalizer,
        spec,
        hwm_pred,
        hwm_cfg,
        inverse,
        inverse_ckpt,
        goal_index,
        device,
        macro_horizon,
        macro_samples,
        macro_iters,
        exec_k,
        candidates,
        noise_std,
        target_horizon,
    ) -> None:
        self.name = "latent_hwm_inverse"
        self.wm = wm
        self.normalizer = normalizer
        self.spec = spec
        self.hwm_pred = hwm_pred
        self.hwm_cfg = hwm_cfg
        self.inverse = inverse
        self.inverse_ckpt = inverse_ckpt
        self.goal_index = goal_index
        self.device = device
        self.macro_horizon = macro_horizon
        self.macro_samples = macro_samples
        self.macro_iters = macro_iters
        self.exec_k = exec_k
        self.candidates = candidates
        self.noise_std = noise_std
        self.target_horizon = target_horizon
        self.cached: list[np.ndarray] = []
        self.goal_state: np.ndarray | None = None

    def reset(self) -> None:
        self.cached = []
        self.goal_state = None

    @torch.no_grad()
    def _plan(self, obs, env):
        raw = flatten_obs(obs)
        if self.goal_state is None:
            self.goal_state = self.goal_index.query(raw, self.normalizer)
        s = torch.from_numpy(self.normalizer.encode(raw)).unsqueeze(0).to(self.device)
        g = torch.from_numpy(self.normalizer.encode(self.goal_state)).unsqueeze(0).to(self.device)
        z = self.wm.encode(s)
        z_goal = self.wm.encode_target(g)
        z_subgoal = cem_plan_macro(
            self.hwm_pred,
            z,
            z_goal,
            self.hwm_cfg,
            self.device,
            horizon=self.macro_horizon,
            samples=self.macro_samples,
            iters=self.macro_iters,
            elite_frac=0.1,
            goal_weight=1.0,
            subgoal_weight=0.05,
        )
        horizons = list(self.inverse_ckpt.get("future_horizons", [int(self.inverse_ckpt["H"])]))
        target_h = int(self.target_horizon or min(max(horizons), int(self.hwm_cfg["stride"])))
        h_token = torch.tensor([[float(target_h) / float(max(horizons))]], dtype=z.dtype, device=self.device)
        cond = torch.cat([z, z_subgoal, h_token], dim=-1)
        base = self.inverse(cond).view(1, int(self.inverse_ckpt["H"]), int(self.inverse_ckpt["action_dim"]))
        if self.candidates > 1 and self.noise_std > 0:
            actions = base.repeat(self.candidates, 1, 1) + torch.randn(self.candidates, *base.shape[1:], device=self.device) * self.noise_std
            actions[0] = base[0]
        else:
            actions = base.repeat(self.candidates, 1, 1)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        actions = actions.clamp(low, high)
        traj = self.wm.predict_rollout(z.repeat(actions.shape[0], 1), actions, int(self.inverse_ckpt["H"]))
        score = torch.sum((F.normalize(traj[:, -1], dim=-1) - F.normalize(z_subgoal, dim=-1)) ** 2, dim=-1)
        best = int(torch.argmin(score).detach().cpu())
        return actions[best].detach().cpu().numpy().astype(np.float32)

    def act(self, obs, env):
        if not self.cached:
            plan = self._plan(obs, env)
            k = max(1, min(self.exec_k, len(plan)))
            self.cached = [plan[i].copy() for i in range(k)]
        return np.clip(self.cached.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--hwm-path", type=Path, required=True)
    p.add_argument("--inverse-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--macro-horizon", type=int, default=3)
    p.add_argument("--macro-samples", type=int, default=128)
    p.add_argument("--macro-iters", type=int, default=4)
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.03)
    p.add_argument("--target-horizon", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    hwm = torch.load(args.hwm_path, map_location=dev, weights_only=False)
    hcfg = hwm["config"]
    _macro = MacroActionEncoder(int(hcfg["action_dim"]), int(hcfg["macro_dim"])).to(dev)
    _macro.load_state_dict(hwm["macro_state"])
    hwm_pred = LatentMacroPredictor(
        int(hcfg["latent_dim"]),
        int(hcfg["macro_dim"]),
        int(hcfg["hidden"]),
        int(hcfg["n_blocks"]),
    ).to(dev)
    hwm_pred.load_state_dict(hwm["predictor_state"])
    hwm_pred.eval()
    inv_ckpt = torch.load(args.inverse_path, map_location=dev, weights_only=False)
    inverse = InversePrior(int(inv_ckpt["cond_dim"]), int(inv_ckpt["chunk_dim"]), int(inv_ckpt["hidden"]), int(inv_ckpt["n_blocks"])).to(dev)
    inverse.load_state_dict(inv_ckpt["state_dict"])
    inverse.eval()
    goal_index = StartGoalIndex(np.asarray(hwm["start_states"]), np.asarray(hwm["final_states"]), norm)
    policy = LatentHWMPolicy(
        wm=wm,
        normalizer=norm,
        spec=spec,
        hwm_pred=hwm_pred,
        hwm_cfg=hcfg,
        inverse=inverse,
        inverse_ckpt=inv_ckpt,
        goal_index=goal_index,
        device=dev,
        macro_horizon=args.macro_horizon,
        macro_samples=args.macro_samples,
        macro_iters=args.macro_iters,
        exec_k=args.exec_k,
        candidates=args.candidates,
        noise_std=args.noise_std,
        target_horizon=args.target_horizon,
    )
    tasks_done, successes = [], []
    for ep in range(args.episodes):
        env = make_env(task.env_id, seed=args.seed + ep, max_episode_steps=task.max_episode_steps)
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()
        term = trunc = False
        info = {}
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(policy.act(obs, env))
        tasks_done.append(int(info.get("tasks_done", 0)))
        successes.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    row = {
        "event": "latent_hwm_eval",
        "task": task.name,
        "episodes": int(args.episodes),
        "success_rate": float(np.mean(successes)),
        "mean_tasks": float(np.mean(tasks_done)) if tasks_done else float(np.mean(successes)),
    }
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
