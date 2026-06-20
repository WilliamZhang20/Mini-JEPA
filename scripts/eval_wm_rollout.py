"""Generic world-model rollout-accuracy eval.

The JEPA world model is a *predictor*, so we evaluate it by open-loop rollout
fidelity: encode a real state, roll the latent forward with the *actual* actions
taken, decode each predicted state via the state probe, and compare to ground
truth at every horizon. We report state MAE vs a static ("state never changes")
baseline; a model that has learned the dynamics must beat static. For ensemble
models we also report inter-head disagreement. Works for flat (Adroit) and goal
envs alike.

Exit code 0 if acceptable (model beats static by the margin), 3 if not — so a
caller can decide whether to resume training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import make_env, flatten_obs
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task


def collect_random(env_id, max_steps, n_eps, seed, action_dim):
    eps = []
    for ep in range(n_eps):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        states, actions = [flatten_obs(obs)], []
        term = trunc = False
        while not (term or trunc):
            a = env.action_space.sample().astype(np.float32)
            obs, _, term, trunc, _ = env.step(a)
            actions.append(a)
            states.append(flatten_obs(obs))
        env.close()
        eps.append((np.asarray(states), np.asarray(actions)))
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--max-horizon", type=int, default=16)
    p.add_argument("--accept-ratio", type=float, default=0.70,
                   help="Acceptable if model/static state-MAE ratio at the max horizon is <= this.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    model, normalizer, spec, config = load_jepa_artifact(args.model_path, device)
    model.eval()
    H = min(args.max_horizon, int(config["max_horizon"]))
    heads = int(config.get("ensemble_heads") or 1)
    print(f'{{"task": "{task.name}", "model": "{args.model_path.name}", "probe_horizon": {H}, '
          f'"ensemble_heads": {heads}, "state_dim": {spec.state_dim}}}', flush=True)

    eps = collect_random(task.env_id, task.max_episode_steps, args.episodes, args.seed, spec.action_dim)

    model_err = {h: [] for h in range(1, H + 1)}
    static_err = {h: [] for h in range(1, H + 1)}
    disagree = {h: [] for h in range(1, H + 1)}
    for states, actions in eps:
        T = len(actions)
        for t0 in range(0, max(1, T - H)):
            raw0 = states[t0]
            z0 = model.encode(torch.from_numpy(normalizer.encode(raw0)).unsqueeze(0).to(device))
            aw = torch.from_numpy(actions[t0:t0 + H]).unsqueeze(0).to(device)
            with torch.no_grad():
                traj = model.predict_rollout(z0, aw, H)
                pred = normalizer.decode_tensor(model.state_probe(traj))[0].cpu().numpy()
                if heads > 1:
                    hd = model.rollout_heads(z0, aw, H).var(dim=0).mean(dim=-1)[0].cpu().numpy()
                else:
                    hd = np.zeros(H)
            for h in range(1, H + 1):
                gt = states[t0 + h]
                model_err[h].append(np.mean(np.abs(pred[h - 1] - gt)))
                static_err[h].append(np.mean(np.abs(raw0 - gt)))
                disagree[h].append(float(hd[h - 1]))

    print("horizon  state_mae   static_mae   model/static   disagreement")
    ratios = {}
    for h in [1, 2, 4, 8, 12, 16]:
        if h > H:
            continue
        mm = float(np.mean(model_err[h])); sm = float(np.mean(static_err[h]))
        r = mm / sm if sm > 1e-9 else float("nan")
        ratios[h] = r
        print(f"{h:>5}   {mm:9.4f}   {sm:10.4f}   {r:11.3f}   {float(np.mean(disagree[h])):11.4f}", flush=True)

    final_ratio = ratios.get(H, ratios[max(ratios)])
    acceptable = final_ratio <= args.accept_ratio
    print(f'{{"verdict": "{"acceptable" if acceptable else "unacceptable"}", '
          f'"final_ratio": {final_ratio:.3f}, "accept_threshold": {args.accept_ratio}}}', flush=True)
    sys.exit(0 if acceptable else 3)


if __name__ == "__main__":
    main()
