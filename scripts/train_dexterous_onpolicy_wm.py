"""On-policy world-model calibration loop for demo-free in-hand reorientation.

Long-horizon iCEM over the DexterousJEPA world model breaks the rotation ceiling
(the object gaits ~70 deg) but wanders, because the WM — trained only on OU — is
inaccurate on the action distribution the planner actually uses (model
exploitation; WM 8-step error ~12 deg on iCEM actions vs ~8 deg on OU). GC-PMPC
(arXiv:2504.21585) fixes this by ALTERNATING plan<->retrain: collect on-policy
transitions with the current iCEM, retrain the WM on them, repeat. The WM sharpens
where the planner explores, so steering and terminal precision improve. Pure SSL:
no demos, no reward-shaped policy — just a self-supervised WM + planning.

    round r:
      1. collect episodes_per_round on the task with the current iCEM (random goals)
      2. buffer = OU seed + all collected on-policy episodes
      3. retrain WM (long horizons) on the buffer -> wm_round{r}.pt
      4. report the round's task success rate; next round plans over the new WM
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.data import Episode, load_episodes_npz, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.tasks import resolve_task
from jepa_robotics.evaluate import load_jepa_artifact
from scripts.eval_handmanipulate_icem import ICEM

REPO = Path(__file__).resolve().parent.parent


def qgeo(a, b):
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(a @ b))))))


def collect(task, spec, mpc, env, n_eps, lo, hi, dev, rng):
    ag, dgo = spec.obs_dim, spec.obs_dim + spec.goal_dim
    eps, succ = [], 0
    for _ in range(n_eps):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        mpc.reset()
        s = flatten_obs(obs)
        dg_q = torch.as_tensor(s[dgo + 3:dgo + 7] / (np.linalg.norm(s[dgo + 3:dgo + 7]) + 1e-9),
                               dtype=torch.float32, device=dev)
        states, actions = [s.copy()], []
        term = trunc = False; info = {}
        while not (term or trunc):
            a = mpc.act(obs, env, dg_q, lo, hi)
            obs, _, term, trunc, info = env.step(a)
            s = flatten_obs(obs)
            states.append(s.copy()); actions.append(a.copy())
        succ += int(info.get("is_success", 0.0))
        if len(actions) >= 2:
            eps.append(Episode(states=np.asarray(states, np.float32), actions=np.asarray(actions, np.float32)))
    return eps, succ / max(1, n_eps)


def build_icem(model_path, spec, dev, lo, hi, args):
    wm, norm, sp, cfg = load_jepa_artifact(model_path, dev)
    H = min(args.horizon, int(cfg.get("max_horizon", args.horizon)))
    mpc = ICEM(wm, norm, sp, dev, H=H, N=args.candidates, iters=args.iters, elite_frac=0.1,
               init_std=0.5, beta=args.beta, keep_frac=0.3, exec_k=args.exec_k,
               disagree_w=args.disagree_weight, reset_w=args.reset_weight, path_w=0.25,
               fine_deg=args.fine_deg, fine_H=2, fine_N=128)  # accurate short fine gear
    return mpc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--seed-npz", type=Path, required=True)
    p.add_argument("--init-model", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--episodes-per-round", type=int, default=40)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--wm-steps", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--candidates", type=int, default=192)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--beta", type=float, default=2.5)
    p.add_argument("--disagree-weight", type=float, default=1.0)
    p.add_argument("--reset-weight", type=float, default=0.0)
    p.add_argument("--fine-deg", type=float, default=22.0)
    p.add_argument("--seed-cap", type=int, default=600, help="max OU seed episodes kept in the retrain buffer")
    p.add_argument("--onpolicy-repeat", type=int, default=3, help="oversample factor for on-policy episodes")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    args.workdir.mkdir(parents=True, exist_ok=True)
    task = resolve_task(args.task, None)
    env = make_env(task.env_id, seed=1, max_episode_steps=args.max_episode_steps)
    spec = obs_spec_from_env(env)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    ag, goal_dim = spec.obs_dim, spec.goal_dim
    rng = np.random.default_rng(0)

    seed_eps = load_episodes_npz(args.seed_npz)
    onpolicy = []
    model_path = args.init_model
    for r in range(args.rounds):
        mpc = build_icem(model_path, spec, dev, lo, hi, args)
        new_eps, succ = collect(task, spec, mpc, env, args.episodes_per_round, lo, hi, dev, rng)
        onpolicy += new_eps
        # Cap OU seed + oversample on-policy so the planner's own distribution is a
        # real fraction of the retrain data (else 1500 OU eps drown it).
        buffer = seed_eps[: args.seed_cap] + onpolicy * args.onpolicy_repeat
        buf_path = args.workdir / f"buffer_round{r}.npz"
        save_episodes_npz(buf_path, buffer, spec)
        print(json.dumps({"event": "onpolicy_wm_round", "round": r, "task_success": round(succ, 3),
                          "onpolicy_eps": len(onpolicy), "buffer_eps": len(buffer)}), flush=True)
        wm_out = args.workdir / f"wm_round{r}.pt"
        # Warm-start from the CURRENT model (cumulative refinement + consistent
        # normalizer), the fix for the from-scratch loop that degraded to 0.
        subprocess.run([sys.executable, "scripts/train_dexterous_jepa.py", "--task", args.task,
                        "--episodes-npz", str(buf_path), "--out", str(wm_out),
                        "--init-model", str(model_path),
                        "--horizons", "1,2,4,8,16,24,32", "--object-dims", f"{ag},{ag + goal_dim}",
                        "--contact-dims", f"{ag},{ag + goal_dim}", "--lambda-object", "5.0",
                        "--ensemble-heads", "3", "--steps", str(args.wm_steps),
                        "--device", dev.type], check=True, cwd=str(REPO))
        model_path = wm_out
        print(json.dumps({"event": "onpolicy_wm_retrained", "round": r, "wm": str(wm_out)}), flush=True)
    env.close()
    print(json.dumps({"event": "onpolicy_wm_done", "final_wm": str(model_path)}), flush=True)


if __name__ == "__main__":
    main()
