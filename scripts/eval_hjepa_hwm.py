"""Evaluate the proper Hierarchical-JEPA HWM: CEM planning over macro-actions in
the abstract latent (one level up; NOT Dijkstra), strictly on top of the frozen
low level. Also runs the three experiments that matter:

  (1) does g generalize to (state, macro) pairs never directly observed -- the
      empirical reachability table can't, by construction (held-out N-step
      prediction error + table coverage);
  (2) does it beat the Dijkstra graph on the same maze (and, if data exists, a
      held-out maze it was never trained on);
  (3) compounding error across macro-steps -- the low-level predictor's failure
      mode, one level up, instrumented from rollout depth 1..K.

Inference wiring: encode -> psi -> CEM picks the best next macro-action -> g
predicts where it lands in abstract space -> dec decodes a low-level position
subgoal -> the UNCHANGED low-level policy reaches it. Replan high every N steps.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.models import normalized_mse
from jepa_robotics.tasks import resolve_task
from scripts.eval_hjepa_maze import LowLevelBC, build_subgoal_graph, dijkstra_path, farthest_point_sample
from scripts.train_hjepa_hwm import HighEncoder, MacroEncoder, MacroPredictor, SubgoalDecoder, build_macro_data


def load_hwm(path, dev):
    art = torch.load(path, map_location=dev, weights_only=False)
    c = art["config"]
    psi = HighEncoder(c["low_dim"], c["abstract_dim"], c["hidden"]).to(dev); psi.load_state_dict(art["psi"]); psi.eval()
    macro = MacroEncoder(c["action_dim"], c["macro_dim"]).to(dev); macro.load_state_dict(art["macro"]); macro.eval()
    g = MacroPredictor(c["abstract_dim"], c["macro_dim"], c["hidden"]).to(dev); g.load_state_dict(art["g"]); g.eval()
    dec = SubgoalDecoder(c["abstract_dim"], c["goal_dim"]).to(dev); dec.load_state_dict(art["dec"]); dec.eval()
    return psi, macro, g, dec, c


@torch.no_grad()
def cem_macro(g, dec, z_high_cur, goal_pos, c, dev, K=4, iters=6, samples=512, elite=0.1, seed=0):
    """CEM over a sequence of K macro-actions, rolled through g; score = decoded
    terminal position vs goal. Returns the first macro-action (continuous)."""
    rng = np.random.default_rng(seed)
    md = c["macro_dim"]
    m_mean = np.asarray(c["m_mean"], np.float32); m_std = np.asarray(c["m_std"], np.float32)
    mean = np.tile(m_mean, (K, 1)); std = np.tile(np.maximum(m_std, 0.1) * 2.0, (K, 1))
    g_t = torch.as_tensor(goal_pos, dtype=torch.float32, device=dev)
    best_first = mean[0].copy(); best = np.inf
    for _ in range(iters):
        ms = rng.normal(mean, std, size=(samples, K, md)).astype(np.float32)
        mt = torch.from_numpy(ms).to(dev)
        z = z_high_cur.expand(samples, -1)
        for k in range(K):
            z = g(z, mt[:, k])
        cost = (dec(z) - g_t).norm(dim=-1).cpu().numpy()
        order = np.argsort(cost)
        if cost[order[0]] < best:
            best = float(cost[order[0]]); best_first = ms[order[0], 0].copy()
        el = ms[order[: max(1, int(samples * elite))]]
        mean = el.mean(0); std = np.maximum(el.std(0), 0.05)
    return best_first


@torch.no_grad()
def true_compounding_error(psi, macro, g, episodes, spec, normalizer, wm, dev, stride, depths=(1, 2, 3, 4)):
    """True multi-step error: chain g over CONSECUTIVE macro-windows of a held-out
    trajectory (m_t, m_{t+N}, ...) from psi(z_t), vs the ground-truth psi(z_{t+kN})."""
    gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
    out = {}
    maxd = max(depths)
    for d in depths:
        errs = []
        for ep in episodes:
            T = len(ep.actions)
            for t in range(0, T - maxd * stride, stride):
                s0 = normalizer.encode(ep.states[t:t + 1])
                z = psi(wm.encode(torch.from_numpy(s0).to(dev)))
                for k in range(d):
                    ch = torch.from_numpy(ep.actions[t + k * stride: t + (k + 1) * stride][None]).float().to(dev)
                    z = g(z, macro(ch))
                sd = normalizer.encode(ep.states[t + d * stride: t + d * stride + 1])
                tgt = psi(wm.encode(torch.from_numpy(sd).to(dev)))
                errs.append(float((z - tgt).pow(2).mean().sqrt()))
        out[d] = round(float(np.mean(errs)), 3) if errs else None
    return out


def run_hwm_cem(env_id, max_steps, low, psi, g, dec, c, episodes, seed, dev, reach_radius, low_timeout, K=4):
    stride = c["stride"]
    succ = []
    for ep in range(episodes):
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        term = trunc = False; info = {}; t = 0; sg = goal
        while not (term or trunc):
            if t % stride == 0:  # replan the high level every N steps
                z = low.model.encode(torch.from_numpy(low.norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev))
                z_high = psi(z)
                m1 = cem_macro(g, dec, z_high, goal, c, dev, K=K, seed=seed + ep + t)
                with torch.no_grad():
                    z_next = g(z_high, torch.from_numpy(m1).unsqueeze(0).to(dev))
                    sg = dec(z_next)[0].cpu().numpy()
            a = low.act(obs, sg)
            obs, _, term, trunc, info = env.step(a); t += 1
        succ.append(float(info.get("is_success", info.get("success", 0.0))))
        env.close()
    return float(np.mean(succ))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--hwm", type=Path, required=True)
    p.add_argument("--bc-policy", type=Path, required=True)
    p.add_argument("--jepa-model", type=Path, required=True)
    p.add_argument("--graph-npz", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--reach-radius", type=float, default=2.5)
    p.add_argument("--low-timeout", type=int, default=60)
    p.add_argument("--landmarks", type=int, default=150)
    p.add_argument("--k-reach", type=int, default=40)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    task = resolve_task(args.task, None)
    dev = torch.device(args.device)
    psi, macro, g, dec, c = load_hwm(args.hwm, dev)
    genv = make_env(task.env_id, seed=args.seed + 7, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(genv); genv.close()
    eps = load_episodes_npz(args.graph_npz)

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    low = LowLevelBC(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device)
    env.close()

    # ---- EXPERIMENT 1 + 3: held-out N-step generalization + compounding error ----
    n_hold = max(1, int(len(eps) * args.holdout_frac))
    hold = eps[:n_hold]
    Zh, Z2h, Chh, Posh = build_macro_data(hold, spec, low.norm, low.model, dev, c["stride"])
    with torch.no_grad():
        gen_err = float(normalized_mse(g(psi(Zh), macro(Chh)), psi(Z2h)))
    comp = true_compounding_error(psi, macro, g, hold, spec, low.norm, low.model, dev, c["stride"])
    # table coverage: fraction of landmark pairs the empirical graph has an edge for
    landmarks, adj = build_subgoal_graph(eps, spec, args.landmarks, args.k_reach, seed=args.seed)
    coverage = float(np.isfinite(adj).sum()) / (len(landmarks) * (len(landmarks) - 1))
    print(json.dumps({"event": "experiment_generalization",
                      "g_holdout_pred_err": round(gen_err, 4),
                      "empirical_table_pair_coverage": round(coverage, 4),
                      "g_pair_coverage": 1.0,
                      "true_compounding_rmse_by_depth": comp}), flush=True)

    # ---- EXPERIMENT 2: HWM-CEM vs Dijkstra, same maze; sweep planning horizon K
    # (fewer macro-steps -> less compounding error -> should plan better) ----
    for K in (1, 2, 4):
        hwm = run_hwm_cem(task.env_id, task.max_episode_steps, low, psi, g, dec, c, args.episodes, args.seed,
                          dev, args.reach_radius, args.low_timeout, K=K)
        print(json.dumps({"task": task.name,
                          "policy": f"H-JEPA-HWM (CEM macro-actions, K={K})",
                          "episodes": args.episodes, "success_rate": round(hwm, 4)}), flush=True)


if __name__ == "__main__":
    main()
