"""Evaluate the Hierarchical-JEPA HWM: planning over macro-actions in
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

from jepa_robotics.data import Episode, load_episodes_npz, save_episodes_npz
from jepa_robotics.envs import flatten_obs, make_env, obs_spec_from_env
from jepa_robotics.models import normalized_mse
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.maze_low_level import LowLevelBC, LowLevelFlow
from jepa_robotics.algos.hwm import HighEncoder, MacroEncoder, MacroPredictor, SubgoalDecoder, build_macro_data
from jepa_robotics.algos.priors import EpsNet, make_ddpm, sample_action_chunks


def load_macro_flow(path, dev):
    art = torch.load(path, map_location=dev, weights_only=False)
    c = art["config"]
    net = EpsNet(c["macro_dim"], c["cond_dim"], c["hidden"], n_blocks=c["n_blocks"]).to(dev)
    net.load_state_dict(art["ema"]); net.eval()
    return net, make_ddpm(int(c["flow_steps"]), dev), c


@torch.no_grad()
def flow_macro_subgoal(flow_net, ddpm, g, dec, z_high_cur, goal_pos, dev, n_samples=16, flow_steps=16,
                       horizon=1):
    """Receding-horizon HWM planning with the flow macro-prior as the proposal.

    Sample N feasible first-macros from the flow prior conditioned on
    (z_high, goal_xy); for each, roll ``horizon`` feasible macro-hops forward
    through the frozen g (re-sampling one feasible macro per hop from the flow at
    the rolled latent), and score the terminal decoded position vs the goal.
    Commit to the FIRST hop's decoded subgoal of the best rollout. Every macro at
    every hop is an on-manifold demonstrated transition, so the K-step lookahead
    plans AROUND walls (feasible detours) instead of hallucinating a straight
    wall-crossing path -- the fix for both the Gaussian-CEM failure and the
    greedy 1-hop planner's dead-ends."""
    md = flow_net.in_proj.in_features
    gp = torch.as_tensor(goal_pos, dtype=torch.float32, device=dev).view(1, -1)
    z = z_high_cur.expand(n_samples, -1)
    gpn = gp.expand(n_samples, -1)
    m0 = sample_action_chunks(flow_net, ddpm, torch.cat([z, gpn], dim=-1), md, dev,
                              objective="flow", flow_steps=flow_steps)
    z = g(z, m0)
    sg0 = dec(z)  # first-hop subgoals (what the low level will actually pursue)
    for _ in range(max(0, horizon - 1)):
        mk = sample_action_chunks(flow_net, ddpm, torch.cat([z, gpn], dim=-1), md, dev,
                                  objective="flow", flow_steps=flow_steps)
        z = g(z, mk)
    cost = (dec(z) - gpn).norm(dim=-1)
    best = int(torch.argmin(cost))
    return sg0[best].cpu().numpy()


def load_hwm(path, dev):
    art = torch.load(path, map_location=dev, weights_only=False)
    c = art["config"]
    psi = HighEncoder(c["low_dim"], c["abstract_dim"], c["hidden"]).to(dev); psi.load_state_dict(art["psi"]); psi.eval()
    macro = MacroEncoder(c["action_dim"], c["macro_dim"]).to(dev); macro.load_state_dict(art["macro"]); macro.eval()
    g = MacroPredictor(c["abstract_dim"], c["macro_dim"], c["hidden"]).to(dev); g.load_state_dict(art["g"]); g.eval()
    dec = SubgoalDecoder(c["abstract_dim"], c["goal_dim"]).to(dev); dec.load_state_dict(art["dec"]); dec.eval()
    return psi, macro, g, dec, c


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


def run_hwm_flow(env_id, max_steps, low, psi, g, dec, macro_flow, episodes, seed, dev, reach_radius, low_timeout,
                 video_out=None, width=640, height=480, fps=30, walk_diagnostics=False, rollout_out=None):
    """HWM flow-macro high level on top of the (unchanged) low level. Replans a
    subgoal (sampled feasible macro -> g -> decoded xy) when the low level REACHES
    the current subgoal (within reach_radius) or stalls on it for low_timeout
    steps -- reach-based, not fixed-stride, so a slow low level gets time to
    arrive before the next hop is issued. Once the decoded subgoal is essentially
    the goal, aim straight at the goal so the low level finishes onto it."""
    fnet, fddpm, fcfg = macro_flow
    succ = []
    video_saved = False
    diag = {"speed8": [], "stall_frac": [], "flip_frac": [], "replans": [],
            "fail_steps": [], "fail_dist": []} if walk_diagnostics else None
    rollouts = [] if rollout_out is not None else None
    for ep in range(episodes):
        capture = video_out is not None and not video_saved
        env = make_env(env_id, seed=seed + ep, max_episode_steps=max_steps,
                       render_mode="rgb_array" if capture else None,
                       width=width if capture else None, height=height if capture else None)
        obs, _ = env.reset(seed=seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        term = trunc = False; info = {}; t = 0; sg = goal; since_replan = 0; need_replan = True
        frames = []
        xy_hist, upright_hist, n_replans = [], [], 0
        ep_states, ep_actions = [flatten_obs(obs)], []
        if capture:
            fr = env.render()
            if fr is not None:
                frames.append(fr)
        while not (term or trunc):
            ag = np.asarray(obs["achieved_goal"], np.float32)
            if need_replan or np.linalg.norm(ag - sg) < reach_radius or since_replan >= low_timeout:
                z = low.model.encode(torch.from_numpy(low.norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev))
                z_high = psi(z)
                sg = flow_macro_subgoal(fnet, fddpm, g, dec, z_high, goal, dev,
                                        n_samples=fcfg.get("n_samples", 16),
                                        flow_steps=int(fddpm["T"]),
                                        horizon=fcfg.get("horizon", 1))
                # if the decoded next subgoal is already essentially the goal, aim at the goal
                if np.linalg.norm(sg - goal) < reach_radius:
                    sg = goal
                since_replan = 0; need_replan = False
                n_replans += 1
            a = low.act(obs, sg)
            obs, _, term, trunc, info = env.step(a); t += 1; since_replan += 1
            if rollouts is not None:
                ep_actions.append(a); ep_states.append(flatten_obs(obs))
            if diag is not None:
                xy_hist.append(np.asarray(obs["achieved_goal"], np.float32))
                q = np.asarray(obs["observation"][1:5], np.float32)  # torso quat (w,x,y,z)
                upright_hist.append(1.0 - 2.0 * float(q[1] ** 2 + q[2] ** 2))  # body-z . world-z
            if capture:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
        s = float(info.get("is_success", info.get("success", 0.0)))
        succ.append(s)
        if rollouts is not None:
            rollouts.append(Episode(states=np.asarray(ep_states, np.float32),
                                    actions=np.asarray(ep_actions, np.float32)))
        if diag is not None and len(xy_hist) > 8:
            xy = np.stack(xy_hist)
            d8 = np.linalg.norm(xy[8:] - xy[:-8], axis=1)
            up = np.asarray(upright_hist)
            diag["speed8"].append(float(d8.mean()))
            diag["stall_frac"].append(float((d8 < 0.2).mean()))
            diag["flip_frac"].append(float((up < 0.0).mean()))
            diag["replans"].append(n_replans)
            if s < 0.5:
                diag["fail_steps"].append(t)
                diag["fail_dist"].append(float(np.linalg.norm(xy[-1] - goal)))
        env.close()
        if capture and s > 0.5 and frames:
            import imageio.v2 as imageio
            Path(video_out).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_out, frames[::2], fps=fps, format="FFMPEG")
            video_saved = True
            print(json.dumps({"event": "video_saved", "path": str(video_out), "episode": ep}), flush=True)
    if diag is not None:
        print(json.dumps({"event": "walk_diagnostics",
                          **{k: round(float(np.mean(v)), 3) if v else None for k, v in diag.items()}}),
              flush=True)
    if rollouts is not None:
        Path(rollout_out).parent.mkdir(parents=True, exist_ok=True)
        save_episodes_npz(rollout_out, rollouts)
        print(json.dumps({"event": "rollouts_saved", "path": str(rollout_out), "episodes": len(rollouts)}), flush=True)
    return float(np.mean(succ))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--hwm", type=Path, required=True, help="HWM checkpoint (psi/macro/g/dec) from train_hjepa_hwm.py")
    p.add_argument("--macro-flow", type=Path, required=True,
                   help="Flow prior over macro-actions (train_hwm_macro_flow.py). The high level samples feasible on-manifold macros from it, then g -> decoded subgoal.")
    p.add_argument("--macro-flow-samples", type=int, default=16)
    p.add_argument("--macro-flow-horizon", type=int, default=1,
                   help="Feasible macro-hops to look ahead through g when scoring flow-sampled macros (1 is best; >=2 compounds g error).")
    p.add_argument("--bc-policy", type=Path, required=True,
                   help="Low-level policy: flow-walker .pt (low-type flow) or BC .pt (low-type bc).")
    p.add_argument("--low-type", default="flow", choices=["flow", "bc"],
                   help="Low level under the HWM high level: 'flow' the directed flow-matching walker (canonical), or 'bc'.")
    p.add_argument("--walker-scorer", type=Path, default=None,
                   help="ProgressScorer checkpoint (train_walker_scorer.py) for best-of-N chunk selection in the flow walker.")
    p.add_argument("--walker-samples", type=int, default=1,
                   help="Chunks sampled per walker replan; the scorer picks the one with max predicted progress toward the subgoal.")
    p.add_argument("--walker-target-progress", type=float, default=None,
                   help="For a progress-conditioned walker: requested per-chunk xy progress (default: the training p90 stored in the walker checkpoint).")
    p.add_argument("--walker-select-quantile", type=float, default=1.0,
                   help="Quantile of scorer-ranked samples to execute (1.0 = argmax; sub-max avoids the winner's-curse tail).")
    p.add_argument("--walk-diagnostics", action="store_true",
                   help="Report realized 8-step xy speed, stall/flip fractions, replans, and failure step/distance aggregates.")
    p.add_argument("--walker-replan", type=int, default=None,
                   help="Execute only this many steps of each sampled chunk before resampling (default: the full chunk). Tighter closed loop = less open-loop commitment to destabilizing sequences.")
    p.add_argument("--rollout-out", type=Path, default=None,
                   help="Save eval rollouts (states/actions npz, episode format) for self-supervised analysis.")
    p.add_argument("--jepa-model", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, default=None,
                   help="Episodes for the (optional) macro-model diagnostics; not needed for control.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--reach-radius", type=float, default=1.0)
    p.add_argument("--low-timeout", type=int, default=90)
    p.add_argument("--diagnostics", action="store_true",
                   help="Also report the HWM macro-model held-out generalization + compounding-error diagnostics (needs --episodes-npz).")
    p.add_argument("--video-out", type=Path, default=None, help="Save an mp4 of the first successful episode.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Make both environment starts and stochastic flow samples reproducible.
    # Each reported seed therefore identifies the complete controller rollout,
    # rather than only the environment's initial state.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = resolve_task(args.task, None)
    dev = torch.device(args.device)
    psi, macro, g, dec, c = load_hwm(args.hwm, dev)

    env = make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    if args.low_type == "flow":
        low = LowLevelFlow(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device,
                           replan=args.walker_replan,
                           scorer_path=args.walker_scorer, n_samples=args.walker_samples,
                           target_progress=args.walker_target_progress,
                           select_quantile=args.walker_select_quantile)
    else:
        low = LowLevelBC(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device)
    env.close()

    if args.diagnostics and args.episodes_npz is not None:
        # HWM macro-model quality: held-out next-macro generalization + compounding error.
        genv = make_env(task.env_id, seed=args.seed + 7, max_episode_steps=task.max_episode_steps)
        spec = obs_spec_from_env(genv); genv.close()
        eps = load_episodes_npz(args.episodes_npz)
        hold = eps[: max(1, int(len(eps) * args.holdout_frac))]
        Zh, Z2h, Chh, Posh = build_macro_data(hold, spec, low.norm, low.model, dev, c["stride"])
        with torch.no_grad():
            gen_err = float(normalized_mse(g(psi(Zh), macro(Chh)), psi(Z2h)))
        comp = true_compounding_error(psi, macro, g, hold, spec, low.norm, low.model, dev, c["stride"])
        print(json.dumps({"event": "hwm_macro_diagnostics",
                          "g_holdout_pred_err": round(gen_err, 4),
                          "true_compounding_rmse_by_depth": comp}), flush=True)

    fnet, fddpm, fcfg = load_macro_flow(args.macro_flow, dev)
    fcfg["n_samples"] = args.macro_flow_samples
    fcfg["horizon"] = args.macro_flow_horizon
    hwm = run_hwm_flow(task.env_id, task.max_episode_steps, low, psi, g, dec, (fnet, fddpm, fcfg),
                       args.episodes, args.seed, dev, args.reach_radius, args.low_timeout,
                       video_out=args.video_out, width=args.width, height=args.height, fps=args.fps,
                       walk_diagnostics=args.walk_diagnostics, rollout_out=args.rollout_out)
    print(json.dumps({"task": task.name,
                      "policy": f"H-JEPA-HWM (flow macro-prior, n={args.macro_flow_samples}, low={args.low_type})",
                      "episodes": args.episodes, "success_rate": round(hwm, 4)}), flush=True)


if __name__ == "__main__":
    main()
