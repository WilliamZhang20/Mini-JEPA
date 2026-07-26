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
from collections import deque
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
from jepa_robotics.algos.maze_low_level import (
    LowLevelActionMemory,
    LowLevelBC,
    LowLevelChunkBC,
    LowLevelFlow,
)
from jepa_robotics.algos.hwm import (
    DiscreteTopologyRouter,
    GoalConditionedWaypointMemory,
    HighEncoder,
    MacroEncoder,
    MacroPredictor,
    SubgoalDecoder,
    CoordinateTopologyScorer,
    TopologyScorer,
    build_macro_data,
)
from jepa_robotics.algos.priors import EpsNet, make_ddpm, sample_action_chunks


def make_minari_eval_env(
    dataset_id: str,
    render_mode: str | None = None,
    width: int | None = None,
    height: int | None = None,
):
    """Recover the benchmark's fixed-start/fixed-goal evaluation environment.

    Minari's AntMaze evaluation specs keep the underlying environment
    continuing so its sparse reward can accumulate. Published AntMaze scores
    are conventionally success-within-horizon, so stop at the first success
    while preserving the official reset/goal map and episode limit.
    """
    import gymnasium as gym
    import minari

    dataset = minari.load_dataset(dataset_id, download=False)
    if render_mode is None:
        env = dataset.recover_environment(eval_env=True)
    else:
        render_kwargs = {"render_mode": render_mode}
        if width is not None:
            render_kwargs["width"] = width
        if height is not None:
            render_kwargs["height"] = height
        eval_spec = getattr(dataset, "eval_env_spec", dataset._eval_env_spec)
        env = eval_spec.make(**render_kwargs)

    class _TerminateOnSuccess(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            success = float(info.get("is_success", info.get("success", 0.0)))
            if success > 0.5:
                terminated = True
            return obs, reward, terminated, truncated, info

    return _TerminateOnSuccess(env)


def load_macro_flow(path, dev):
    art = torch.load(path, map_location=dev, weights_only=False)
    c = art["config"]
    net = EpsNet(c["macro_dim"], c["cond_dim"], c["hidden"], n_blocks=c["n_blocks"]).to(dev)
    net.load_state_dict(art["ema"]); net.eval()
    return net, make_ddpm(int(c["flow_steps"]), dev), c


def load_topology_scorer(path, dev):
    artifact = torch.load(path, map_location=dev, weights_only=False)
    config = artifact["config"]
    scorer = (
        CoordinateTopologyScorer(
            int(config["goal_dim"]),
            int(config["hidden"]),
        )
        if config.get("coordinate_only", False)
        else TopologyScorer(
            int(config["abstract_dim"]),
            int(config["goal_dim"]),
            int(config["hidden"]),
        )
    ).to(dev)
    scorer.load_state_dict(artifact["state_dict"])
    scorer.eval()
    return scorer, config


def load_waypoint_flow(path, dev):
    artifact = torch.load(path, map_location=dev, weights_only=False)
    config = artifact["config"]
    net = EpsNet(
        int(config["waypoint_dim"]),
        int(config["cond_dim"]),
        int(config["hidden"]),
        n_blocks=int(config["n_blocks"]),
    ).to(dev)
    net.load_state_dict(artifact["ema"])
    net.eval()
    return net, make_ddpm(int(config["flow_steps"]), dev), config


def load_waypoint_memory(path, current_weight, goal_weight, neighbors):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return (
        GoalConditionedWaypointMemory(
            artifact["current"],
            artifact["goals"],
            artifact["waypoints"],
            current_weight=current_weight,
            goal_weight=goal_weight,
            neighbors=neighbors,
        ),
        artifact["config"],
    )


def load_discrete_topology_router(path, dev):
    artifact = torch.load(path, map_location=dev, weights_only=False)
    config = artifact["config"]
    router = DiscreteTopologyRouter(
        int(config["region_count"]), int(config["hidden"])
    ).to(dev)
    router.load_state_dict(artifact["state_dict"])
    router.eval()
    return router, np.asarray(artifact["centers"], dtype=np.float32), config


@torch.no_grad()
def learned_topology_waypoint(router_artifact, current_pos, goal_pos, dev):
    router, centers, _ = router_artifact
    current = torch.as_tensor(current_pos, dtype=torch.float32, device=dev)[None]
    goal = torch.as_tensor(goal_pos, dtype=torch.float32, device=dev)[None]
    region = int(router(current, goal).argmax(dim=-1))
    waypoint = centers[region].copy()
    goal_region = int(np.linalg.norm(centers - np.asarray(goal_pos), axis=1).argmin())
    if region == goal_region:
        waypoint = np.asarray(goal_pos, dtype=np.float32)
    return waypoint


def maze_map_route(env, start_pos, goal_pos):
    """Diagnostic shortest cell route from the environment's exposed maze map."""
    maze = env.unwrapped.maze
    start = tuple(maze.cell_xy_to_rowcol(np.asarray(start_pos)))
    goal = tuple(maze.cell_xy_to_rowcol(np.asarray(goal_pos)))
    queue = deque([start])
    parent = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        row, col = cell
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            rr, cc = neighbor
            if (
                0 <= rr < maze.map_length
                and 0 <= cc < maze.map_width
                and maze.maze_map[rr][cc] != 1
                and neighbor not in parent
            ):
                parent[neighbor] = cell
                queue.append(neighbor)
    if goal not in parent:
        raise RuntimeError(f"No maze-cell route from {start} to {goal}")
    cells = []
    cell = goal
    while cell is not None:
        cells.append(cell)
        cell = parent[cell]
    cells.reverse()
    waypoints = [
        np.asarray(maze.cell_rowcol_to_xy(cell), dtype=np.float32)
        for cell in cells[1:]
    ]
    if waypoints:
        waypoints[-1] = np.asarray(goal_pos, dtype=np.float32)
    return waypoints


@torch.no_grad()
def flow_waypoint_subgoal(
    waypoint_flow,
    z_high_cur,
    goal_pos,
    dev,
    n_samples=32,
    topology_scorer=None,
    selector="topology",
):
    net, ddpm, config = waypoint_flow
    goal = torch.as_tensor(goal_pos, dtype=torch.float32, device=dev).view(1, -1)
    condition = torch.cat([z_high_cur, goal], dim=-1).expand(n_samples, -1)
    waypoints = sample_action_chunks(
        net,
        ddpm,
        condition,
        int(config["waypoint_dim"]),
        dev,
        objective="flow",
        flow_steps=int(ddpm["T"]),
    )
    goal_batch = goal.expand(n_samples, -1)
    if selector == "median":
        return waypoints.median(dim=0).values.cpu().numpy()
    if selector == "mean":
        return waypoints.mean(dim=0).cpu().numpy()
    if topology_scorer is None:
        cost = (waypoints - goal_batch).norm(dim=-1)
    else:
        scorer, _ = topology_scorer
        cost = scorer(
            z_high_cur.expand(n_samples, -1),
            waypoints,
            goal_batch,
        )
    return waypoints[int(torch.argmin(cost))].cpu().numpy()


@torch.no_grad()
def flow_macro_subgoal(flow_net, ddpm, g, dec, z_high_cur, goal_pos, dev, n_samples=16, flow_steps=16,
                       horizon=1, topology_scorer=None):
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
    terminal_pos = dec(z)
    if topology_scorer is None:
        cost = (terminal_pos - gpn).norm(dim=-1)
    else:
        scorer, _ = topology_scorer
        cost = scorer(z, terminal_pos, gpn)
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
                 video_out=None, video_episodes=1, width=640, height=480, fps=30,
                 overview_camera=False, walk_diagnostics=False, rollout_out=None,
                 minari_eval_dataset=None, topology_scorer=None,
                 log_high_level=False, waypoint_flow=None, waypoint_memory=None,
                 direct_goal=False, maze_map_router=False,
                 learned_router=None):
    """HWM flow-macro high level on top of the (unchanged) low level. Replans a
    subgoal (sampled feasible macro -> g -> decoded xy) when the low level REACHES
    the current subgoal (within reach_radius) or stalls on it for low_timeout
    steps -- reach-based, not fixed-stride, so a slow low level gets time to
    arrive before the next hop is issued. Once the decoded subgoal is essentially
    the goal, aim straight at the goal so the low level finishes onto it."""
    fnet, fddpm, fcfg = macro_flow
    succ = []
    video_frames = []
    video_env = None
    overview_configured = False
    diag = {"speed8": [], "stall_frac": [], "flip_frac": [], "replans": [],
            "fail_steps": [], "fail_dist": []} if walk_diagnostics else None
    rollouts = [] if rollout_out is not None else None
    for ep in range(episodes):
        capture = video_out is not None and ep < video_episodes
        if capture:
            if video_env is None:
                video_env = (
                    make_minari_eval_env(
                        minari_eval_dataset,
                        render_mode="rgb_array",
                        width=width,
                        height=height,
                    )
                    if minari_eval_dataset is not None
                    else make_env(
                        env_id, seed=seed + ep, max_episode_steps=max_steps,
                        render_mode="rgb_array", width=width, height=height,
                    )
                )
            env = video_env
        else:
            env = (
                make_minari_eval_env(minari_eval_dataset)
                if minari_eval_dataset is not None else
                make_env(env_id, seed=seed + ep, max_episode_steps=max_steps)
            )
        obs, _ = env.reset(seed=seed + ep)
        goal = np.asarray(obs["desired_goal"], np.float32)
        route = (
            maze_map_route(env, obs["achieved_goal"], goal)
            if maze_map_router else None
        )
        route_index = 0
        term = trunc = False; info = {}; t = 0; sg = goal; since_replan = 0; need_replan = True
        frames = []
        xy_hist, upright_hist, n_replans = [], [], 0
        ep_states, ep_actions = [flatten_obs(obs)], []
        if capture:
            # The first render initializes Gymnasium's lazy off-screen viewer.
            # For AntMaze showcases, switch that viewer from the robot-following
            # XML camera to a stable free camera framing the complete maze.
            env.render()
            if overview_camera and not overview_configured:
                maze_env = env.unwrapped
                renderer = maze_env.ant_env.mujoco_renderer
                renderer.camera_id = -1
                camera = renderer._viewers["rgb_array"].cam
                camera.lookat[:] = [0.0, 0.0, 1.0]
                camera.azimuth = 90.0
                camera.elevation = -70.0
                camera.distance = 5.0 * max(
                    maze_env.maze.map_length, maze_env.maze.map_width
                )
                overview_configured = True
            fr = env.render()
            if fr is not None:
                frames.append(fr)
        while not (term or trunc):
            ag = np.asarray(obs["achieved_goal"], np.float32)
            if need_replan or np.linalg.norm(ag - sg) < reach_radius or since_replan >= low_timeout:
                if route is not None:
                    if (
                        not need_replan
                        and route_index < len(route) - 1
                        and np.linalg.norm(ag - sg) < reach_radius
                    ):
                        route_index += 1
                    sg = route[route_index]
                elif learned_router is not None:
                    sg = learned_topology_waypoint(
                        learned_router, ag, goal, dev
                    )
                elif direct_goal:
                    sg = goal
                else:
                    z = low.model.encode(torch.from_numpy(low.norm.encode(flatten_obs(obs))).unsqueeze(0).to(dev))
                    z_high = psi(z)
                if route is None and learned_router is None and not direct_goal and waypoint_memory is not None:
                    memory, _ = waypoint_memory
                    sg = memory.query(ag, goal)
                elif route is None and learned_router is None and not direct_goal and waypoint_flow is None:
                    sg = flow_macro_subgoal(
                        fnet, fddpm, g, dec, z_high, goal, dev,
                        n_samples=fcfg.get("n_samples", 16),
                        flow_steps=int(fddpm["T"]),
                        horizon=fcfg.get("horizon", 1),
                        topology_scorer=topology_scorer,
                    )
                elif route is None and learned_router is None and not direct_goal:
                    sg = flow_waypoint_subgoal(
                        waypoint_flow,
                        z_high,
                        goal,
                        dev,
                        n_samples=fcfg.get("n_samples", 16),
                        topology_scorer=topology_scorer,
                        selector=fcfg.get("waypoint_selector", "topology"),
                    )
                # if the decoded next subgoal is already essentially the goal, aim at the goal
                if np.linalg.norm(sg - goal) < reach_radius:
                    sg = goal
                since_replan = 0; need_replan = False
                n_replans += 1
                if log_high_level and ep == 0:
                    print(
                        json.dumps(
                            {
                                "event": "high_level_replan",
                                "step": t,
                                "achieved": ag.tolist(),
                                "subgoal": np.asarray(sg).tolist(),
                                "goal": goal.tolist(),
                            }
                        ),
                        flush=True,
                    )
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
        if not capture:
            env.close()
        if capture:
            video_frames.extend(frames[::2])
    if video_env is not None:
        video_env.close()
    if video_out is not None and video_frames:
        import imageio.v2 as imageio
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_out, video_frames, fps=fps, format="FFMPEG")
        print(json.dumps({"event": "video_saved", "path": str(video_out),
                          "episodes": min(video_episodes, episodes),
                          "overview_camera": overview_camera}), flush=True)
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
    p.add_argument("--low-type", default="flow", choices=["flow", "bc", "memory", "chunk_bc"],
                   help="Low level under the HWM high level: flow, BC, memory, or deterministic chunk BC.")
    p.add_argument("--action-memory", type=Path, default=None)
    p.add_argument("--action-memory-neighbors", type=int, default=3)
    p.add_argument("--chunk-bc", type=Path, default=None)
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
    p.add_argument(
        "--log-high-level",
        action="store_true",
        help="Log the first episode's achieved position and selected subgoal at every high-level replan.",
    )
    p.add_argument("--walker-replan", type=int, default=None,
                   help="Execute only this many steps of each sampled chunk before resampling (default: the full chunk). Tighter closed loop = less open-loop commitment to destabilizing sequences.")
    p.add_argument("--rollout-out", type=Path, default=None,
                   help="Save eval rollouts (states/actions npz, episode format) for self-supervised analysis.")
    p.add_argument("--jepa-model", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, default=None,
                   help="Episodes for the (optional) macro-model diagnostics; not needed for control.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--minari-eval-dataset",
        default=None,
        help="Recover this Minari dataset's official fixed-start/fixed-goal eval env.",
    )
    p.add_argument(
        "--topology-scorer",
        type=Path,
        default=None,
        help="Self-supervised temporal-distance scorer for topology-aware macro ranking.",
    )
    p.add_argument(
        "--waypoint-flow",
        type=Path,
        default=None,
        help="Direct demonstrated-waypoint flow; bypasses macro latent endpoint prediction.",
    )
    p.add_argument(
        "--waypoint-memory",
        type=Path,
        default=None,
        help="Goal-conditioned retrieval memory of demonstrated local waypoints.",
    )
    p.add_argument("--memory-current-weight", type=float, default=1.0)
    p.add_argument("--memory-goal-weight", type=float, default=4.0)
    p.add_argument("--memory-neighbors", type=int, default=7)
    p.add_argument(
        "--waypoint-selector",
        choices=["topology", "median", "mean"],
        default="topology",
        help="How to collapse direct waypoint-flow samples.",
    )
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--reach-radius", type=float, default=1.0)
    p.add_argument("--low-timeout", type=int, default=90)
    p.add_argument(
        "--direct-goal",
        action="store_true",
        help="Bypass high-level subgoals and hold the environment's final goal.",
    )
    p.add_argument(
        "--maze-map-router",
        action="store_true",
        help="Diagnostic oracle: shortest free-cell route from the exposed maze map.",
    )
    p.add_argument(
        "--learned-router",
        type=Path,
        default=None,
        help="Discrete learned next-region router; does not query the maze map at inference.",
    )
    p.add_argument("--diagnostics", action="store_true",
                   help="Also report the HWM macro-model held-out generalization + compounding-error diagnostics (needs --episodes-npz).")
    p.add_argument("--video-out", type=Path, default=None,
                   help="Save an mp4 of evaluated episodes.")
    p.add_argument("--video-episodes", type=int, default=1,
                   help="Number of consecutive episodes to concatenate into the video.")
    p.add_argument("--overview-camera", action="store_true",
                   help="Use a fixed elevated camera framing the complete AntMaze.")
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

    env = (
        make_minari_eval_env(args.minari_eval_dataset)
        if args.minari_eval_dataset is not None else
        make_env(task.env_id, seed=args.seed, max_episode_steps=task.max_episode_steps)
    )
    if args.low_type == "flow":
        low = LowLevelFlow(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device,
                           replan=args.walker_replan,
                           scorer_path=args.walker_scorer, n_samples=args.walker_samples,
                           target_progress=args.walker_target_progress,
                           select_quantile=args.walker_select_quantile)
    elif args.low_type == "bc":
        low = LowLevelBC(args.jepa_model, args.bc_policy, env.action_space.low, env.action_space.high, device=args.device)
    elif args.low_type == "memory":
        if args.action_memory is None:
            p.error("--low-type memory requires --action-memory")
        low = LowLevelActionMemory(
            args.jepa_model,
            args.action_memory,
            env.action_space.low,
            env.action_space.high,
            device=args.device,
            neighbors=args.action_memory_neighbors,
            replan=args.walker_replan,
        )
    else:
        if args.chunk_bc is None:
            p.error("--low-type chunk_bc requires --chunk-bc")
        low = LowLevelChunkBC(
            args.jepa_model,
            args.chunk_bc,
            env.action_space.low,
            env.action_space.high,
            device=args.device,
            replan=args.walker_replan,
        )
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
    topology_scorer = (
        load_topology_scorer(args.topology_scorer, dev)
        if args.topology_scorer is not None else None
    )
    waypoint_flow = (
        load_waypoint_flow(args.waypoint_flow, dev)
        if args.waypoint_flow is not None else None
    )
    waypoint_memory = (
        load_waypoint_memory(
            args.waypoint_memory,
            args.memory_current_weight,
            args.memory_goal_weight,
            args.memory_neighbors,
        )
        if args.waypoint_memory is not None else None
    )
    learned_router = (
        load_discrete_topology_router(args.learned_router, dev)
        if args.learned_router is not None else None
    )
    fcfg["n_samples"] = args.macro_flow_samples
    fcfg["horizon"] = args.macro_flow_horizon
    fcfg["waypoint_selector"] = args.waypoint_selector
    hwm = run_hwm_flow(task.env_id, task.max_episode_steps, low, psi, g, dec, (fnet, fddpm, fcfg),
                       args.episodes, args.seed, dev, args.reach_radius, args.low_timeout,
                       video_out=args.video_out, video_episodes=args.video_episodes,
                       width=args.width, height=args.height, fps=args.fps,
                       overview_camera=args.overview_camera,
                       walk_diagnostics=args.walk_diagnostics, rollout_out=args.rollout_out,
                       minari_eval_dataset=args.minari_eval_dataset,
                       topology_scorer=topology_scorer,
                       log_high_level=args.log_high_level,
                       waypoint_flow=waypoint_flow,
                       waypoint_memory=waypoint_memory,
                       direct_goal=args.direct_goal,
                       maze_map_router=args.maze_map_router,
                       learned_router=learned_router)
    print(json.dumps({"task": task.name,
                      "policy": f"H-JEPA-HWM (flow macro-prior, n={args.macro_flow_samples}, low={args.low_type})",
                      "episodes": args.episodes, "success_rate": round(hwm, 4),
                      "minari_eval_dataset": args.minari_eval_dataset,
                      "topology_scorer": None if args.topology_scorer is None else str(args.topology_scorer),
                      "waypoint_flow": None if args.waypoint_flow is None else str(args.waypoint_flow),
                      "waypoint_memory": None if args.waypoint_memory is None else str(args.waypoint_memory),
                      "direct_goal": args.direct_goal,
                      "maze_map_router": args.maze_map_router,
                      "learned_router": None if args.learned_router is None else str(args.learned_router)}),
          flush=True)


if __name__ == "__main__":
    main()
