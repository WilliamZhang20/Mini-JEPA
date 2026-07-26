"""Train the action-chunked rectified-flow walker for AntMaze — the kitchen
recipe applied to locomotion, and the real shot at a robust ant gait.

Why flow, not BC: with future-goal relabeling, one state maps to MANY action chunks
relabeled goal/direction), so behaviour cloning *averages* them into mush (the
single-step and chunked-BC walkers both collapsed). A flow/diffusion policy
instead learns the conditional *distribution* over chunks and *samples* a single
coherent gait toward the goal — multimodality preserved, gait phase consistent.

  * conditioning c = JEPA latent of the future-goal-relabeled observation,
    concatenated with the raw normalized obs (`--concat-raw`, the kitchen fix for
    control precision -- the predictive latent alone sheds proprioceptive detail).
  * target x = the next-H action chunk (flattened).
  * rectified flow: x_t = (1-t) x0 + t x1, x0~N(0,I), x1=chunk; the velocity field
    v(x_t, t, c) regresses x1 - x0. Sampling = Euler-integrate the ODE noise->chunk.

Executed receding-horizon by eval_hjepa_hwm.py (flow-macro HWM) low level.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.data import load_episodes_npz
from jepa_robotics.envs import make_env, obs_spec_from_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.tasks import resolve_task
from jepa_robotics.algos.control.flow import build_flow_net
from jepa_robotics.algos.task_families.maze import (
    build_auxiliary_chunks,
    build_chunks,
    build_directed_chunks,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes-npz", type=Path, required=True)
    p.add_argument(
        "--auxiliary-episodes-npz",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Optional state-dependent behavior banks mixed into the same flow. "
            "Their chunks are paired with random navigation goals so the model "
            "learns when they apply from state rather than a runtime mode switch."
        ),
    )
    p.add_argument(
        "--auxiliary-fraction",
        type=float,
        default=0.25,
        help="Fraction of each batch drawn from auxiliary behavior banks.",
    )
    p.add_argument("--max-auxiliary-episodes", type=int, default=1200)
    p.add_argument(
        "--auxiliary-goal-copies",
        type=int,
        default=2,
        help="Random navigation-goal relabels per auxiliary chunk.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional compatible flow checkpoint to continue training from.",
    )
    p.add_argument("--max-episodes", type=int, default=1200)
    p.add_argument(
        "--route-start",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional offline specialization: retain episodes starting near this xy.",
    )
    p.add_argument(
        "--route-goal",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional offline specialization: retain episodes targeting near this xy.",
    )
    p.add_argument("--route-radius", type=float, default=1.5)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument(
        "--flow-architecture",
        choices=["mlp", "residual"],
        default="mlp",
        help="residual injects the state condition at every FiLM block.",
    )
    p.add_argument("--flow-blocks", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--future-goal-frac", type=float, default=0.85)
    p.add_argument(
        "--successful-prefix-final-goal",
        action="store_true",
        help="Use only successful pre-goal chunks, conditioned on the episode final goal.",
    )
    p.add_argument("--directed", action="store_true",
                   help="Use near-future relabeling plus a progress filter for a faster, decisive gait.")
    p.add_argument("--max-relabel-h", type=int, default=60,
                   help="[directed] relabel the goal to a future achieved position within this many steps.")
    p.add_argument("--min-progress", type=float, default=0.15,
                   help="[directed] keep a sample only if the ant closes at least this much xy distance to the relabeled goal over the chunk.")
    p.add_argument("--speed-weight", type=float, default=0.0,
                   help="[directed] bias training sampling toward faster segments: p ~ progress^speed_weight. Keeps slow turning segments (unlike a hard filter) but speeds up the gait. 0 = uniform.")
    p.add_argument("--progress-cond", action="store_true",
                   help="[directed] append the chunk's realized xy progress to the flow conditioning. At eval the walker requests a high demonstrated value (default: the p90 stored in the checkpoint), steering sampling into the fast gait mode without leaving the data manifold.")
    p.add_argument("--concat-raw", action="store_true",
                   help="condition on [JEPA latent | raw normalized obs] (kitchen fix for control precision)")
    p.add_argument("--emphasis-repeat", type=int, default=0,
                   help="Duplicate the desired-minus-achieved xy vector so sampled gait chunks preserve the live servo direction.")
    p.add_argument("--agent-dims", default="27,29",
                   help="Normalized-obs slice (lo,hi) of the achieved-goal / agent xy.")
    p.add_argument("--goal-dims", default="29,31",
                   help="Normalized-obs slice (lo,hi) of the desired-goal xy.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = resolve_task(args.task, None)
    wm, norm, _, cfg = load_jepa_artifact(args.model_path, dev)
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    env = make_env(task.env_id, seed=0, max_episode_steps=task.max_episode_steps)
    spec = obs_spec_from_env(env); env.close()

    rng = np.random.default_rng(0)
    eps = load_episodes_npz(args.episodes_npz)[: args.max_episodes]
    if args.route_start is not None or args.route_goal is not None:
        if args.route_start is None or args.route_goal is None:
            p.error("--route-start and --route-goal must be provided together")
        gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
        ds, de = ge, ge + spec.goal_dim
        route_start = np.asarray(args.route_start, dtype=np.float32)
        route_goal = np.asarray(args.route_goal, dtype=np.float32)
        eps = [
            episode
            for episode in eps
            if np.linalg.norm(episode.states[0, gs:ge] - route_start) <= args.route_radius
            and np.linalg.norm(episode.states[0, ds:de] - route_goal) <= args.route_radius
        ]
        if not eps:
            raise RuntimeError("Route specialization selected no episodes")
        print(
            json.dumps(
                {
                    "event": "route_specialization",
                    "episodes": len(eps),
                    "start": route_start.tolist(),
                    "goal": route_goal.tolist(),
                    "radius": args.route_radius,
                }
            ),
            flush=True,
        )
    prog = None
    if args.successful_prefix_final_goal:
        gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
        ds, de = ge, ge + spec.goal_dim
        prefix_states: list[np.ndarray] = []
        prefix_chunks: list[np.ndarray] = []
        prefix_progress: list[float] = []
        for episode in eps:
            states_ep, actions_ep = episode.states, episode.actions
            goal_ep = states_ep[0, ds:de]
            distance = np.linalg.norm(states_ep[:, gs:ge] - goal_ep[None], axis=1)
            hits = np.flatnonzero(distance <= 0.5)
            if not len(hits):
                continue
            stop = min(len(actions_ep), int(hits[0]) + 1)
            for t in range(stop):
                end = min(t + args.chunk, len(states_ep) - 1)
                progress = float(distance[t] - distance[end])
                if progress < args.min_progress:
                    continue
                chunk = actions_ep[t : t + args.chunk]
                if len(chunk) < args.chunk:
                    chunk = np.concatenate(
                        [chunk, np.repeat(chunk[-1:], args.chunk - len(chunk), axis=0)]
                    )
                prefix_states.append(states_ep[t])
                prefix_chunks.append(chunk)
                prefix_progress.append(progress)
        S = norm.encode(np.asarray(prefix_states, dtype=np.float32))
        C = np.asarray(prefix_chunks, dtype=np.float32)
        prog = np.asarray(prefix_progress, dtype=np.float32)
    elif args.directed:
        S, C, prog = build_directed_chunks(eps, spec, norm, args.chunk, rng,
                                           max_relabel_h=args.max_relabel_h, min_progress=args.min_progress)
    else:
        S, C = build_chunks(eps, spec, norm, args.chunk, args.future_goal_frac, rng)
    a_lo, a_hi = (int(x) for x in args.agent_dims.split(","))
    g_lo, g_hi = (int(x) for x in args.goal_dims.split(","))
    progress_stats = None
    if args.progress_cond:
        if prog is None:
            raise SystemExit("--progress-cond requires a dataset mode with realized progress.")
        progress_stats = {f"p{q}": round(float(np.percentile(prog, q)), 4) for q in (50, 75, 90, 95)}
    requested_progress = float(progress_stats["p90"]) if progress_stats is not None else None

    def encode_condition(states, progress_values=None):
        st = torch.from_numpy(states).to(dev)
        with torch.no_grad():
            z = torch.cat([wm.encode(st[i:i + 16384]) for i in range(0, len(st), 16384)], 0)
        out = torch.cat([z, st], dim=1) if args.concat_raw else z
        if args.emphasis_repeat > 0:
            delta = (st[:, g_lo:g_hi] - st[:, a_lo:a_hi]).repeat(1, args.emphasis_repeat)
            out = torch.cat([out, delta], dim=1)
        if args.progress_cond:
            if progress_values is None:
                progress_values = np.full(len(st), requested_progress, dtype=np.float32)
            out = torch.cat(
                [out, torch.as_tensor(progress_values, dtype=st.dtype, device=dev).unsqueeze(1)],
                dim=1,
            )
        return out

    cond = encode_condition(S, prog)
    Ct = torch.from_numpy(C.reshape(len(C), -1)).to(dev)
    aux_cond = aux_chunks = None
    n_aux = 0
    if args.auxiliary_episodes_npz:
        aux_eps = []
        for path in args.auxiliary_episodes_npz:
            aux_eps.extend(load_episodes_npz(path))
        aux_eps = aux_eps[: args.max_auxiliary_episodes]
        gs, ge = spec.obs_dim, spec.obs_dim + spec.goal_dim
        goal_pool = np.concatenate([ep.states[::20, gs:ge] for ep in eps], axis=0)
        aux_states, aux_actions = build_auxiliary_chunks(
            aux_eps,
            spec,
            norm,
            args.chunk,
            rng,
            goal_pool=goal_pool,
            goal_copies=args.auxiliary_goal_copies,
        )
        # Recovery chunks preserve posture rather than making route progress.
        # Labeling them with the requested p90 token makes the progress control
        # ambiguous and can collapse a "fast" request into stationary recovery.
        aux_progress = (
            np.zeros(len(aux_states), dtype=np.float32)
            if args.progress_cond else None
        )
        aux_cond = encode_condition(aux_states, aux_progress)
        aux_chunks = torch.from_numpy(aux_actions.reshape(len(aux_actions), -1)).to(dev)
        n_aux = len(aux_cond)
    chunk_dim = Ct.shape[1]; cond_dim = cond.shape[1]; N = len(cond)
    # Speed-weighted sampling: bias training toward the FASTER directed segments
    # (higher xy progress per chunk) WITHOUT dropping the slower turning segments,
    # so the gait gets faster without losing the ability to turn. p ~ progress^w.
    sample_p = None
    if prog is not None and args.speed_weight > 0:
        w = np.clip(prog, 1e-3, None) ** args.speed_weight
        sample_p = torch.from_numpy((w / w.sum()).astype(np.float32)).to(dev)
    print(json.dumps({"event": "flow_data", "n": N, "chunk_dim": chunk_dim, "cond_dim": cond_dim,
                      "concat_raw": bool(args.concat_raw), "emphasis_repeat": int(args.emphasis_repeat),
                      "speed_weight": float(args.speed_weight),
                      "auxiliary_n": n_aux,
                      "auxiliary_fraction": float(args.auxiliary_fraction) if n_aux else 0.0,
                      "flow_architecture": args.flow_architecture,
                      "flow_blocks": int(args.flow_blocks)}), flush=True)

    net = build_flow_net(
        chunk_dim,
        cond_dim,
        args.hidden,
        architecture=args.flow_architecture,
        n_blocks=args.flow_blocks,
    ).to(dev)
    if args.resume is not None:
        previous = torch.load(args.resume, map_location=dev, weights_only=False)
        previous_cfg = previous["config"]
        expected = {
            "chunk_dim": int(chunk_dim),
            "cond_dim": int(cond_dim),
            "hidden": int(args.hidden),
            "flow_architecture": args.flow_architecture,
            "flow_blocks": int(args.flow_blocks),
        }
        actual = {
            "chunk_dim": int(previous_cfg["chunk_dim"]),
            "cond_dim": int(previous_cfg["cond_dim"]),
            "hidden": int(previous_cfg["hidden"]),
            "flow_architecture": str(previous_cfg.get("flow_architecture", "mlp")),
            "flow_blocks": int(previous_cfg.get("flow_blocks", 4)),
        }
        if actual != expected:
            raise ValueError(f"Incompatible --resume checkpoint: expected {expected}, found {actual}")
        net.load_state_dict(previous["flow"])
        print(json.dumps({"event": "flow_resumed", "path": str(args.resume)}), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        batch_main = args.batch_size
        batch_aux = 0
        if aux_cond is not None:
            batch_aux = min(
                args.batch_size - 1,
                max(1, int(round(args.batch_size * args.auxiliary_fraction))),
            )
            batch_main -= batch_aux
        if sample_p is not None:
            idx = torch.multinomial(sample_p, batch_main, replacement=True)
        else:
            idx = torch.randint(0, N, (batch_main,), device=dev)
        x1 = Ct[idx]; c = cond[idx]
        if batch_aux:
            idx_aux = torch.randint(0, n_aux, (batch_aux,), device=dev)
            x1 = torch.cat([x1, aux_chunks[idx_aux]], dim=0)
            c = torch.cat([c, aux_cond[idx_aux]], dim=0)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=dev)
        xt = (1 - t) * x0 + t * x1
        v = net(xt, t, c)
        loss = nn.functional.mse_loss(v, x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0:
            print(json.dumps({"event": "flow_train", "step": step, "loss": round(float(loss), 4)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"flow": net.state_dict(),
                "config": {"chunk": args.chunk, "action_dim": spec.action_dim, "hidden": args.hidden,
                           "chunk_dim": chunk_dim, "cond_dim": cond_dim, "latent_dim": int(cfg["latent_dim"]),
                           "concat_raw": bool(args.concat_raw),
                           "emphasis_repeat": int(args.emphasis_repeat),
                           "progress_cond": bool(args.progress_cond),
                           "progress_stats": progress_stats,
                           "unified_auxiliary": bool(n_aux),
                           "auxiliary_fraction": float(args.auxiliary_fraction) if n_aux else 0.0,
                           "flow_architecture": args.flow_architecture,
                           "flow_blocks": int(args.flow_blocks),
                           "agent_dims": [a_lo, a_hi], "goal_dims": [g_lo, g_hi]}},
               args.out)
    print(json.dumps({"event": "flow_saved", "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
