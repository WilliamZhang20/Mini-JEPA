"""Label FrankaKitchen demos by subtask for skill-hierarchy training.

The conditioning/capacity levers plateau at ~1.85/4 (full-4=0) because the policy
must chain 4 subtasks from sparse full-sequence data. A skill-hierarchy fixes this:
a trivial high-level scheduler picks the next subtask, and a subtask-CONDITIONED low
level executes it — so each subtask is a fresh short-horizon problem.

To train the subtask-conditioned skill we need per-transition labels of *which*
subtask is being pursued. The Minari reward is only a count, so we recover the
labels by REPLAYING each demo's actions in the env and reading
``info['step_task_completions']`` (the env's own task names). Each transition is
tagged with the task that completes NEXT in this demo (look-ahead) — i.e. what the
segment is working toward. The stored states are the observations from that same
replay, not the Minari observations from a potentially different reset. Output:
an npz of (states, actions, target one-hot, cumulative completions).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MINARI_DATASETS_PATH", "/u5/w223zhan/jepa-mini/.cache/minari")

from jepa_robotics.algos.task_families.kitchen import parse_kitchen_tasks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="D4RL/kitchen/partial-v2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=600)
    p.add_argument("--tasks", default="all",
                   help="Comma-separated completion vocabulary or 'all' for all seven Kitchen tasks.")
    args = p.parse_args()
    tasks = parse_kitchen_tasks(args.tasks)

    import minari
    from jepa_robotics.envs import make_env

    ds = minari.load_dataset(args.dataset)
    env = make_env("FrankaKitchen-v1", seed=0, max_episode_steps=2000, kitchen_tasks=tasks)
    t2i = {task: index for index, task in enumerate(tasks)}

    states_all, actions_all, target_all, completion_all = [], [], [], []
    n_kept = 0
    for ep_i, ep in enumerate(ds.iterate_episodes()):
        if ep_i >= args.max_episodes:
            break
        A = np.asarray(ep.actions, np.float32)
        obs, _ = env.reset(seed=ep_i)
        # roll the demo; record the set of done tasks after each step
        done_after = []           # done_after[t] = set of tasks completed after action t
        replay_states = [np.asarray(obs, dtype=np.float32).copy()]
        done = set()
        ok = True
        for a in A:
            try:
                obs, r, term, trunc, info = env.step(np.clip(a, env.action_space.low, env.action_space.high))
            except Exception:
                ok = False; break
            done |= set(info.get("step_task_completions", []))
            done_after.append(set(done))
            replay_states.append(np.asarray(obs, dtype=np.float32).copy())
            if term or trunc:
                A = A[: len(done_after)]; break
        if not ok or len(done_after) < 5:
            continue
        T = len(done_after)
        # target at transition t = the next NEW task completed at some t' >= t (look-ahead)
        target = np.full(T, -1, np.int64)
        nxt = -1
        for t in range(T - 1, -1, -1):
            newly = done_after[t] - (done_after[t - 1] if t > 0 else set())
            for tk in newly:
                if tk in t2i:
                    nxt = t2i[tk]
            target[t] = nxt
        # keep only transitions with a defined target (working toward a known task)
        S = np.asarray(replay_states, np.float32)[: T + 1]
        valid = target >= 0
        if valid.sum() < 5:
            continue
        oh = np.zeros((T, len(tasks)), np.float32)
        oh[np.arange(T)[valid], target[valid]] = 1.0
        completed = np.zeros((T + 1, len(tasks)), np.float32)
        for t, done_t in enumerate(done_after):
            for task in done_t:
                if task in t2i:
                    completed[t + 1, t2i[task]] = 1.0
        states_all.append(S); actions_all.append(A[:T]); target_all.append(oh)
        completion_all.append(completed)
        n_kept += 1

    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             states=np.array(states_all, dtype=object),
             actions=np.array(actions_all, dtype=object),
             targets=np.array(target_all, dtype=object),
             completions=np.array(completion_all, dtype=object),
             task_names=np.asarray(tasks),
             state_source=np.asarray("replay"))
    # quick label sanity: distribution of targets
    allt = np.concatenate([t.argmax(-1)[t.sum(-1) > 0] for t in target_all]) if target_all else np.array([])
    dist = {tasks[i]: int((allt == i).sum()) for i in range(len(tasks))} if len(allt) else {}
    print(f'{{"event": "labeled", "dataset": "{args.dataset}", "episodes_kept": {n_kept}, '
          f'"out": "{args.out}", "tasks": {tasks}, "target_dist": {dist}}}', flush=True)


if __name__ == "__main__":
    main()
