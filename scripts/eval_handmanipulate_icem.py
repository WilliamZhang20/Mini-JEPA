"""Long-horizon iCEM planner for Shadow Hand in-hand reorientation (demo-free SSL).

Grounded in the literature (GC-PMPC arXiv:2504.21585 reaches 70-80% demo-free with
CEM-MPC at horizon 30-50; iCEM arXiv:2008.06389; Contact-Implicit MPC arXiv:2402.18897
adds a finger-reset term so the optimizer actively breaks contact). The earlier
short-horizon (H=8-16) CEM/flow controllers stalled at a ~30 deg regrasp ceiling
because a finger-gait cycle spans ~30-50 steps and a short window cannot contain
one. This planner:

* rolls the DexterousJEPA world model over a LONG horizon (H~30-40),
* uses iCEM: colored-noise (temporally correlated) action samples + keep-and-shift
  elite memory across MPC steps + best-first execution,
* scores geodesic distance to the goal orientation (+ ensemble-disagreement),
* adds a contact-breaking bonus that rewards finger motion, so the optimizer does
  not fall into the "freeze the grip" local optimum where it never regrasps.

Pure model-based planning over a self-supervised world model with no demos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from jepa_robotics.envs import flatten_obs, make_env
from jepa_robotics.evaluate import load_jepa_artifact
from jepa_robotics.algos.planning.objectives import CommonScoringMixin
from jepa_robotics.tasks import resolve_task


def colored_noise(n, H, A, beta, device):
    """Temporally-correlated noise, PSD ~ 1/f^beta (iCEM). Returns [n,H,A], unit std."""
    freqs = torch.fft.rfftfreq(H, device=device).clamp_min(1.0 / H)  # avoid div-by-0 at DC
    scale = freqs ** (-beta / 2.0)
    spec = (torch.randn(n, A, freqs.shape[0], device=device)
            + 1j * torch.randn(n, A, freqs.shape[0], device=device)) * scale.view(1, 1, -1)
    noise = torch.fft.irfft(spec, n=H, dim=-1)                       # [n,A,H]
    noise = noise / (noise.std(dim=-1, keepdim=True) + 1e-8)
    return noise.transpose(1, 2).contiguous()                        # [n,H,A]


class ICEM(CommonScoringMixin):
    def __init__(self, wm, norm, spec, dev, *, H, N, iters, elite_frac, init_std, beta,
                 keep_frac, exec_k, disagree_w, reset_w, path_w, fine_deg=0.0, fine_H=8,
                 fine_N=512, action_l2_w=0.0, action_delta_w=0.0, slew_limit=0.0):
        self.wm, self.norm, self.spec, self.dev = wm, norm, spec, dev
        self.H, self.N, self.iters = H, N, iters
        self.n_elite = max(2, int(N * elite_frac))
        self.init_std, self.beta = init_std, beta
        self.keep = max(1, int(self.n_elite * keep_frac))
        self.exec_k, self.disagree_w, self.reset_w, self.path_w = exec_k, disagree_w, reset_w, path_w
        self.action_l2_weight = action_l2_w
        self.action_delta_weight = action_delta_w
        self.slew_limit = slew_limit
        self.fine_rad, self.fine_H, self.fine_N = np.radians(fine_deg), fine_H, fine_N
        self.fine_kappa = 0.0
        self.mid_rad, self.mid_H, self.mid_N, self.mid_kappa = 0.0, 8, 256, 0.0  # middle 'settle' gear
        self.short_exec, self._gear = 3, None  # steps executed per short-gear plan
        self.coarse_metric = 'abs'
        self.min_approach = False
        self.A = spec.action_dim
        self.ag = spec.obs_dim
        self.mean_t = torch.as_tensor(norm.mean, dtype=torch.float32, device=dev)
        self.std_t = torch.as_tensor(norm.std, dtype=torch.float32, device=dev)
        self._buf = []
        self.reset()

    def reset(self):
        self.mean = torch.zeros(self.H, self.A, device=self.dev)
        self.kept = None
        self._buf = []
        self._gear = None
        self.prev_action = np.zeros(self.A, dtype=np.float32)

    def _decode(self, z_seq):  # [.,H,latent] -> raw state [.,H,state]
        return self.norm.decode_tensor(self.wm.state_probe(z_seq))

    @torch.no_grad()
    def _cost(self, z0, acts, dg_q, H=None, kappa=None, use_abs=True, metric=None):
        H = H or self.H
        kappa = self.disagree_w if kappa is None else kappa
        metric = metric or ("abs" if use_abs else "signed")
        Nn = acts.shape[0]
        rolls = self.wm.rollout_heads(z0.expand(Nn, -1), acts, H)                  # [K,Nn,H,latent]
        K = rolls.shape[0]
        # Risk-averse PETS-style cost: decode EACH head's rollout and score its own
        # geodesic-to-goal, then aggregate as mean + risk_kappa*std across heads.
        # Ensemble disagreement correlates with prediction error (measured 0.75 @H16),
        # so penalizing cost-variance makes CEM avoid trajectories it would only reach
        # in an optimistic head -- the anti-model-exploitation the ensemble buys.
        pred = self._decode(rolls.reshape(K * Nn, H, -1)).reshape(K, Nn, H, -1)    # [K,Nn,H,state]
        qa = pred[..., self.ag + 3:self.ag + 7]
        qa = qa / torch.linalg.vector_norm(qa, dim=-1, keepdim=True).clamp_min(1e-6)
        dot = (qa * dg_q.view(1, 1, 1, 4)).sum(-1)
        # 'abs'   : 2*acos(|dot|) shortest-arc — but dot=-1 (wrong hemisphere, a full
        #           turn off) is a SPURIOUS minimum, so on ~180deg flips the planner
        #           picks the losing spin direction ~half the time.
        # 'signed': 2*acos(dot) — exact env metric, but the 2*pi range disrupts search.
        # 'smooth': 1-dot — minimized ONLY at dot=+1 (the winning hemisphere), smooth,
        #           bounded [0,2]; drives the flip the correct way without the acos blowup.
        if metric == "abs":
            rot = 2.0 * torch.acos(dot.abs().clamp(-1.0, 1.0))                     # [K,Nn,H]
        elif metric == "smooth":
            rot = 1.0 - dot
        else:
            rot = 2.0 * torch.acos(dot.clamp(-1.0, 1.0))
        if self.min_approach:
            # Reward the CLOSEST approach anywhere in the rollout (+ mild terminal),
            # so far targets get driven INTO the fine gear's range instead of the
            # planner stopping short at a terminal-only optimum and wandering.
            pcost = rot.min(dim=-1).values + self.path_w * rot[:, :, -1]           # [K,Nn]
        else:
            pcost = rot[:, :, -1] + self.path_w * rot.mean(-1)                     # [K,Nn] per-head cost
        cost = pcost.mean(0) + kappa * pcost.std(0)                                # risk aggregate (kappa<0 = risk-seeking)
        if self.reset_w > 0:  # contact-breaking: reward finger motion (mean head)
            hand = pred[:, :, :, :24].mean(0)
            cost = cost - self.reset_w * (hand[:, 1:] - hand[:, :-1]).abs().mean(dim=(1, 2))
        return cost + self._action_regularizers(acts)

    @torch.no_grad()
    def _plan_fine(self, raw, dg_q, lo, hi):
        """Very-short-horizon accurate CEM straight to the goal, for the tight
        terminal precision. Uses fine_H (~2, where WM err ~3.8deg < the 5.7deg
        threshold) and a RISK-NEUTRAL cost (kappa=fine_kappa~0): near the goal we
        want the accurate ensemble mean, not the risk-seeking exploration the
        coarse gaiting stage needs."""
        return self._plan_short(raw, dg_q, lo, hi, self.fine_H, self.fine_kappa, self.fine_N)

    @torch.no_grad()
    def _plan_short(self, raw, dg_q, lo, hi, H, kappa, N, iters=6, use_abs=True):
        """Plain CEM (white noise) over a short horizon straight to the goal, replan
        every step. Used for the middle 'settle' gear (H~8) and the precise gear
        (H~2), both risk-neutral. The precise gear passes use_abs=False so it drives
        to the correct hemisphere (dot->+1), which is what the env scores."""
        z0 = self.wm.encode(torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev))
        mean = torch.zeros(H, self.A, device=self.dev)
        std = torch.full((H, self.A), 0.3, device=self.dev)
        best = None
        for _ in range(iters):
            acts = (mean.unsqueeze(0) + std.unsqueeze(0)
                    * torch.randn(N, H, self.A, device=self.dev)).clamp(lo, hi)
            acts = self._rate_limit_actions(acts)
            order = torch.argsort(self._cost(z0, acts, dg_q, H=H, kappa=kappa, use_abs=use_abs))
            elite = acts[order[: max(2, N // 10)]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.02)
            best = acts[order[0]]
        return best.cpu().numpy()

    @torch.no_grad()
    def _plan(self, raw, dg_q, lo, hi):
        z0 = self.wm.encode(torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev))
        mean, std = self.mean.clone(), torch.full((self.H, self.A), self.init_std, device=self.dev)
        elites = None
        for it in range(self.iters):
            noise = colored_noise(self.N, self.H, self.A, self.beta, self.dev)
            samples = (mean.unsqueeze(0) + std.unsqueeze(0) * noise)
            if self.kept is not None and it == 0:                       # elite memory across steps
                samples = torch.cat([samples, self.kept], dim=0)
            samples = samples.clamp(lo, hi)
            samples = self._rate_limit_actions(samples)
            cost = self._cost(z0, samples, dg_q, metric=self.coarse_metric)
            order = torch.argsort(cost)
            elites = samples[order[: self.n_elite]]
            mean, std = elites.mean(0), elites.std(0).clamp_min(0.02)
        self.kept = elites[: self.keep].clone()                         # keep for next iter/step
        # shift mean and kept elites one step for the next env step (receding horizon warm start)
        self.mean = torch.cat([mean[1:], mean[-1:].detach() * 0.0], dim=0)
        self.kept = torch.cat([self.kept[:, 1:], self.kept[:, -1:]], dim=1)
        return elites[0].cpu().numpy()

    def act(self, obs, env, dg_q, lo, hi):
        raw = flatten_obs(obs)
        clip = lambda a: np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
        qc = raw[self.ag + 3:self.ag + 7]; qc = qc / (np.linalg.norm(qc) + 1e-9)
        gap = 2.0 * np.arccos(min(1.0, abs(float(qc @ dg_q.cpu().numpy()))))
        # THREE GEARS by distance to target: precise (very close) | settle (middle) |
        # aggressive gaiting (far). The middle gear bridges the ~20-40deg band where
        # the far gaiting wanders and the precise gear is too slow. Short gears
        # execute a few steps per plan (not every step) so they stay affordable.
        gear = "fine" if (self.fine_rad > 0 and gap < self.fine_rad) else \
               "mid" if (self.mid_rad > 0 and gap < self.mid_rad) else "coarse"
        if gear != self._gear:
            self._buf = []; self._gear = gear
        if not self._buf:
            if gear == "fine":
                plan = self._plan_short(raw, dg_q, lo, hi, self.fine_H, self.fine_kappa, self.fine_N, use_abs=False)
                k = min(self.short_exec, self.fine_H)
            elif gear == "mid":
                plan = self._plan_short(raw, dg_q, lo, hi, self.mid_H, self.mid_kappa, self.mid_N, iters=5)
                k = self.short_exec
            else:
                plan = self._plan(raw, dg_q, lo, hi)
                k = self.exec_k
            self._buf = [plan[i].copy() for i in range(max(1, min(k, len(plan))))]
        action = clip(self._buf.pop(0))
        if self.slew_limit > 0:
            action = clip(
                self.prev_action
                + np.clip(action - self.prev_action, -self.slew_limit, self.slew_limit)
            )
        self.prev_action = action.copy()
        return action


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="handmanipulate_block_rotate_z")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=60000)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--keep-frac", type=float, default=0.3)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=2.5, help="colored-noise exponent (0=white)")
    p.add_argument("--exec-k", type=int, default=4)
    p.add_argument("--path-weight", type=float, default=0.25)
    p.add_argument("--disagree-weight", type=float, default=0.0)
    p.add_argument("--reset-weight", type=float, default=0.0, help="contact-breaking: reward finger motion")
    p.add_argument("--action-l2-weight", type=float, default=0.0,
                   help="Penalize large candidate actions in the iCEM objective.")
    p.add_argument("--action-delta-weight", type=float, default=0.0,
                   help="Penalize candidate action changes, including the first action versus the command just executed.")
    p.add_argument("--slew-limit", type=float, default=0.0,
                   help="Per-actuator rate bound applied before world-model scoring and at execution (0 disables).")
    p.add_argument("--fine-deg", type=float, default=0.0, help="within this angle of goal, switch to short-horizon accurate CEM (0=off)")
    p.add_argument("--fine-horizon", type=int, default=2, help="WM meets the 5.7deg threshold only at H<=2 (err 3.8deg)")
    p.add_argument("--fine-candidates", type=int, default=512)
    p.add_argument("--fine-kappa", type=float, default=0.0, help="risk coeff for the fine stage (0=neutral/accurate)")
    p.add_argument("--mid-deg", type=float, default=0.0, help="below this gap (and above fine-deg) use the middle 'settle' gear (0=off)")
    p.add_argument("--mid-horizon", type=int, default=8)
    p.add_argument("--mid-candidates", type=int, default=256)
    p.add_argument("--mid-kappa", type=float, default=0.0)
    p.add_argument("--coarse-metric", default="abs", choices=["abs", "signed", "smooth"], help="cost for the big-turn stage; smooth=1-dot fixes ~180deg flips going the wrong way")
    p.add_argument("--min-approach", action="store_true", help="reward closest approach in the rollout (drives far targets into the fine gear range)")
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-episodes", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dev = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.torch_seed)
    task = resolve_task(args.task, None)
    wm, norm, spec, cfg = load_jepa_artifact(args.model_path, dev)
    H = min(args.horizon, int(cfg.get("max_horizon", args.horizon)))
    env = make_env(task.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    lo = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=dev)
    mpc = ICEM(wm, norm, spec, dev, H=H, N=args.candidates, iters=args.iters,
               elite_frac=args.elite_frac, init_std=args.init_std, beta=args.beta,
               keep_frac=args.keep_frac, exec_k=args.exec_k, disagree_w=args.disagree_weight,
               reset_w=args.reset_weight, path_w=args.path_weight,
               fine_deg=args.fine_deg, fine_H=args.fine_horizon, fine_N=args.fine_candidates,
               action_l2_w=args.action_l2_weight, action_delta_w=args.action_delta_weight,
               slew_limit=args.slew_limit)
    mpc.fine_kappa = args.fine_kappa
    mpc.mid_rad = np.radians(args.mid_deg)
    mpc.mid_H, mpc.mid_N, mpc.mid_kappa = args.mid_horizon, args.mid_candidates, args.mid_kappa
    mpc.coarse_metric = args.coarse_metric
    mpc.min_approach = args.min_approach
    dgo = spec.obs_dim + spec.goal_dim

    def qgeo(a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.degrees(2 * np.arccos(min(1.0, abs(float(a @ b))))))

    succ, gaps = [], []
    action_delta_sq, action_jerk_sq = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        mpc.reset()
        s = flatten_obs(obs)
        dg_q = torch.as_tensor(s[dgo + 3:dgo + 7] / (np.linalg.norm(s[dgo + 3:dgo + 7]) + 1e-9),
                               dtype=torch.float32, device=dev)
        term = trunc = False; info = {}
        actions: list[np.ndarray] = []
        while not (term or trunc):
            action = mpc.act(obs, env, dg_q, lo, hi)
            actions.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            s = flatten_obs(obs)
        succ.append(float(info.get("is_success", 0.0)))
        gaps.append(qgeo(s[spec.obs_dim + 3:spec.obs_dim + 7], s[dgo + 3:dgo + 7]))
        ep_actions = np.asarray(actions, dtype=np.float32)
        deltas = np.diff(
            np.concatenate([np.zeros((1, spec.action_dim), np.float32), ep_actions], axis=0),
            axis=0,
        )
        jerks = np.diff(deltas, axis=0)
        action_delta_sq.extend(np.square(deltas).reshape(-1).tolist())
        action_jerk_sq.extend(np.square(jerks).reshape(-1).tolist())
        if args.log_episodes:
            print(json.dumps({
                "event": "handmanipulate_icem_episode",
                "episode": ep,
                "success": succ[-1],
                "final_gap_deg": round(gaps[-1], 1),
                "action_delta_rms": round(float(np.sqrt(np.mean(np.square(deltas)))), 4),
                "action_jerk_rms": round(float(np.sqrt(np.mean(np.square(jerks)))), 4),
            }), flush=True)
    env.close()
    row = {"event": "handmanipulate_icem_eval", "task": task.name, "horizon": H,
           "model_path": str(args.model_path), "seed": args.seed, "torch_seed": args.torch_seed,
           "episodes": args.episodes, "success_rate": round(float(np.mean(succ)), 3),
           "median_final_gap_deg": round(float(np.median(gaps)), 1),
           "action_delta_rms": round(float(np.sqrt(np.mean(action_delta_sq))), 4),
           "action_jerk_rms": round(float(np.sqrt(np.mean(action_jerk_sq))), 4),
           "action_l2_weight": args.action_l2_weight,
           "action_delta_weight": args.action_delta_weight, "slew_limit": args.slew_limit,
           "beta": args.beta, "candidates": args.candidates, "reset_weight": args.reset_weight,
           "exec_k": args.exec_k}
    print(json.dumps(row), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
