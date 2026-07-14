"""Goal-conditioned low-level controllers for the H-JEPA maze hierarchy.

The high level (HWM flow-macro prior, ``eval_hjepa_hwm.py``) sets a subgoal in
achieved_goal (xy) space; these low levels pursue it. The rectified-flow walker
(``LowLevelFlow``) is the canonical one — a directed-motion goal-delta-emphasis
gait that samples a coherent H-step chunk per subgoal. BC and inverse variants
are retained for point tasks / comparison.

(Moved out of the retired ``eval_hjepa_maze.py``, which was the Dijkstra-graph
controller. The empirical subgoal graph is no longer used — the neural flow-macro
HWM high level replaced it on every maze.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from jepa_robotics.envs import flatten_obs


class LowLevelBC:
    """Goal-conditioned BC policy on the JEPA latent."""

    def __init__(self, wm_path, bc_path, low, high, device="cpu"):
        import torch
        from jepa_robotics.evaluate import load_jepa_artifact, load_policy_artifact
        self._torch = torch
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        self.policy, pcfg = load_policy_artifact(Path(bc_path), self.dev)
        # A raw policy (train_gcrl_raw.py) acts on the normalized observation
        # directly; a latent policy acts on the JEPA encoding of it.
        self.raw = bool(pcfg.get("raw", False))
        self.model.eval(); self.policy.eval()
        self.low, self.high = low, high

    def act(self, obs, subgoal):
        o = {k: np.array(v, copy=True) for k, v in obs.items()}
        o["desired_goal"] = np.asarray(subgoal, dtype=np.float32)
        s = self._torch.from_numpy(self.norm.encode(flatten_obs(o))).unsqueeze(0).to(self.dev)
        with self._torch.no_grad():
            z = s if self.raw else self.model.encode(s)
            a = self.policy(z)[0].cpu().numpy()
        return np.clip(a, self.low, self.high).astype(np.float32)


class ProgressScorer:
    """Chunk-outcome predictor S(cond, chunk) -> xy progress toward the conditioned
    goal over the chunk. The walker's analog of the Fetch JEPA chunk selector:
    sample N on-manifold gait chunks from the flow, execute the one predicted to
    close the most distance. Extracts the demonstrated FAST gait mode without
    off-manifold conditioning (the demos are bimodal: ~60% near-stationary, but
    the moving mode reaches ~1.1-1.5 xy units per 8-step chunk)."""

    def __init__(self, cond_dim, chunk_dim, hidden=512):
        from jepa_robotics.models import MLP
        self.net = MLP([cond_dim + chunk_dim, hidden, hidden, 1], layer_norm=True)

    @classmethod
    def load(cls, path, device):
        import torch
        art = torch.load(Path(path), map_location=device, weights_only=False)
        cfg = art["config"]
        s = cls(int(cfg["cond_dim"]), int(cfg["chunk_dim"]), int(cfg["hidden"]))
        s.net.load_state_dict(art["scorer"])
        s.net.to(device).eval()
        return s

    def score(self, cond, chunks):
        import torch
        with torch.no_grad():
            return self.net(torch.cat([cond, chunks], dim=-1)).squeeze(-1)


class LowLevelFlow:
    """Action-chunked rectified-flow goal-conditioned walker (canonical).

    Samples a coherent H-step gait chunk from the conditional flow (multimodality
    preserved, no BC mush) and executes it receding-horizon. Conditions on the
    JEPA latent (+ raw obs if concat_raw), with optional Relocate-style goal-delta
    emphasis: append the (desired - achieved) xy vector so the sampled gait heads
    at the live subgoal. Config is read from the checkpoint."""

    def __init__(self, wm_path, bc_path, low, high, device="cpu", replan=None, flow_steps=10,
                 scorer_path=None, n_samples=1, target_progress=None, select_quantile=1.0,
                 recovery_path=None, recovery_exit=0.7, risk_scorer_path=None, risk_threshold=0.5):
        import torch
        from jepa_robotics.evaluate import load_jepa_artifact
        from scripts.train_flow_walker import FlowNet
        self._torch = torch
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        art = torch.load(Path(bc_path), map_location=self.dev, weights_only=False)
        cfg = art["config"]
        self.chunk = int(cfg["chunk"]); self.action_dim = int(cfg["action_dim"])
        self.concat_raw = bool(cfg["concat_raw"]); self.chunk_dim = int(cfg["chunk_dim"])
        self.net = FlowNet(int(cfg["chunk_dim"]), int(cfg["cond_dim"]), int(cfg["hidden"])).to(self.dev)
        self.net.load_state_dict(art["flow"]); self.net.eval(); self.model.eval()
        self.low, self.high = low, high
        self.replan = replan or self.chunk
        self.flow_steps = flow_steps
        self.emphasis_repeat = int(cfg.get("emphasis_repeat", 0) or 0)
        self.agent_dims = tuple(cfg.get("agent_dims", [27, 29]))
        self.goal_dims = tuple(cfg.get("goal_dims", [29, 31]))
        # Best-of-N chunk selection (Fetch chunk-selection move for the gait):
        # sample n_samples chunks per replan and execute the one the progress
        # scorer predicts closes the most xy distance to the live subgoal.
        self.n_samples = max(1, int(n_samples))
        self.scorer = None
        if scorer_path is not None and self.n_samples > 1:
            self.scorer = ProgressScorer.load(scorer_path, self.dev)
        # 1.0 = argmax predicted progress. Pure argmax over many replans selects
        # the winner's-curse tail (over-predicted / physically riskiest chunks —
        # an ant flip is unrecoverable), so a sub-max quantile picks a
        # fast-but-not-extreme chunk instead.
        self.select_quantile = float(select_quantile)
        # Progress-conditioned flow: request a high demonstrated per-chunk xy
        # progress (default: the training p90 stored in the checkpoint) to steer
        # sampling into the fast gait mode. The scorer, if any, sees the BASE
        # cond (without this scalar), so the two levers compose.
        self.progress_cond = bool(cfg.get("progress_cond", False))
        self.target_progress = None
        if self.progress_cond:
            stats = cfg.get("progress_stats") or {}
            self.target_progress = float(target_progress if target_progress is not None
                                         else stats.get("p90", 1.0))
        # Flip-risk veto (train_flip_risk_scorer.py): among n_samples gait chunks,
        # discard those whose predicted min uprightness over the chunk falls below
        # risk_threshold, then pick RANDOMLY among the safe ones — selection
        # pressure for safety only, no winner's-curse speed chasing.
        self.risk_scorer = None
        if risk_scorer_path is not None and self.n_samples > 1:
            self.risk_scorer = ProgressScorer.load(risk_scorer_path, self.dev)
            self.risk_threshold = float(risk_threshold)
        # Self-trial flip recovery (train_recovery_walker.py): when live torso
        # uprightness drops below the trained enter threshold, sample chunks from
        # the recovery prior (conditioned on [latent | raw], goal-free) until
        # uprightness clears recovery_exit (hysteresis), then resume the gait.
        # The demos contain no flipped states, so the gait flow alone flails
        # off-manifold after a flip until episode timeout.
        self.recovery = None
        if recovery_path is not None:
            art_r = torch.load(Path(recovery_path), map_location=self.dev, weights_only=False)
            rcfg = art_r["config"]
            self.recovery = FlowNet(int(rcfg["chunk_dim"]), int(rcfg["cond_dim"]), int(rcfg["hidden"])).to(self.dev)
            self.recovery.load_state_dict(art_r["flow"]); self.recovery.eval()
            self.recovery_chunk_dim = int(rcfg["chunk_dim"])
            self.recovery_enter = float(rcfg["enter"])
            self.recovery_exit = float(recovery_exit)
            self._recovering = False
        self._buf = []
        self._last_sg = None

    def act(self, obs, subgoal):
        if self.recovery is not None:
            q = np.asarray(obs["observation"][1:5], np.float32)
            upright = 1.0 - 2.0 * float(q[1] ** 2 + q[2] ** 2)
            was = self._recovering
            self._recovering = upright < (self.recovery_exit if was else self.recovery_enter)
            if self._recovering != was:
                self._buf = []  # mode switch: resample immediately
            if self._recovering:
                if not self._buf:
                    o = {k: np.array(v, copy=True) for k, v in obs.items()}
                    s = self._torch.from_numpy(self.norm.encode(flatten_obs(o))).unsqueeze(0).to(self.dev)
                    with self._torch.no_grad():
                        z = self.model.encode(s)
                        x = self.recovery.sample(self._torch.cat([z, s], dim=1),
                                                 self.recovery_chunk_dim, self.flow_steps)[0].cpu().numpy()
                    self._buf = list(x.reshape(self.chunk, self.action_dim)[: self.replan])
                    self._last_sg = None  # force gait resample after recovery
                a = self._buf.pop(0)
                return np.clip(a, self.low, self.high).astype(np.float32)
        sg = np.asarray(subgoal, dtype=np.float32)
        if (not self._buf) or self._last_sg is None or np.linalg.norm(sg - self._last_sg) > 1e-6:
            o = {k: np.array(v, copy=True) for k, v in obs.items()}
            o["desired_goal"] = sg
            s = self._torch.from_numpy(self.norm.encode(flatten_obs(o))).unsqueeze(0).to(self.dev)
            with self._torch.no_grad():
                z = self.model.encode(s)
                c = self._torch.cat([z, s], dim=1) if self.concat_raw else z
                if self.emphasis_repeat > 0:
                    a_lo, a_hi = self.agent_dims; g_lo, g_hi = self.goal_dims
                    delta = (s[:, g_lo:g_hi] - s[:, a_lo:a_hi]).repeat(1, self.emphasis_repeat)
                    c = self._torch.cat([c, delta], dim=1)
                c_flow = c
                if self.progress_cond:
                    tp = self._torch.full((c.shape[0], 1), self.target_progress,
                                          dtype=c.dtype, device=self.dev)
                    c_flow = self._torch.cat([c, tp], dim=1)
                if self.risk_scorer is not None:
                    cn = c.repeat(self.n_samples, 1)
                    xs = self.net.sample(c_flow.repeat(self.n_samples, 1), self.chunk_dim, self.flow_steps)
                    pred_u = self.risk_scorer.score(cn, xs)
                    safe = (pred_u > self.risk_threshold).nonzero().flatten()
                    pick = int(safe[0]) if len(safe) else int(pred_u.argmax())
                    x = xs[pick].cpu().numpy()
                elif self.scorer is not None:
                    cn = c.repeat(self.n_samples, 1)
                    xs = self.net.sample(c_flow.repeat(self.n_samples, 1), self.chunk_dim, self.flow_steps)
                    order = self.scorer.score(cn, xs).argsort()
                    pick = min(self.n_samples - 1, int(round(self.select_quantile * (self.n_samples - 1))))
                    x = xs[int(order[pick])].cpu().numpy()
                else:
                    x = self.net.sample(c_flow, self.chunk_dim, self.flow_steps)[0].cpu().numpy()
            chunk = x.reshape(self.chunk, self.action_dim)
            self._buf = list(chunk[: self.replan])
            self._last_sg = sg
        a = self._buf.pop(0)
        return np.clip(a, self.low, self.high).astype(np.float32)


class LowLevelInverse:
    """Self-supervised inverse chunk low level: (z_t, z_subgoal) -> action chunk;
    no action labels copied at runtime."""

    def __init__(self, wm_path, inverse_path, low, high, device="cpu", target_horizon=None):
        import torch
        from jepa_robotics.envs import goal_state_from_state
        from jepa_robotics.evaluate import load_jepa_artifact
        from jepa_robotics.algos.priors import InversePrior
        self._torch = torch
        self._goal_state_from_state = goal_state_from_state
        self.dev = torch.device(device)
        self.model, self.norm, self.spec, _ = load_jepa_artifact(Path(wm_path), self.dev)
        art = torch.load(Path(inverse_path), map_location=self.dev, weights_only=False)
        self.ckpt = art
        self.prior = InversePrior(
            int(art["cond_dim"]), int(art["chunk_dim"]), int(art["hidden"]), int(art["n_blocks"]),
        ).to(self.dev)
        self.prior.load_state_dict(art["state_dict"])
        self.prior.eval(); self.model.eval()
        self.low, self.high = low, high
        self.target_horizon = target_horizon or int(art["H"])

    def act(self, obs, subgoal):
        o = {k: np.array(v, copy=True) for k, v in obs.items()}
        o["desired_goal"] = np.asarray(subgoal, dtype=np.float32)
        raw = flatten_obs(o)
        target = self._goal_state_from_state(raw, self.spec)
        s = self._torch.from_numpy(self.norm.encode(raw)).unsqueeze(0).to(self.dev)
        tgt = self._torch.from_numpy(self.norm.encode(target)).unsqueeze(0).to(self.dev)
        with self._torch.no_grad():
            z = self.model.encode(s)
            z_goal = self.model.encode_target(tgt)
            horizons = list(self.ckpt.get("future_horizons", [int(self.ckpt["H"])]))
            h = float(self.target_horizon) / float(max(horizons))
            h_token = self._torch.tensor([[h]], dtype=z.dtype, device=self.dev)
            cond = self._torch.cat([z, z_goal, h_token], dim=-1)
            chunk = self.prior(cond).view(int(self.ckpt["H"]), int(self.ckpt["action_dim"]))
            a = chunk[0].cpu().numpy()
        return np.clip(a, self.low, self.high).astype(np.float32)
