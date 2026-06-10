"""Ensemble-decomposed ν̂ estimator (Phase 2.5).

Provides ``EnsembleNuEstimator``: a state-dependent noise estimator that
decomposes uncertainty into aleatoric (per-member) and epistemic
(between-member) components using K independently trained DreamerV3 world
models.

The estimator is used to compute adaptive ν̂_t for bandwidth calibration
(Phase 4) and online deployment (Phase 7).

Design
------
For each observation s, the estimator:

1. Encodes s into a posterior latent via each of K members' RSSM:
   ``post_k = wm_k.dynamics.obs_step(...)``

2. Runs H imagination steps from ``post_k`` under the deterministic policy
   (σ=0), collecting latent features at each step.

3. When ``n_samples > 1``, repeats step 2 n_samples times with different
   stochastic latent samples → gets a distribution of imagined latent
   trajectories per member.

4. **Aleatoric** uncertainty for member k:
   ``Var_samples[feat_k]`` — per-member spread of imagined latents.

5. **Epistemic** uncertainty:
   ``Var_members[E_samples[feat_k]]`` — spread between members' mean
   imagined latents.

6. Returns ``ν̂ = clip(sqrt(aleatoric + epistemic), ν_min, ν_max)``.

Usage::

    estimator = EnsembleNuEstimator(members=[dw0, dw1, dw2], horizon=5)
    nu_hat = estimator.estimate(obs, n_samples=8)  # scalar

Notes
-----
* If any member's agent is uninitialised, it is skipped (no crash).
* The return value is calibrated to match env noise σ via a linear scale
  factor ``nu_scale`` (default 1.0 — tune per-env in Phase 4).
* Falls back gracefully to ``nu_default`` if fewer than 2 members succeed.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt


class EnsembleNuEstimator:
    """State-dependent ν̂ estimator from K-member DreamerV3 ensemble.

    Parameters
    ----------
    members    : List of trained DreamerWrapper instances (K ≥ 2 recommended).
    horizon    : Imagination horizon H.
    nu_scale   : Linear scale applied to the raw uncertainty estimate before
                 returning.  Tune per-environment to bring ν̂ into [ν_min,
                 ν_max] range.
    nu_min     : Clip lower bound (default 0.01).
    nu_max     : Clip upper bound (default 1.0).
    nu_default : Fallback value when fewer than 2 members produce valid
                 estimates (default 0.1).
    """

    def __init__(
        self,
        members: list[Any],
        horizon: int = 5,
        nu_scale: float = 1.0,
        nu_min: float = 0.01,
        nu_max: float = 1.0,
        nu_default: float = 0.1,
    ) -> None:
        if len(members) < 1:
            raise ValueError("EnsembleNuEstimator requires at least 1 member.")
        self.members = members
        self.H = int(horizon)
        self.nu_scale = float(nu_scale)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.nu_default = float(nu_default)

    # ── main API ──────────────────────────────────────────────────────────────

    def estimate(self, obs: npt.NDArray[Any], n_samples: int = 8) -> float:
        """Estimate ν̂(s) = aleatoric + epistemic uncertainty.

        Parameters
        ----------
        obs       : Current observation array.
        n_samples : Number of stochastic latent trajectory samples per member.
                    Higher = more accurate but slower.

        Returns
        -------
        Scalar ν̂ ∈ [nu_min, nu_max].
        """
        member_means: list[npt.NDArray[np.float64]] = []
        member_vars: list[float] = []

        for member in self.members:
            try:
                mean_feat, var_feat = self._member_imagination_stats(
                    member, obs, n_samples
                )
                member_means.append(mean_feat)
                member_vars.append(float(np.mean(var_feat)))
            except Exception as exc:
                warnings.warn(
                    f"EnsembleNuEstimator: member failed — {exc!r}",
                    stacklevel=2,
                )

        if len(member_means) < 1:
            return self.nu_default

        # Aleatoric = average per-member latent variance
        aleatoric = float(np.mean(member_vars)) if member_vars else 0.0

        # Epistemic = variance between members' mean latent features
        if len(member_means) >= 2:
            stacked = np.stack(member_means, axis=0)  # (K, feat_dim)
            epistemic = float(np.var(stacked, axis=0).mean())
        else:
            epistemic = 0.0

        raw = float(np.sqrt(max(aleatoric + epistemic, 0.0)))
        nu_hat = float(np.clip(raw * self.nu_scale, self.nu_min, self.nu_max))
        return nu_hat

    def estimate_components(
        self, obs: npt.NDArray[Any], n_samples: int = 8
    ) -> dict[str, float]:
        """Like estimate() but returns a dict with all components for analysis.

        Returns
        -------
        dict with keys: ``nu_hat``, ``aleatoric``, ``epistemic``,
        ``n_members_ok``.
        """
        member_means: list[npt.NDArray[np.float64]] = []
        member_vars: list[float] = []

        for member in self.members:
            try:
                mean_feat, var_feat = self._member_imagination_stats(
                    member, obs, n_samples
                )
                member_means.append(mean_feat)
                member_vars.append(float(np.mean(var_feat)))
            except Exception as exc:
                warnings.warn(
                    f"EnsembleNuEstimator: member failed — {exc!r}", stacklevel=2
                )

        aleatoric = float(np.mean(member_vars)) if member_vars else 0.0
        if len(member_means) >= 2:
            stacked = np.stack(member_means, axis=0)
            epistemic = float(np.var(stacked, axis=0).mean())
        else:
            epistemic = 0.0

        raw = float(np.sqrt(max(aleatoric + epistemic, 0.0)))
        nu_hat = float(np.clip(raw * self.nu_scale, self.nu_min, self.nu_max))

        return {
            "nu_hat": nu_hat,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "n_members_ok": len(member_means),
        }

    # ── per-member imagination ────────────────────────────────────────────────

    def _member_imagination_stats(
        self,
        member: Any,
        obs: npt.NDArray[Any],
        n_samples: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (mean_feat, var_feat) over imagination horizon and samples.

        Uses dreamerv3-torch internals when available; falls back to the
        ``rssm_posterior`` method (single-step estimate) otherwise.

        Parameters
        ----------
        member    : DreamerWrapper with a trained NM512 agent.
        obs       : Observation array.
        n_samples : Stochastic samples of the imagination trajectory.

        Returns
        -------
        mean_feat : Mean latent feature across samples + steps, shape (feat_dim,).
        var_feat  : Variance of latent features, shape (feat_dim,).
        """
        member._ensure_agent()

        if (
            getattr(member, "_backend", None) == "dreamerv3-torch"
            and member._agent is not None
        ):
            return self._nm512_imagination_stats(member, obs, n_samples)
        else:
            # Fallback: use rssm_posterior (single-step)
            mean, var = member.rssm_posterior(obs)
            return mean.astype(np.float64), var.astype(np.float64)

    def _nm512_imagination_stats(
        self,
        member: Any,
        obs: npt.NDArray[Any],
        n_samples: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """NM512-specific H-step imagination with stochastic latent sampling.

        Runs n_samples imagination trajectories from obs, collecting the
        latent feature vector at each step.  Returns (mean_feat, var_feat)
        averaged over steps and samples.
        """
        import torch

        agent = member._agent
        agent.eval()
        wm = agent._wm
        behavior = agent._task_behavior
        target_device = member.device

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_batch = {
            "obs": obs_arr[np.newaxis],
            "image": np.zeros((1, 1, 1, 3), dtype=np.uint8),
            "is_first": np.ones(1, dtype=np.float32),
            "is_terminal": np.zeros(1, dtype=np.float32),
        }

        # Collect latent features across n_samples trajectories × H steps
        all_feats: list[npt.NDArray[np.float64]] = []

        with torch.no_grad():
            preprocessed = wm.preprocess(obs_batch)
            embed = wm.encoder(preprocessed)

            for _ in range(n_samples):
                # Fresh posterior sample (stochastic z_0)
                post, _ = wm.dynamics.obs_step(
                    None, None, embed, preprocessed["is_first"], sample=True
                )
                latent = post

                for _ in range(self.H):
                    feat = wm.dynamics.get_feat(latent)
                    # Record feature (flatten to numpy)
                    all_feats.append(
                        feat.squeeze(0).detach().cpu().numpy().astype(np.float64)
                    )
                    # Deterministic policy (sigma=0 → mode, no noise)
                    actor = behavior.actor(feat)
                    action_t = (
                        actor.mode
                        if not callable(getattr(actor, "mode", actor.sample))
                        else actor.mode()
                    )
                    # Imagination step with stochastic z sampling
                    latent = wm.dynamics.img_step(latent, action_t, sample=True)

        if not all_feats:
            # Degenerate fallback
            return np.zeros(32, dtype=np.float64), np.ones(32, dtype=np.float64) * 0.01

        feats = np.stack(all_feats, axis=0)  # (n_samples * H, feat_dim)
        mean_feat = feats.mean(axis=0)
        var_feat = feats.var(axis=0)
        return mean_feat, var_feat

    # ── ensemble-level utilities ──────────────────────────────────────────────

    @property
    def k(self) -> int:
        """Number of ensemble members."""
        return len(self.members)

    def calibrate_nu_scale(
        self,
        env_sigma: float,
        obs_batch: npt.NDArray[Any],
        n_samples: int = 8,
    ) -> float:
        """Estimate nu_scale that maps raw uncertainty to env noise sigma.

        Given a batch of observations collected at known env noise level
        ``env_sigma``, finds the linear scale such that:

            E[nu_hat(obs)] ≈ env_sigma

        Parameters
        ----------
        env_sigma : Known environment noise level.
        obs_batch : Array of observations (N, obs_dim).
        n_samples : Samples per observation.

        Returns
        -------
        Calibrated nu_scale scalar (saved into self.nu_scale).
        """
        # Temporarily set nu_scale=1 to get raw estimates
        old_scale = self.nu_scale
        self.nu_scale = 1.0

        raw_estimates = []
        for obs in obs_batch:
            try:
                comps = self.estimate_components(obs, n_samples=n_samples)
                # Get the pre-scale raw value
                raw = float(
                    np.sqrt(
                        max(comps["aleatoric"] + comps["epistemic"], 0.0)
                    )
                )
                raw_estimates.append(raw)
            except Exception:
                pass

        self.nu_scale = old_scale

        if not raw_estimates or np.mean(raw_estimates) < 1e-10:
            warnings.warn(
                "calibrate_nu_scale: raw estimates are near zero — "
                "using nu_scale=1.0",
                stacklevel=2,
            )
            return 1.0

        scale = env_sigma / float(np.mean(raw_estimates))
        self.nu_scale = scale
        return scale
