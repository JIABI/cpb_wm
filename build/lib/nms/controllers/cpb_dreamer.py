"""CPB (Certified Policy Bandwidth) controller for DreamerV3 (Phase B+).

Calibrates the NMS bandwidth B_{H,α}(ν) via paired imagined/realized rollouts
at each σ, then at inference clips m̂(ν̂) to the certified bandwidth.

The CPB controller wraps a DreamerWrapper and a KMeansActionDiscretizer.
It does NOT modify NMSOperator.

Usage::

    ctrl = CPBDreamerController(dreamer, kmeans, alpha=0.1, horizon=15)
    ctrl.calibrate(env, nu_hat=0.1, sigma_grid=np.linspace(0, 0.5, 11), n_pairs=200)
    action = ctrl.select_action(obs, nu_hat=0.1)
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt


class CPBDreamerController:
    """Wraps DreamerWrapper with CPB bandwidth calibration.

    Parameters
    ----------
    dreamer      : Trained DreamerWrapper (or MinimalDreamer) instance.
    kmeans       : Fitted KMeansActionDiscretizer.
    alpha        : Violation rate threshold α (default 0.1).
    horizon      : Imagination/rollout horizon H (default 15).
    c_cal        : Optional pre-computed c_cal scalar from Exp 1 calibration.
                   If provided, used as fallback when bandwidth is uncalibrated.
    """

    def __init__(
        self,
        dreamer: Any,
        kmeans: Any,
        alpha: float = 0.1,
        horizon: int = 15,
        c_cal: float | None = None,
    ) -> None:
        self.dreamer = dreamer
        self.kmeans = kmeans
        self.alpha = alpha
        self.horizon = horizon
        self.c_cal = c_cal

        # Calibrated bandwidth: sigma_max → {nu: sigma_certified}
        self._bandwidth: dict[float, float] = {}
        self._sigma_grid: npt.NDArray[np.float64] | None = None
        self._violation_rates: dict[float, float] = {}  # sigma → empirical R_H

    # ── calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        env: Any,
        nu_hat: float,
        sigma_grid: npt.NDArray[np.float64],
        n_pairs: int = 200,
        seed: int = 0,
    ) -> None:
        """Calibrate bandwidth B_{H,α}(ν̂) at a fixed ν̂.

        For each σ in sigma_grid, collects n_pairs of (imagined, realized) rollouts
        and computes the empirical violation rate R_H(σ; ν̂).

        The certified bandwidth is the largest σ with R_H ≤ α.

        Parameters
        ----------
        env        : gymnasium-compatible environment (used for realized rollouts).
        nu_hat     : Estimated noise level ν̂ (used for stochastic env).
        sigma_grid : Array of σ values to sweep.
        n_pairs    : Number of rollout pairs per σ.
        seed       : Base seed.
        """
        from nms.metrics.wm_violation import batch_violation_rate

        self._sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
        violation_rates = {}
        rng = np.random.default_rng(seed)

        for sigma in self._sigma_grid:
            imagined_batch = []
            realized_batch = []
            for _ in range(n_pairs):
                pair_seed = int(rng.integers(0, 2**31))
                obs, _ = env.reset(seed=pair_seed)

                img = self.dreamer.rollout_imagined(  # noqa: E501
                    obs, self.horizon, sigma=float(sigma), seed=pair_seed)
                rea = self.dreamer.rollout_realized(  # noqa: E501
                    env, self.horizon, sigma=float(sigma), seed=pair_seed)
                imagined_batch.append(img)
                realized_batch.append(rea)

            r_h = batch_violation_rate(imagined_batch, realized_batch, self.kmeans, self.horizon)
            violation_rates[float(sigma)] = r_h
            print(f"  σ={sigma:.3f} → R_H={r_h:.4f}")

        self._violation_rates = violation_rates

        # Certified bandwidth: largest σ with R_H ≤ α
        safe_sigmas = [s for s, r in violation_rates.items() if r <= self.alpha]
        if safe_sigmas:
            self._bandwidth[nu_hat] = max(safe_sigmas)
        else:
            self._bandwidth[nu_hat] = 0.0
            warnings.warn(
                f"No safe σ found for ν̂={nu_hat:.3f} with α={self.alpha}. "
                "Bandwidth set to 0.",
                stacklevel=2,
            )

        print(f"  B_{{H,α}}(ν̂={nu_hat:.3f}) = {self._bandwidth[nu_hat]:.4f}")

    def get_bandwidth(self, nu_hat: float) -> float:
        """Return calibrated bandwidth B_{H,α}(ν̂) for a given ν̂.

        Falls back to c_cal * nu_hat if calibration has not been run.
        """
        if nu_hat in self._bandwidth:
            return self._bandwidth[nu_hat]

        # Linear interpolation over calibrated nu values
        if len(self._bandwidth) >= 2:
            nus = np.array(sorted(self._bandwidth.keys()))
            bws = np.array([self._bandwidth[n] for n in nus])
            return float(np.interp(nu_hat, nus, bws))

        if self.c_cal is not None:
            return float(self.c_cal * nu_hat)

        warnings.warn(
            "Bandwidth not calibrated for ν̂={nu_hat:.3f} and no c_cal provided. "
            "Returning σ=0.",
            stacklevel=2,
        )
        return 0.0

    # ── inference ─────────────────────────────────────────────────────────────

    def select_action(
        self,
        obs: npt.NDArray[Any],
        nu_hat: float,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Select an action using CPB-projected actor temperature.

        Computes σ_CPB = clip(B_{H,α}(ν̂), σ_min, σ_max), sets the actor
        temperature accordingly, samples an action, then restores temperature.

        Parameters
        ----------
        obs       : Current observation.
        nu_hat    : Estimated environmental noise ν̂.
        sigma_min : Minimum σ (clips below).
        sigma_max : Maximum σ (clips above).

        Returns
        -------
        Continuous action from actor at σ_CPB.
        """
        sigma_cpb = float(np.clip(self.get_bandwidth(nu_hat), sigma_min, sigma_max))
        prev_tau = self.dreamer.get_actor_temperature()
        self.dreamer.set_actor_temperature(sigma_cpb)
        action = self.dreamer.sample_action(obs)
        self.dreamer.set_actor_temperature(prev_tau)
        return action

    # ── full calibration across multiple ν values ─────────────────────────────

    def calibrate_multi_nu(
        self,
        env_factory: Any,
        nu_values: list[float],
        sigma_grid: npt.NDArray[np.float64],
        n_pairs: int = 200,
        seed: int = 0,
    ) -> dict[float, float]:
        """Calibrate bandwidth for multiple ν values.

        Parameters
        ----------
        env_factory : Callable(nu) → stochastic gymnasium env at that noise level.
        nu_values   : List of ν̂ values to calibrate for.
        sigma_grid  : σ sweep grid (same for all ν).
        n_pairs     : Number of rollout pairs per (σ, ν).
        seed        : Base seed.

        Returns
        -------
        Dict mapping ν → certified bandwidth.
        """
        for nu in nu_values:
            env = env_factory(nu)
            self.calibrate(env, nu_hat=nu, sigma_grid=sigma_grid, n_pairs=n_pairs, seed=seed)
            env.close()
        return dict(self._bandwidth)

    @property
    def violation_rates(self) -> dict[float, float]:
        """Read-only view of calibrated σ → R_H mapping."""
        return dict(self._violation_rates)

    @property
    def bandwidth_dict(self) -> dict[float, float]:
        """Read-only view of calibrated ν → bandwidth mapping."""
        return dict(self._bandwidth)
