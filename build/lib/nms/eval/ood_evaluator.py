"""OOD (Out-of-Distribution) evaluator for Phase D.

Tests how a trained policy performs when physics parameters shift
beyond the training distribution via PhysicsParameterShiftWrapper.

Metrics:
  - mean episodic return under shifted physics
  - performance collapse: (return_train - return_ood) / return_train
  - safety: fraction of episodes within certified bandwidth
"""
from __future__ import annotations

from typing import Any

import numpy as np


class OODEvaluator:
    """Evaluates a policy under OOD physics parameter shifts.

    Parameters
    ----------
    policy      : Any policy with sample_action(obs).
    base_env    : Nominal (non-shifted) gymnasium-compatible environment.
    alpha       : Violation rate threshold for safety metric.
    n_episodes  : Number of evaluation episodes per shift level.
    max_steps   : Max steps per episode.
    seed        : Base RNG seed.
    """

    def __init__(
        self,
        policy: Any,
        base_env: Any,
        alpha: float = 0.1,
        n_episodes: int = 10,
        max_steps: int = 1000,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.base_env = base_env
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

    def _run_episodes(self, env: Any) -> list[float]:
        returns = []
        for _ in range(self.n_episodes):
            obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31)))
            ep_ret = 0.0
            for _ in range(self.max_steps):
                action = self.policy.sample_action(obs)
                obs, rew, term, trunc, _ = env.step(action)
                ep_ret += float(rew)
                if term or trunc:
                    break
            returns.append(ep_ret)
        return returns

    def evaluate_nominal(self) -> dict[str, Any]:
        """Evaluate on nominal (training) environment."""
        returns = self._run_episodes(self.base_env)
        return {
            "shift": "nominal",
            "mass_mult": 1.0,
            "friction_mult": 1.0,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "returns": returns,
        }

    def evaluate_shifted(
        self,
        mass_mult: float = 1.0,
        friction_mult: float = 1.0,
        damping_mult: float = 1.0,
    ) -> dict[str, Any]:
        """Evaluate on physics-shifted environment.

        Creates a copy of base_env wrapped with PhysicsParameterShiftWrapper.
        """
        from nms.envs.stochastic_dm_control import PhysicsParameterShiftWrapper

        shifted_env = PhysicsParameterShiftWrapper(
            self.base_env,
            mass_mult=mass_mult,
            friction_mult=friction_mult,
            damping_mult=damping_mult,
        )
        returns = self._run_episodes(shifted_env)
        return {
            "shift": f"m{mass_mult:.1f}_f{friction_mult:.1f}_d{damping_mult:.1f}",
            "mass_mult": mass_mult,
            "friction_mult": friction_mult,
            "damping_mult": damping_mult,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "returns": returns,
        }

    def sweep_mass(
        self,
        mult_values: list[float],
    ) -> list[dict[str, Any]]:
        """Sweep over mass multipliers; returns list of result dicts."""
        results = []
        for m in mult_values:
            r = self.evaluate_shifted(mass_mult=m)
            results.append(r)
            print(f"  mass_mult={m:.2f} → return={r['mean_return']:.2f} ± {r['std_return']:.2f}")
        return results

    def collapse_fraction(
        self,
        nominal_return: float,
        shifted_returns: list[float],
    ) -> float:
        """Performance collapse = (R_nominal - R_ood) / max(R_nominal, 1e-6)."""
        r_ood = float(np.mean(shifted_returns))
        return (nominal_return - r_ood) / max(abs(nominal_return), 1e-6)
