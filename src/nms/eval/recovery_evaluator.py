"""Recovery evaluator for Phase E.

Measures how quickly a policy recovers performance after an abrupt
noise increase (e.g., σ steps from 0 → 0.3 mid-episode).

Metrics:
  - recovery_steps: steps to return within 90% of pre-perturbation return
  - area_under_recovery: cumulative regret during recovery window
  - bandwidth_utilisation: fraction of CPB bandwidth actually used
"""
from __future__ import annotations

from typing import Any

import numpy as np


class RecoveryEvaluator:
    """Evaluates policy recovery after abrupt noise onset.

    Parameters
    ----------
    policy         : Any policy with sample_action(obs).
    env_factory    : Callable(sigma) → stochastic gymnasium env.
    horizon        : Total evaluation horizon H.
    switch_step    : Step at which noise switches from 0 → sigma_ood.
    sigma_ood      : OOD noise level applied after switch_step.
    n_episodes     : Number of evaluation episodes.
    recovery_tol   : Fraction of pre-switch return for "recovered" threshold.
    seed           : Base seed.
    """

    def __init__(
        self,
        policy: Any,
        env_factory: Any,
        horizon: int = 500,
        switch_step: int = 100,
        sigma_ood: float = 0.3,
        n_episodes: int = 10,
        recovery_tol: float = 0.9,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.env_factory = env_factory
        self.horizon = horizon
        self.switch_step = switch_step
        self.sigma_ood = sigma_ood
        self.n_episodes = n_episodes
        self.recovery_tol = recovery_tol
        self._rng = np.random.default_rng(seed)

    def _run_switch_episode(self) -> dict[str, Any]:
        """Single episode: first switch_step steps with σ=0, then σ=sigma_ood."""
        env_clean = self.env_factory(0.0)
        env_noisy = self.env_factory(self.sigma_ood)

        ep_seed = int(self._rng.integers(0, 2**31))
        obs, _ = env_clean.reset(seed=ep_seed)
        rewards: list[float] = []

        for t in range(self.horizon):
            if t == self.switch_step:
                # Switch to noisy env (reset position to current obs approximation)
                obs_tmp, _ = env_noisy.reset()
                obs = obs_tmp  # approximate: reset noisy env

            env = env_clean if t < self.switch_step else env_noisy
            action = self.policy.sample_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            rewards.append(float(rew))
            if term or trunc:
                break

        env_clean.close()
        env_noisy.close()
        return {"rewards": rewards, "switch_step": self.switch_step}

    def evaluate(self) -> dict[str, Any]:
        """Run n_episodes switch evaluations and compute recovery metrics."""
        all_rewards: list[list[float]] = []
        for _ in range(self.n_episodes):
            result = self._run_switch_episode()
            all_rewards.append(result["rewards"])

        # Pad to same length
        max_len = max(len(r) for r in all_rewards)
        padded = np.array([r + [r[-1]] * (max_len - len(r)) for r in all_rewards])
        mean_rew = padded.mean(axis=0)

        pre_switch = mean_rew[: self.switch_step]
        post_switch = mean_rew[self.switch_step :]
        baseline_return = float(pre_switch.mean()) if len(pre_switch) > 0 else 1.0
        threshold = self.recovery_tol * baseline_return

        # Recovery step: first step after switch where rolling avg ≥ threshold
        recovery_step = None
        window = 20
        for t in range(len(post_switch)):
            window_mean = float(post_switch[max(0, t - window) : t + 1].mean())
            if window_mean >= threshold:
                recovery_step = t
                break

        aur = float(np.maximum(baseline_return - post_switch, 0).sum())

        return {
            "mean_pre_switch_return": float(baseline_return),
            "mean_post_switch_return": float(post_switch.mean()) if len(post_switch) > 0 else 0.0,
            "recovery_step": recovery_step,
            "area_under_recovery": aur,
            "mean_rewards": mean_rew.tolist(),
            "n_episodes": self.n_episodes,
            "sigma_ood": self.sigma_ood,
            "switch_step": self.switch_step,
        }
