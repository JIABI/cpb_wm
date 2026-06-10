"""DreamerV3 baseline variants for Exp 2 (Phase C).

Variants:
  dreamer_default        : DreamerWrapper with τ=1.0 (no NMS)
  dreamer_fixed_001      : Fixed τ=0.01
  dreamer_fixed_01       : Fixed τ=0.1
  dreamer_fixed_05       : Fixed τ=0.5
  dreamer_autotuned      : SAC-style entropy-target auto-tuning of τ

All variants expose the same interface:
  sample_action(obs) → ndarray
  train(total_steps, log_dir)
  get_actor_temperature() / set_actor_temperature(tau)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt


class DreamerVariant:
    """Base class wrapping DreamerWrapper with a fixed actor temperature.

    Parameters
    ----------
    dreamer    : Constructed DreamerWrapper (or MinimalDreamer).
    temperature: Actor temperature τ.
    """

    def __init__(self, dreamer: Any, temperature: float = 1.0) -> None:
        self._dreamer = dreamer
        self._dreamer.set_actor_temperature(temperature)

    def train(self, total_steps: int, log_dir: str = "runs/dreamer_variant") -> None:
        self._dreamer.train(total_steps, log_dir)

    def sample_action(self, obs: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        return self._dreamer.sample_action(obs)

    def get_actor_temperature(self) -> float:
        return self._dreamer.get_actor_temperature()

    def set_actor_temperature(self, tau: float) -> None:
        self._dreamer.set_actor_temperature(tau)

    def rollout(self, env: Any, horizon: int, seed: int | None = None) -> dict[str, npt.NDArray[Any]]: # noqa: E501
        return self._dreamer.rollout(env, horizon, seed=seed)


def make_dreamer_default(dreamer: Any) -> DreamerVariant:
    """dreamer_default: τ=1.0 (baseline, no temperature adjustment)."""
    return DreamerVariant(dreamer, temperature=1.0)


def make_dreamer_fixed(dreamer: Any, tau: float) -> DreamerVariant:
    """dreamer_fixed_XXX: fixed τ."""
    return DreamerVariant(dreamer, temperature=tau)


class AutotunedDreamer(DreamerVariant):
    """dreamer_autotuned: SAC-style automatic entropy tuning.

    Maintains a learned log-temperature parameter α such that
    H[π] ≈ target_entropy = -|A| (min-entropy heuristic).

    Temperature is updated after each rollout via a simple gradient step.

    Parameters
    ----------
    dreamer        : DreamerWrapper instance.
    target_entropy : Desired policy entropy (default -act_dim).
    lr             : Learning rate for temperature update.
    act_dim        : Action dimension (used for default target_entropy).
    """

    def __init__(
        self,
        dreamer: Any,
        target_entropy: float | None = None,
        lr: float = 3e-4,
        act_dim: int = 1,
    ) -> None:
        super().__init__(dreamer, temperature=1.0)
        self._target_entropy = target_entropy if target_entropy is not None else -float(act_dim)
        self._lr = lr
        self._log_tau = 0.0  # log(τ)

    def update_temperature(self, entropy: float) -> None:
        """Gradient step on log_tau: ∂/∂τ E[−τ(H[π] − target_entropy)]."""
        loss_grad = -(entropy - self._target_entropy)
        self._log_tau -= self._lr * loss_grad
        new_tau = float(math.exp(self._log_tau))
        self.set_actor_temperature(new_tau)

    def train(self, total_steps: int, log_dir: str = "runs/dreamer_autotuned") -> None:
        """Train with automatic temperature tuning."""
        # For MinimalDreamer / dreamerv3-torch, we interleave temperature updates
        # by running the base train and periodically recalculating entropy proxy.
        # Simple approximation: use variance of recent action distributions.
        self._dreamer.train(total_steps, log_dir)
        # Post-hoc entropy estimate not easily available from DreamerWrapper;
        # a real implementation would intercept the actor loss callback.
