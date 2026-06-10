"""Reciprocal policies for the noisy IPD experiment (Exp 1).

Physical meaning of σ in this experiment
-----------------------------------------
σ is the *forgiveness probability*: when a player observes the opponent
defecting, they cooperate anyway with probability σ.

  σ = 0  →  deterministic Tit-for-Tat (mirror every observation)
  σ = 1  →  always cooperate (unconditionally cooperative, easily exploited)

This σ is the *interface stochasticity parameter* that NMS certifies.  Its
physical meaning differs across experiments (σ = softmax temperature in Exp 2,
belief-noise amplitude in Exp 4), but the certified object B_{H,α}(ν) and the
projection operator Π_B remain the same throughout the paper.
"""
from __future__ import annotations

import numpy as np

from nms.envs.noisy_games import C, D


class DeterministicTFT:
    """Classic Tit-for-Tat: copy the opponent's last observed action."""

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:  # noqa: ARG002
        return D if last_obs_opponent == D else C


class WSLS:
    """Win-Stay-Lose-Shift (Pavlov).

    Cooperate after matching outcomes (CC or DD), defect after mismatches.

    Warning
    -------
    This policy is **stateful** — it remembers the previous own action.
    Create a fresh instance for each episode; do *not* reuse across episodes.
    """

    def __init__(self) -> None:
        self._last_own: int = C

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:  # noqa: ARG002
        # Win-stay: if last actions matched → stay; else shift
        match = self._last_own == last_obs_opponent
        action = C if match else D
        self._last_own = action
        return action


class GenerousTFT:
    """TFT with a fixed forgiveness probability σ — used as a fixed-σ baseline."""

    def __init__(self, sigma: float) -> None:
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"sigma must be in [0, 1], got {sigma}")
        self.sigma = sigma

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:
        if last_obs_opponent == C:
            return C
        return C if rng.random() < self.sigma else D


class SoftReciprocalPolicy:
    """Main Exp 1 policy — σ is the forgiveness probability.

    Stateless: safe to reuse across episodes.
    """

    def __init__(self, sigma: float) -> None:
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"sigma must be in [0, 1], got {sigma}")
        self.sigma = sigma

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:
        if last_obs_opponent == C:
            return C
        # Opponent defected (possibly due to misperception) — forgive with prob σ
        return C if rng.random() < self.sigma else D
