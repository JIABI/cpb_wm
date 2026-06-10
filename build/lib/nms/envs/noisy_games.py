"""Noisy Iterated Prisoner's Dilemma (IPD) environment.

Misperception noise is bilateral and symmetric: at every step each player
independently flips their perception of the opponent's *true* action with
probability ν.  Trajectories keep both the true and the observed action
sequences so that false-retaliation diagnostics can be computed downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# Action constants
C: int = 0  # Cooperate
D: int = 1  # Defect

# Standard IPD payoff matrix  (action_player1, action_player2) -> (r1, r2)
PAYOFF_IPD: dict[tuple[int, int], tuple[float, float]] = {
    (C, C): (3.0, 3.0),
    (C, D): (0.0, 5.0),
    (D, C): (5.0, 0.0),
    (D, D): (1.0, 1.0),
}


@dataclass
class IPDTrajectory:
    """One H-step rollout of the noisy IPD.

    All arrays have shape ``(H,)``.

    Attributes
    ----------
    a1, a2       : True actions of player 1 / player 2 (C=0 or D=1).
    obs1, obs2   : What each player *observed* (opponent's action, possibly flipped).
    r1, r2       : Per-step payoffs.
    false_obs1   : Boolean mask — True when player 1's observation was a misperception
                   (obs1[t] != a2[t]).
    false_obs2   : Same for player 2.
    nu           : The noise rate used for this rollout.
    """

    a1: npt.NDArray[np.int_]
    a2: npt.NDArray[np.int_]
    obs1: npt.NDArray[np.int_]
    obs2: npt.NDArray[np.int_]
    r1: npt.NDArray[np.float64]
    r2: npt.NDArray[np.float64]
    false_obs1: npt.NDArray[np.bool_]
    false_obs2: npt.NDArray[np.bool_]
    nu: float


class _PolicyProtocol:
    """Minimal duck-type interface for IPD policies."""

    def act(self, last_obs_opponent: int, rng: np.random.Generator) -> int:
        raise NotImplementedError


def rollout_ipd(
    policy1: _PolicyProtocol,
    policy2: _PolicyProtocol,
    horizon: int,
    nu: float,
    rng: np.random.Generator,
) -> IPDTrajectory:
    """Run one H-step episode of the noisy IPD.

    Both players act *simultaneously* based on the previous **observed**
    opponent action.  At t=0 the prior is C for both sides.

    Parameters
    ----------
    policy1, policy2 : Objects with an ``act(last_obs_opponent, rng) -> int``
        method.  Stateless policies (e.g. SoftReciprocalPolicy) are safe to
        reuse across episodes.  Stateful policies (e.g. WSLS) must be reset
        externally between episodes.
    horizon : int — episode length H.
    nu      : float in [0, 1] — misperception probability per step per player.
    rng     : numpy Generator (from ``np.random.default_rng``).
    """
    a1 = np.zeros(horizon, dtype=np.int_)
    a2 = np.zeros(horizon, dtype=np.int_)
    obs1 = np.zeros(horizon, dtype=np.int_)
    obs2 = np.zeros(horizon, dtype=np.int_)
    r1 = np.zeros(horizon, dtype=np.float64)
    r2 = np.zeros(horizon, dtype=np.float64)
    false1 = np.zeros(horizon, dtype=np.bool_)
    false2 = np.zeros(horizon, dtype=np.bool_)

    prev_obs1: int = C  # initial prior: both sides assume opponent cooperated
    prev_obs2: int = C

    for t in range(horizon):
        a1[t] = policy1.act(prev_obs1, rng)
        a2[t] = policy2.act(prev_obs2, rng)

        r1[t], r2[t] = PAYOFF_IPD[(int(a1[t]), int(a2[t]))]

        # Independent bilateral misperception noise
        flip1 = rng.random() < nu
        flip2 = rng.random() < nu
        obs1[t] = (1 - a2[t]) if flip1 else a2[t]
        obs2[t] = (1 - a1[t]) if flip2 else a1[t]
        false1[t] = flip1
        false2[t] = flip2

        prev_obs1, prev_obs2 = int(obs1[t]), int(obs2[t])

    return IPDTrajectory(
        a1=a1, a2=a2, obs1=obs1, obs2=obs2,
        r1=r1, r2=r2,
        false_obs1=false1, false_obs2=false2,
        nu=nu,
    )


def rollout_ipd_with_mixture(
    agent_policy: Any,
    opponent_mixture: dict[Any, float],
    horizon: int,
    nu: float,
    rng: np.random.Generator,
) -> IPDTrajectory:
    """Run one H-step episode of the noisy IPD against a mixture of opponents.

    At the **start of each episode** a single opponent is sampled from
    ``opponent_mixture`` according to its probability weight.  The sampled
    opponent then plays the full episode against ``agent_policy``.

    This is the correct abstraction for Phase A evaluation: the bandwidth
    R_H(σ; ν) is the *expected* violation rate over the mixture distribution,
    not the violation rate for any single opponent type.

    Parameters
    ----------
    agent_policy     : Policy object with ``act(last_obs_opponent, rng) -> int``.
                       Assigned to player 1.
    opponent_mixture : {policy_object: weight} dict (probabilities need not sum
                       to exactly 1 — they are normalised internally).
    horizon          : Episode length H.
    nu               : Misperception probability per step per player.
    rng              : numpy Generator (``np.random.default_rng``).

    Returns
    -------
    IPDTrajectory — identical format to ``rollout_ipd``.
    """
    from nms.policies.opponents import sample_opponent  # noqa: PLC0415

    opponent = sample_opponent(rng, opponent_mixture)
    return rollout_ipd(agent_policy, opponent, horizon=horizon, nu=nu, rng=rng)
