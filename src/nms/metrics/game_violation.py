"""Game-theoretic violation and diagnostic metrics for Exp 1 (noisy IPD).

All functions accept an IPDTrajectory and a horizon so they can be
composed with the core NMS ViolationFn interface.
"""
from __future__ import annotations

import numpy as np

from nms.envs.noisy_games import C, D, IPDTrajectory


def mutual_defection_spiral(
    traj: IPDTrajectory,
    horizon: int,
    min_len: int = 5,
) -> int:
    """Return 1 iff a run of ≥ min_len consecutive (D,D) steps occurs in [0, horizon).

    This is the primary violation indicator for Exp 1.  A 'spiral' captures the
    retaliation cascade that destroys cooperation under misperception noise.

    Parameters
    ----------
    traj     : IPDTrajectory from rollout_ipd.
    horizon  : Only the first horizon steps are evaluated.
    min_len  : Minimum consecutive mutual-defection steps to count as a spiral.
    """
    h_eff = min(horizon, len(traj.a1))
    if h_eff < min_len:
        return 0
    both_d = (traj.a1[:h_eff] == D) & (traj.a2[:h_eff] == D)
    run = 0
    for x in both_d:
        run = (run + 1) if x else 0
        if run >= min_len:
            return 1
    return 0


def cooperation_rate(traj: IPDTrajectory, horizon: int) -> float:
    """Fraction of steps in [0, horizon) where both players cooperate."""
    h_eff = min(horizon, len(traj.a1))
    return float(np.mean((traj.a1[:h_eff] == 0) & (traj.a2[:h_eff] == 0)))


def false_retaliation_rate(traj: IPDTrajectory, horizon: int) -> float:
    """Fraction of player-1 defections that follow a *false* observation.

    Computed as: among all t ∈ [1, horizon) where a1[t] == D, what fraction
    had false_obs1[t-1] == True?

    This is a diagnostic for how much σ=0 (no forgiveness) hurts under noise.
    """
    h_eff = min(horizon, len(traj.a1))
    if h_eff < 2:
        return 0.0
    d_steps = traj.a1[1:h_eff] == D           # player 1 defects at steps 1..h-1
    false_prev = traj.false_obs1[: h_eff - 1]  # false obs at steps 0..h-2
    if not d_steps.any():
        return 0.0
    return float(np.mean(false_prev[d_steps]))


def mean_return(traj: IPDTrajectory, horizon: int) -> float:
    """Mean per-step payoff for player 1 over [0, horizon).

    In symmetric self-play E[r1] == E[r2] in expectation, so this proxy
    serves as the value function V(σ) in Theorem 2.
    """
    h_eff = min(horizon, len(traj.a1))
    return float(np.mean(traj.r1[:h_eff]))


# ── Phase A.3: high-σ and composite violation indicators ────────────────────

def exploitation_spiral(
    traj: IPDTrajectory,
    horizon: int,
    min_len: int = 5,
) -> int:
    """Return 1 iff player 1 is exploited for ≥ min_len consecutive steps.

    'Exploitation' is defined as agent 1 cooperating while agent 2 defects
    (C, D) for min_len or more consecutive steps.  This captures the high-σ
    failure mode: an overly forgiving agent is taken advantage of by an
    unconditional defector in the opponent mixture.

    Parameters
    ----------
    traj     : IPDTrajectory from rollout_ipd or rollout_ipd_with_mixture.
    horizon  : Only the first horizon steps are evaluated.
    min_len  : Minimum consecutive exploitation steps to count as a spiral.
    """
    h_eff = min(horizon, len(traj.a1))
    if h_eff < min_len:
        return 0
    exploited = (traj.a1[:h_eff] == C) & (traj.a2[:h_eff] == D)
    run = 0
    for x in exploited:
        run = (run + 1) if x else 0
        if run >= min_len:
            return 1
    return 0


def payoff_loss_violation(
    traj: IPDTrajectory,
    horizon: int,
    safe_threshold: float = 2.0,
) -> int:
    """Return 1 iff player 1's mean payoff is below safe_threshold.

    The default threshold of 2.0 sits between mutual-defection (1.0) and
    mutual-cooperation (3.0) payoffs, so violation fires when the agent
    consistently does worse than mid-range.

    Parameters
    ----------
    traj            : IPDTrajectory.
    horizon         : Only the first horizon steps are evaluated.
    safe_threshold  : Lower bound on acceptable mean payoff.
    """
    return int(mean_return(traj, horizon) < safe_threshold)


def composite_violation(
    traj: IPDTrajectory,
    horizon: int,
    mode: str = "spiral_or_exploit",
    spiral_len: int = 5,
    exploit_len: int = 5,
    safe_threshold: float = 2.0,
) -> int:
    """Combined violation indicator for Phase A bandwidth sweep.

    Modes
    -----
    "spiral_only"       : Only mutual-defection spiral (low-σ failure).
    "exploit_only"      : Only exploitation spiral (high-σ failure).
    "spiral_or_exploit" : Either spiral fires (U-shaped bandwidth curves).
    "spiral_or_payoff_loss" : Mutual-defection spiral OR payoff loss.

    Parameters
    ----------
    traj           : IPDTrajectory.
    horizon        : Episode length used for evaluation.
    mode           : One of the mode strings above.
    spiral_len     : min_len forwarded to mutual_defection_spiral.
    exploit_len    : min_len forwarded to exploitation_spiral.
    safe_threshold : Forwarded to payoff_loss_violation.
    """
    if mode == "spiral_only":
        return mutual_defection_spiral(traj, horizon, min_len=spiral_len)
    if mode == "exploit_only":
        return exploitation_spiral(traj, horizon, min_len=exploit_len)
    if mode == "spiral_or_exploit":
        return int(
            bool(mutual_defection_spiral(traj, horizon, min_len=spiral_len))
            or bool(exploitation_spiral(traj, horizon, min_len=exploit_len))
        )
    if mode == "spiral_or_payoff_loss":
        return int(
            bool(mutual_defection_spiral(traj, horizon, min_len=spiral_len))
            or bool(payoff_loss_violation(traj, horizon, safe_threshold=safe_threshold))
        )
    raise ValueError(
        f"Unknown violation mode {mode!r}. "
        "Choose from: spiral_only, exploit_only, spiral_or_exploit, spiral_or_payoff_loss."
    )
