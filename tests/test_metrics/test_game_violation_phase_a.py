"""Tests for Phase A violation functions: exploitation_spiral,
payoff_loss_violation, composite_violation."""
from __future__ import annotations

import numpy as np
import pytest

from nms.envs.noisy_games import C, D, IPDTrajectory
from nms.metrics.game_violation import (
    composite_violation,
    exploitation_spiral,
    payoff_loss_violation,
)


def _make_traj(
    a1: list[int],
    a2: list[int],
    r1: list[float] | None = None,
) -> IPDTrajectory:
    h = len(a1)
    a1_arr = np.array(a1, dtype=np.int_)
    a2_arr = np.array(a2, dtype=np.int_)
    if r1 is None:
        r1_arr = np.zeros(h, dtype=np.float64)
    else:
        r1_arr = np.array(r1, dtype=np.float64)
    return IPDTrajectory(
        a1=a1_arr,
        a2=a2_arr,
        obs1=a2_arr.copy(),
        obs2=a1_arr.copy(),
        r1=r1_arr,
        r2=np.zeros(h, dtype=np.float64),
        false_obs1=np.zeros(h, dtype=np.bool_),
        false_obs2=np.zeros(h, dtype=np.bool_),
        nu=0.0,
    )


# ── exploitation_spiral ──────────────────────────────────────────────────────

def test_exploitation_spiral_fires() -> None:
    """5 consecutive (C, D) steps → 1."""
    traj = _make_traj([C, C, C, C, C], [D, D, D, D, D])
    assert exploitation_spiral(traj, horizon=5, min_len=5) == 1


def test_exploitation_spiral_too_short() -> None:
    """4 consecutive (C, D) steps → 0 when min_len=5."""
    traj = _make_traj([C, C, C, C, D], [D, D, D, D, D])
    assert exploitation_spiral(traj, horizon=5, min_len=5) == 0


def test_exploitation_spiral_no_exploitation() -> None:
    """All (C, C) — no exploitation."""
    traj = _make_traj([C] * 10, [C] * 10)
    assert exploitation_spiral(traj, horizon=10, min_len=3) == 0


def test_exploitation_spiral_mutual_defection_not_exploitation() -> None:
    """(D, D) is not exploitation (only (C, D) counts)."""
    traj = _make_traj([D] * 5, [D] * 5)
    assert exploitation_spiral(traj, horizon=5, min_len=3) == 0


def test_exploitation_spiral_horizon_truncation() -> None:
    """Exploitation run occurs after the horizon window — should not fire."""
    # First 5 steps: normal; exploitation at steps 5-9
    a1 = [C] * 5 + [C, C, C, C, C]
    a2 = [C] * 5 + [D, D, D, D, D]
    traj = _make_traj(a1, a2)
    assert exploitation_spiral(traj, horizon=5, min_len=3) == 0
    assert exploitation_spiral(traj, horizon=10, min_len=3) == 1


# ── payoff_loss_violation ─────────────────────────────────────────────────────

def test_payoff_loss_fires_below_threshold() -> None:
    traj = _make_traj([C] * 5, [D] * 5, r1=[0.0] * 5)
    assert payoff_loss_violation(traj, horizon=5, safe_threshold=2.0) == 1


def test_payoff_loss_no_fire_above_threshold() -> None:
    traj = _make_traj([C] * 5, [C] * 5, r1=[3.0] * 5)
    assert payoff_loss_violation(traj, horizon=5, safe_threshold=2.0) == 0


def test_payoff_loss_at_threshold() -> None:
    """Exactly at threshold: mean return == safe_threshold → no violation (not <)."""
    traj = _make_traj([C] * 4, [C] * 4, r1=[2.0] * 4)
    assert payoff_loss_violation(traj, horizon=4, safe_threshold=2.0) == 0


# ── composite_violation ───────────────────────────────────────────────────────

def test_composite_spiral_only() -> None:
    """spiral_only fires on mutual-defection spiral, not exploitation."""
    a1_d = [D] * 5
    a2_d = [D] * 5
    traj_dd = _make_traj(a1_d, a2_d)
    traj_cd = _make_traj([C] * 5, [D] * 5)
    assert composite_violation(traj_dd, 5, mode="spiral_only", spiral_len=5) == 1
    assert composite_violation(traj_cd, 5, mode="spiral_only", spiral_len=5) == 0


def test_composite_exploit_only() -> None:
    """exploit_only fires on exploitation, not mutual-defection."""
    traj_dd = _make_traj([D] * 5, [D] * 5)
    traj_cd = _make_traj([C] * 5, [D] * 5)
    assert composite_violation(traj_dd, 5, mode="exploit_only", exploit_len=5) == 0
    assert composite_violation(traj_cd, 5, mode="exploit_only", exploit_len=5) == 1


def test_composite_spiral_or_exploit_fires_on_either() -> None:
    traj_dd = _make_traj([D] * 5, [D] * 5)
    traj_cd = _make_traj([C] * 5, [D] * 5)
    traj_cc = _make_traj([C] * 5, [C] * 5)
    assert composite_violation(traj_dd, 5, mode="spiral_or_exploit",
                               spiral_len=5, exploit_len=5) == 1
    assert composite_violation(traj_cd, 5, mode="spiral_or_exploit",
                               spiral_len=5, exploit_len=5) == 1
    assert composite_violation(traj_cc, 5, mode="spiral_or_exploit",
                               spiral_len=5, exploit_len=5) == 0


def test_composite_unknown_mode_raises() -> None:
    traj = _make_traj([C] * 3, [C] * 3)
    with pytest.raises(ValueError, match="Unknown violation mode"):
        composite_violation(traj, 3, mode="nonexistent_mode")
