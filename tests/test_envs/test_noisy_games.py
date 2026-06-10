"""Unit tests for noisy_games.py environment."""
from __future__ import annotations

import numpy as np
import pytest

from nms.envs.noisy_games import C, rollout_ipd
from nms.metrics.game_violation import mutual_defection_spiral
from nms.policies.reciprocal import DeterministicTFT, SoftReciprocalPolicy


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── zero-noise correctness ──────────────────────────────────────────────────

def test_zero_noise_no_false_observations() -> None:
    """With ν=0 there must be zero misperceptions."""
    rng = _rng(42)
    traj = rollout_ipd(DeterministicTFT(), DeterministicTFT(), horizon=50, nu=0.0, rng=rng)
    assert not traj.false_obs1.any(), "false_obs1 should be all-False at nu=0"
    assert not traj.false_obs2.any(), "false_obs2 should be all-False at nu=0"


def test_zero_noise_obs_equals_true_action() -> None:
    """With ν=0 every player's observation equals the opponent's true action."""
    rng = _rng(7)
    p = SoftReciprocalPolicy(0.3)
    traj = rollout_ipd(p, SoftReciprocalPolicy(0.3), horizon=100, nu=0.0, rng=rng)
    np.testing.assert_array_equal(traj.obs1, traj.a2,
                                  err_msg="obs1 must equal a2 at nu=0")
    np.testing.assert_array_equal(traj.obs2, traj.a1,
                                  err_msg="obs2 must equal a1 at nu=0")


def test_two_tft_zero_noise_always_cooperate() -> None:
    """Two deterministic TFT agents with ν=0 should cooperate every step."""
    rng = _rng(1)
    traj = rollout_ipd(DeterministicTFT(), DeterministicTFT(), horizon=30, nu=0.0, rng=rng)
    assert (traj.a1 == C).all(), "player 1 should always cooperate"
    assert (traj.a2 == C).all(), "player 2 should always cooperate"


# ── high-noise behaviour ────────────────────────────────────────────────────

def test_two_tft_high_noise_likely_spiral() -> None:
    """Two deterministic TFT agents at ν=0.5 should frequently fall into spirals.

    At ν=0.5 observations are essentially random (50% flip probability), so
    (D,D) steps occur with P≈0.25 independently.  Runs of min_len=3 appear in
    ~70% of 100-step episodes.  We use min_len=3 with a conservative 50% threshold.

    This test verifies that the violation detector works and that DeterministicTFT
    is fragile under symmetric misperception noise — which is the motivation for
    NMS certified stochasticity.
    """
    rng = _rng(0)
    episodes = 300
    spiral_count = 0
    for _ in range(episodes):
        traj = rollout_ipd(
            DeterministicTFT(), DeterministicTFT(), horizon=100, nu=0.5, rng=rng
        )
        spiral_count += mutual_defection_spiral(traj, horizon=100, min_len=3)
    rate = spiral_count / episodes
    assert rate > 0.5, (
        f"Expected spiral rate > 50% at nu=0.5, min_len=3 but got {rate:.2%}. "
        "Check DeterministicTFT or mutual_defection_spiral."
    )


# ── SoftReciprocalPolicy boundary cases ────────────────────────────────────

def test_soft_reciprocal_sigma1_always_cooperates() -> None:
    """σ=1.0 means unconditional cooperation regardless of observation."""
    rng = _rng(3)
    p = SoftReciprocalPolicy(sigma=1.0)
    # Force many D observations by pairing with AllD (sigma=0 opponent)
    traj = rollout_ipd(p, SoftReciprocalPolicy(sigma=0.0), horizon=80, nu=0.0, rng=rng)
    # Player 1 (sigma=1) should always cooperate
    assert (traj.a1 == C).all(), "sigma=1.0 must always cooperate"


def test_soft_reciprocal_sigma0_equivalent_to_tft() -> None:
    """σ=0.0 should be functionally equivalent to DeterministicTFT.

    We verify this by checking that SoftReciprocalPolicy(0.0) and
    DeterministicTFT produce the same action sequence when given the same
    observation sequence (constructed via a shared rng path).
    """
    # Use zero-noise self-play so the observation sequence is deterministic
    rng_a = _rng(99)
    rng_b = _rng(99)
    opponent_a = SoftReciprocalPolicy(sigma=0.5)
    opponent_b = SoftReciprocalPolicy(sigma=0.5)

    traj_srp = rollout_ipd(SoftReciprocalPolicy(0.0), opponent_a,
                           horizon=50, nu=0.0, rng=rng_a)
    traj_tft = rollout_ipd(DeterministicTFT(), opponent_b,
                           horizon=50, nu=0.0, rng=rng_b)

    np.testing.assert_array_equal(
        traj_srp.a1, traj_tft.a1,
        err_msg="SoftReciprocalPolicy(0) and DeterministicTFT must agree at nu=0",
    )


def test_invalid_sigma_raises() -> None:
    with pytest.raises(ValueError):
        SoftReciprocalPolicy(sigma=1.5)
    with pytest.raises(ValueError):
        SoftReciprocalPolicy(sigma=-0.1)


# ── trajectory shape sanity ─────────────────────────────────────────────────

def test_trajectory_array_shapes() -> None:
    horizon = 17
    rng = _rng(5)
    traj = rollout_ipd(
        SoftReciprocalPolicy(0.2), SoftReciprocalPolicy(0.2), horizon=horizon, nu=0.1, rng=rng
    )
    for name, arr in [
        ("a1", traj.a1), ("a2", traj.a2),
        ("obs1", traj.obs1), ("obs2", traj.obs2),
        ("r1", traj.r1), ("r2", traj.r2),
        ("false_obs1", traj.false_obs1), ("false_obs2", traj.false_obs2),
    ]:
        assert arr.shape == (horizon,), f"{name} shape {arr.shape} != ({horizon},)"
