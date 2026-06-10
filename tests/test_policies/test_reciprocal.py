"""Unit tests for reciprocal.py policies."""
from __future__ import annotations

import numpy as np
import pytest

from nms.envs.noisy_games import C, D
from nms.policies.reciprocal import (
    WSLS,
    DeterministicTFT,
    GenerousTFT,
    SoftReciprocalPolicy,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── DeterministicTFT ────────────────────────────────────────────────────────

def test_deterministic_tft_cooperates_after_c() -> None:
    p = DeterministicTFT()
    assert p.act(C, _rng()) == C


def test_deterministic_tft_defects_after_d() -> None:
    p = DeterministicTFT()
    assert p.act(D, _rng()) == D


# ── WSLS (Pavlov) ───────────────────────────────────────────────────────────

def test_wsls_initial_cooperate_after_cc() -> None:
    """Starting from C prior: if opponent also played C (match) → stay (C)."""
    p = WSLS()  # _last_own = C
    action = p.act(last_obs_opponent=C, rng=_rng())
    assert action == C  # CC match → stay → C


def test_wsls_shift_after_cd() -> None:
    """Own C, observed D (mismatch) → shift to D."""
    p = WSLS()  # _last_own = C
    action = p.act(last_obs_opponent=D, rng=_rng())
    assert action == D  # CD mismatch → shift → D


def test_wsls_state_updates_correctly() -> None:
    """Multi-step: verify Pavlov/WSLS logic (match→C, mismatch→D).

    Pavlov rule: C after CC or DD (match); D after CD or DC (mismatch).
    Note: after DD the action is C (Pavlov *shifts* D→C, treating DD as a
    mutual-punishment that both players should escape).
    """
    p = WSLS()
    # Step 1: _last_own=C(init), obs=C → match(C==C) → C
    a1 = p.act(C, _rng())
    assert a1 == C

    # Step 2: _last_own=C, obs=D → mismatch(C≠D) → D
    a2 = p.act(D, _rng())
    assert a2 == D

    # Step 3: _last_own=D, obs=D → match(D==D) → C   [Pavlov: DD escape → C]
    a3 = p.act(D, _rng())
    assert a3 == C  # Pavlov shifts D→C after DD match

    # Step 4: _last_own=C, obs=C → match(C==C) → C
    a4 = p.act(C, _rng())
    assert a4 == C


# ── GenerousTFT ─────────────────────────────────────────────────────────────

def test_generous_tft_cooperates_after_c() -> None:
    assert GenerousTFT(0.5).act(C, _rng()) == C


def test_generous_tft_sigma1_always_forgives() -> None:
    """With σ=1 the policy always cooperates even after D."""
    p = GenerousTFT(sigma=1.0)
    actions = [p.act(D, _rng(i)) for i in range(20)]
    assert all(a == C for a in actions)


def test_generous_tft_sigma0_never_forgives() -> None:
    """With σ=0 the policy always defects after D."""
    p = GenerousTFT(sigma=0.0)
    actions = [p.act(D, _rng(i)) for i in range(20)]
    assert all(a == D for a in actions)


def test_generous_tft_invalid_sigma() -> None:
    with pytest.raises(ValueError):
        GenerousTFT(sigma=1.1)


# ── SoftReciprocalPolicy ────────────────────────────────────────────────────

def test_soft_reciprocal_cooperates_after_c() -> None:
    assert SoftReciprocalPolicy(0.3).act(C, _rng()) == C


def test_soft_reciprocal_sigma0_defects_after_d() -> None:
    """σ=0 ≡ DeterministicTFT: always defect after observing D."""
    p = SoftReciprocalPolicy(sigma=0.0)
    actions = [p.act(D, _rng(i)) for i in range(30)]
    assert all(a == D for a in actions)


def test_soft_reciprocal_sigma1_cooperates_after_d() -> None:
    """σ=1 ≡ always cooperate, even after D."""
    p = SoftReciprocalPolicy(sigma=1.0)
    actions = [p.act(D, _rng(i)) for i in range(30)]
    assert all(a == C for a in actions)


def test_soft_reciprocal_forgiveness_rate() -> None:
    """For σ=0.5, roughly 50% of D-observation calls should return C.

    We run 10 000 calls to get a stable estimate and allow ±5% tolerance.
    """
    rng = _rng(42)
    p = SoftReciprocalPolicy(sigma=0.5)
    n = 10_000
    c_count = sum(p.act(D, rng) == C for _ in range(n))
    rate = c_count / n
    assert abs(rate - 0.5) < 0.05, f"Expected ~0.5 forgiveness rate, got {rate:.3f}"
