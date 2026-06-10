"""Unit tests for opponents.py."""
from __future__ import annotations

import numpy as np
import pytest

from nms.envs.noisy_games import C, D
from nms.policies.opponents import (
    AlwaysCooperate,
    AlwaysDefect,
    parse_mixture_spec,
    sample_opponent,
)
from nms.policies.reciprocal import SoftReciprocalPolicy


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── AlwaysDefect ─────────────────────────────────────────────────────────────

def test_always_defect_returns_d() -> None:
    p = AlwaysDefect()
    for obs in (C, D):
        assert p.act(obs, _rng()) == D


# ── AlwaysCooperate ──────────────────────────────────────────────────────────

def test_always_cooperate_returns_c() -> None:
    p = AlwaysCooperate()
    for obs in (C, D):
        assert p.act(obs, _rng()) == C


# ── sample_opponent ──────────────────────────────────────────────────────────

def test_sample_opponent_uniform() -> None:
    """With equal weights, both policies should appear roughly half the time."""
    p1 = AlwaysDefect()
    p2 = AlwaysCooperate()
    mixture = {p1: 0.5, p2: 0.5}
    rng = _rng(42)
    counts = {p1: 0, p2: 0}
    n = 2000
    for _ in range(n):
        opp = sample_opponent(rng, mixture)
        counts[opp] += 1  # type: ignore[index]
    rate = counts[p1] / n
    assert abs(rate - 0.5) < 0.05, f"Expected ~0.5 got {rate:.3f}"


def test_sample_opponent_deterministic() -> None:
    """Weight=1.0 on a single policy always returns that policy."""
    p = AlwaysDefect()
    mixture = {p: 1.0}
    rng = _rng(0)
    for _ in range(20):
        assert sample_opponent(rng, mixture) is p


# ── parse_mixture_spec ───────────────────────────────────────────────────────

def test_parse_mixture_spec_always_defect() -> None:
    mixture = parse_mixture_spec("always_defect:1.0", agent_sigma=0.3)
    assert len(mixture) == 1
    (policy, weight), = mixture.items()
    assert isinstance(policy, AlwaysDefect)
    assert weight == pytest.approx(1.0)


def test_parse_mixture_spec_always_cooperate() -> None:
    mixture = parse_mixture_spec("always_cooperate:1.0", agent_sigma=0.3)
    (policy, _), = mixture.items()
    assert isinstance(policy, AlwaysCooperate)


def test_parse_mixture_spec_reciprocal_uses_agent_sigma() -> None:
    """The reciprocal weight is the mixture probability; sigma=agent_sigma."""
    mixture = parse_mixture_spec("reciprocal:0.8,always_defect:0.2", agent_sigma=0.4)
    assert len(mixture) == 2
    policies = list(mixture.keys())
    rec = next(p for p in policies if isinstance(p, SoftReciprocalPolicy))
    assert rec.sigma == pytest.approx(0.4)
    alld = next(p for p in policies if isinstance(p, AlwaysDefect))
    assert alld is not None
    # weights
    rec_weight = mixture[rec]
    alld_weight = mixture[alld]
    assert rec_weight == pytest.approx(0.8)
    assert alld_weight == pytest.approx(0.2)


def test_parse_mixture_spec_pure_selfplay() -> None:
    """reciprocal:1.0 is the self-play baseline."""
    mixture = parse_mixture_spec("reciprocal:1.0", agent_sigma=0.3)
    assert len(mixture) == 1
    (policy, weight), = mixture.items()
    assert isinstance(policy, SoftReciprocalPolicy)
    assert weight == pytest.approx(1.0)


def test_parse_mixture_spec_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown opponent type"):
        parse_mixture_spec("grim_trigger:1.0", agent_sigma=0.3)


def test_parse_mixture_spec_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_mixture_spec("", agent_sigma=0.3)
