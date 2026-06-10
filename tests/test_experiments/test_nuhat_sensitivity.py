"""Smoke tests for run_nuhat_sensitivity helpers.

These tests exercise the _build_bandwidth_lookup and _nearest_nu helpers
from run_nuhat_sensitivity without running actual rollouts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.exp1_noisy_games.run_nuhat_sensitivity import (
    _build_bandwidth_lookup,
    _nearest_nu,
)
from nms.core.envelope import Bandwidth

# ── helper builders ────────────────────────────────────────────────────────────

def _make_sweep_df(
    nus: list[float],
    sigmas: list[float],
    risk_fn: object,
) -> pd.DataFrame:
    """Build a minimal sweep CSV DataFrame."""
    rows = []
    for nu in nus:
        for sigma in sigmas:
            rows.append({
                "nu": nu,
                "sigma": sigma,
                "horizon": 200,
                "seed": 0,
                "violation_rate": risk_fn(nu, sigma),  # type: ignore[operator]
                "mean_return": 3.0,
            })
    return pd.DataFrame(rows)


# ── _nearest_nu ───────────────────────────────────────────────────────────────

class TestNearestNu:
    def test_exact_match(self) -> None:
        assert _nearest_nu(0.1, [0.05, 0.10, 0.20]) == pytest.approx(0.10)

    def test_lower_half(self) -> None:
        assert _nearest_nu(0.07, [0.05, 0.10]) == pytest.approx(0.05)

    def test_upper_half(self) -> None:
        assert _nearest_nu(0.08, [0.05, 0.10]) == pytest.approx(0.10)

    def test_single_element(self) -> None:
        assert _nearest_nu(0.99, [0.1]) == pytest.approx(0.1)

    def test_boundary(self) -> None:
        assert _nearest_nu(0.0, [0.0, 0.1, 0.2]) == pytest.approx(0.0)


# ── _build_bandwidth_lookup ───────────────────────────────────────────────────

class TestBuildBandwidthLookup:
    """Risk = 0.05 everywhere → every sigma is admissible → [σ_min, σ_max] = grid."""

    def _low_risk_df(self) -> pd.DataFrame:
        nus = [0.1, 0.2]
        sigmas = [0.0, 0.2, 0.4, 0.6, 0.8]
        return _make_sweep_df(nus, sigmas, lambda nu, sigma: 0.05)

    def test_keys_are_nus(self) -> None:
        df = self._low_risk_df()
        lookup = _build_bandwidth_lookup(df, horizon=200, alpha=0.1)
        assert sorted(lookup.keys()) == pytest.approx([0.1, 0.2])

    def test_returns_bandwidth_objects(self) -> None:
        df = self._low_risk_df()
        lookup = _build_bandwidth_lookup(df, horizon=200, alpha=0.1)
        for bw in lookup.values():
            assert isinstance(bw, Bandwidth)

    def test_full_admissible_range(self) -> None:
        df = self._low_risk_df()
        lookup = _build_bandwidth_lookup(df, horizon=200, alpha=0.1)
        bw = lookup[0.1]
        assert not bw.is_empty()
        assert bw.sigma_min == pytest.approx(0.0, abs=1e-6)
        assert bw.sigma_max == pytest.approx(0.8, abs=1e-6)

    def test_high_risk_all_empty(self) -> None:
        """Risk = 1.0 everywhere → no admissible sigma → empty bandwidth."""
        nus = [0.1]
        sigmas = [0.0, 0.2, 0.4]
        df = _make_sweep_df(nus, sigmas, lambda nu, sigma: 1.0)
        lookup = _build_bandwidth_lookup(df, horizon=200, alpha=0.1)
        assert lookup[0.1].is_empty()

    def test_u_shaped_risk(self) -> None:
        """U-shaped risk: low risk in [0.3, 0.6], high outside → interior band."""
        nus = [0.1]
        sigmas = np.linspace(0.0, 1.0, 21).tolist()

        def u_risk(nu: float, sigma: float) -> float:  # noqa: ARG001
            return 0.05 if 0.3 <= sigma <= 0.6 else 0.8

        df = _make_sweep_df(nus, sigmas, u_risk)
        lookup = _build_bandwidth_lookup(df, horizon=200, alpha=0.1)
        bw = lookup[0.1]
        assert not bw.is_empty()
        assert bw.sigma_min > 0.0
        assert bw.sigma_max < 1.0
        assert bw.sigma_min == pytest.approx(0.3, abs=0.06)
        assert bw.sigma_max == pytest.approx(0.6, abs=0.06)

    def test_horizon_filter(self) -> None:
        """When CSV has multiple horizons only the target horizon is used."""
        rows = []
        for h in [100, 200]:
            for sigma in [0.0, 0.5, 1.0]:
                # Only H=200 has low risk
                risk = 0.05 if h == 200 else 0.9
                rows.append({"nu": 0.1, "sigma": sigma, "horizon": h,
                              "seed": 0, "violation_rate": risk, "mean_return": 3.0})
        df = pd.DataFrame(rows)
        lookup_200 = _build_bandwidth_lookup(df, horizon=200, alpha=0.1)
        lookup_100 = _build_bandwidth_lookup(df, horizon=100, alpha=0.1)
        assert not lookup_200[0.1].is_empty()
        assert lookup_100[0.1].is_empty()
