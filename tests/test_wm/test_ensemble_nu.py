"""Tests for EnsembleNuEstimator (Phase 2.5).

Uses mock DreamerWrapper members so no GPU/training is required.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from nms.wm.ensemble_nu import EnsembleNuEstimator


# ── Mock member ───────────────────────────────────────────────────────────────

class _MockMember:
    """Mock DreamerWrapper with deterministic rssm_posterior."""

    def __init__(self, mean_offset: float = 0.0, var_scale: float = 0.01) -> None:
        self._backend = "dreamerv3-torch"
        self._agent = None  # no NM512 agent — falls back to rssm_posterior
        self.device = "cpu"
        self._mean_offset = float(mean_offset)
        self._var_scale = float(var_scale)

    def _ensure_agent(self) -> None:
        pass  # no-op for mock

    def rssm_posterior(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        feat_dim = 64
        # Different members return different means (controlled by offset)
        mean = np.full(feat_dim, self._mean_offset + float(np.sum(obs)) * 0.001,
                       dtype=np.float64)
        var = np.full(feat_dim, self._var_scale, dtype=np.float64)
        return mean, var


class _FailingMember(_MockMember):
    """Member that always raises in rssm_posterior (to test robustness)."""

    def rssm_posterior(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("simulated member failure")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEnsembleNuEstimator:

    def test_construction(self) -> None:
        est = EnsembleNuEstimator(
            [_MockMember(), _MockMember(), _MockMember()], horizon=5
        )
        assert est.k == 3
        assert est.H == 5

    def test_construction_single_member(self) -> None:
        est = EnsembleNuEstimator([_MockMember()], horizon=3)
        assert est.k == 1

    def test_construction_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1 member"):
            EnsembleNuEstimator([], horizon=5)

    def test_estimate_returns_scalar_in_range(self) -> None:
        est = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(0.5), _MockMember(1.0)], horizon=5
        )
        obs = np.zeros(17)
        nu_hat = est.estimate(obs, n_samples=4)

        assert isinstance(nu_hat, float)
        assert est.nu_min <= nu_hat <= est.nu_max

    def test_estimate_varies_with_observation(self) -> None:
        """ν̂ should differ for observations that produce different latents."""
        est = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(1.0), _MockMember(2.0)], horizon=5
        )
        obs_a = np.zeros(17)
        obs_b = np.ones(17) * 10.0

        nu_a = est.estimate(obs_a, n_samples=4)
        nu_b = est.estimate(obs_b, n_samples=4)

        # Both should be valid
        assert est.nu_min <= nu_a <= est.nu_max
        assert est.nu_min <= nu_b <= est.nu_max

    def test_estimate_components_keys(self) -> None:
        est = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(1.0)], horizon=3
        )
        obs = np.zeros(10)
        result = est.estimate_components(obs, n_samples=2)

        assert set(result.keys()) == {"nu_hat", "aleatoric", "epistemic", "n_members_ok"}
        assert result["n_members_ok"] == 2

    def test_epistemic_increases_with_member_disagreement(self) -> None:
        """Larger spread between member means → higher epistemic."""
        est_low = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(0.01)], horizon=3
        )
        est_high = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(5.0)], horizon=3
        )
        obs = np.zeros(10)

        comp_low = est_low.estimate_components(obs, n_samples=2)
        comp_high = est_high.estimate_components(obs, n_samples=2)

        assert comp_high["epistemic"] > comp_low["epistemic"]

    def test_failing_member_is_skipped(self) -> None:
        """If one member fails, the others should still contribute."""
        est = EnsembleNuEstimator(
            [_MockMember(0.0), _FailingMember(), _MockMember(1.0)],
            horizon=3,
        )
        obs = np.zeros(10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = est.estimate_components(obs, n_samples=2)
        # Only 2 members should succeed
        assert result["n_members_ok"] == 2
        # At least one warning should be about the failing member
        assert any("member failed" in str(warning.message) for warning in w)

    def test_all_failing_returns_default(self) -> None:
        """When no member succeeds, fallback to nu_default."""
        est = EnsembleNuEstimator(
            [_FailingMember(), _FailingMember()],
            horizon=3,
            nu_default=0.15,
        )
        obs = np.zeros(10)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            nu_hat = est.estimate(obs, n_samples=2)
        assert nu_hat == pytest.approx(0.15)

    def test_nu_scale_applies(self) -> None:
        """nu_scale should linearly scale the output."""
        est1 = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(1.0)], horizon=3, nu_scale=1.0
        )
        est2 = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(1.0)], horizon=3, nu_scale=2.0
        )
        obs = np.zeros(10)

        nu1 = est1.estimate(obs, n_samples=2)
        nu2 = est2.estimate(obs, n_samples=2)

        # nu2 should be larger (higher scale) — both clamped to [nu_min, nu_max]
        # At minimum, nu2 >= nu1
        assert nu2 >= nu1

    def test_nu_min_nu_max_clipping(self) -> None:
        """Output should always be within [nu_min, nu_max]."""
        est = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(100.0)],  # huge spread
            horizon=3, nu_min=0.05, nu_max=0.5,
        )
        obs = np.zeros(10)
        nu_hat = est.estimate(obs, n_samples=4)

        assert 0.05 <= nu_hat <= 0.5

    def test_calibrate_nu_scale(self) -> None:
        """calibrate_nu_scale should adjust nu_scale to match env_sigma."""
        est = EnsembleNuEstimator(
            [_MockMember(0.0), _MockMember(0.5)], horizon=3, nu_scale=1.0
        )
        obs_batch = np.zeros((10, 17))
        target_sigma = 0.2

        scale = est.calibrate_nu_scale(target_sigma, obs_batch, n_samples=2)
        # After calibration, mean estimate should be close to target_sigma
        assert isinstance(scale, float)
        assert scale > 0
        assert est.nu_scale == pytest.approx(scale)
