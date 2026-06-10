"""Tests for CPBDreamerController."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from nms.controllers.cpb_dreamer import CPBDreamerController
from nms.wm.dreamer_wrapper import DreamerWrapper, KMeansActionDiscretizer


class _MockActionSpace:
    shape = (2,)
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])


class _MockObsSpace:
    shape = (4,)


class _MockEnv:
    observation_space = _MockObsSpace()
    action_space = _MockActionSpace()

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), 1.0, False, False, {}

    def close(self) -> None:
        pass


def _make_fitted_km(n: int = 5) -> KMeansActionDiscretizer:
    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("scikit-learn not installed")
    rng = np.random.default_rng(0)
    actions = rng.standard_normal((200, 2))
    km = KMeansActionDiscretizer(n_clusters=n, random_state=0)
    km.fit(actions)
    return km


class TestCPBDreamerController:
    def _make_ctrl(self, c_cal: float | None = 3.45) -> CPBDreamerController:
        env = _MockEnv()
        km = _make_fitted_km()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dreamer = DreamerWrapper(config_path="fake.yaml", env=env, seed=0)
        return CPBDreamerController(dreamer, km, alpha=0.1, horizon=5, c_cal=c_cal)

    def test_construction(self) -> None:
        ctrl = self._make_ctrl()
        assert ctrl.alpha == pytest.approx(0.1)
        assert ctrl.horizon == 5
        assert ctrl.c_cal == pytest.approx(3.45)

    def test_get_bandwidth_uses_c_cal(self) -> None:
        ctrl = self._make_ctrl(c_cal=3.45)
        # No calibration run → falls back to c_cal * nu
        bw = ctrl.get_bandwidth(nu_hat=0.1)
        assert bw == pytest.approx(0.345, rel=0.01)

    def test_get_bandwidth_no_c_cal(self) -> None:
        ctrl = self._make_ctrl(c_cal=None)
        bw = ctrl.get_bandwidth(nu_hat=0.1)
        assert bw == 0.0  # no c_cal, no calibration → 0

    def test_select_action_shape(self) -> None:
        ctrl = self._make_ctrl()
        obs = np.zeros(4, dtype=np.float32)
        action = ctrl.select_action(obs, nu_hat=0.1)
        assert action.ndim == 1
        assert action.shape == (2,)

    def test_select_action_clips_temperature(self) -> None:
        """sigma_cpb = clip(c_cal * nu, sigma_min, sigma_max)."""
        ctrl = self._make_ctrl(c_cal=3.45)
        obs = np.zeros(4, dtype=np.float32)
        # Before action, temperature should be restored
        original_tau = ctrl.dreamer.get_actor_temperature()
        ctrl.select_action(obs, nu_hat=0.1, sigma_min=0.0, sigma_max=0.5)
        assert ctrl.dreamer.get_actor_temperature() == pytest.approx(original_tau)

    def test_violation_rates_initially_empty(self) -> None:
        ctrl = self._make_ctrl()
        assert ctrl.violation_rates == {}

    def test_bandwidth_dict_initially_empty(self) -> None:
        ctrl = self._make_ctrl()
        assert ctrl.bandwidth_dict == {}
