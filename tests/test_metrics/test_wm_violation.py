"""Tests for wm_violation.py metrics."""
from __future__ import annotations

import numpy as np
import pytest

from nms.metrics.wm_violation import (
    batch_violation_rate,
    imagined_realized_disagreement,
    per_step_disagreement_rate,
    violation_indicator_wm,
)


class _MockKMeans:
    """Trivial k-means: argmax of action vector."""

    def encode(self, action: np.ndarray) -> int:
        return int(np.argmax(np.asarray(action)))

    def predict(self, x_arr: np.ndarray) -> np.ndarray:
        return np.array([int(np.argmax(x)) for x in x_arr])


_KM = _MockKMeans()


def _make_traj(actions: np.ndarray) -> dict:
    return {"actions": actions}


class TestImagedRealizedDisagreement:
    def test_identical_actions_zero_disagreement(self) -> None:
        acts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        img = _make_traj(acts)
        rea = _make_traj(acts)
        assert imagined_realized_disagreement(img, rea, _KM) == pytest.approx(0.0)

    def test_all_different_max_disagreement(self) -> None:
        img_acts = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        rea_acts = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
        img = _make_traj(img_acts)
        rea = _make_traj(rea_acts)
        assert imagined_realized_disagreement(img, rea, _KM) == pytest.approx(1.0)

    def test_half_different(self) -> None:
        img_acts = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        rea_acts = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
        img = _make_traj(img_acts)
        rea = _make_traj(rea_acts)
        assert imagined_realized_disagreement(img, rea, _KM) == pytest.approx(0.5)

    def test_empty_actions_zero(self) -> None:
        img = _make_traj(np.zeros((0, 2), dtype=float))
        rea = _make_traj(np.zeros((0, 2), dtype=float))
        assert imagined_realized_disagreement(img, rea, _KM) == pytest.approx(0.0)


class TestViolationIndicator:
    def test_no_disagreement_zero(self) -> None:
        acts = np.array([[1, 0], [0, 1]], dtype=float)
        img = _make_traj(acts)
        rea = _make_traj(acts)
        assert violation_indicator_wm(img, rea, _KM, horizon=2) == 0

    def test_disagreement_one(self) -> None:
        img = _make_traj(np.array([[1, 0]], dtype=float))
        rea = _make_traj(np.array([[0, 1]], dtype=float))
        assert violation_indicator_wm(img, rea, _KM, horizon=1) == 1

    def test_horizon_limits_check(self) -> None:
        # disagreement at step 2 (index 1), horizon=1 → no violation
        img = _make_traj(np.array([[1, 0], [0, 1]], dtype=float))
        rea = _make_traj(np.array([[1, 0], [1, 0]], dtype=float))
        assert violation_indicator_wm(img, rea, _KM, horizon=1) == 0
        assert violation_indicator_wm(img, rea, _KM, horizon=2) == 1


class TestBatchViolationRate:
    def test_all_agree_zero(self) -> None:
        acts = np.array([[1, 0], [0, 1]], dtype=float)
        batch = [_make_traj(acts)] * 5
        assert batch_violation_rate(batch, batch, _KM, horizon=2) == pytest.approx(0.0)

    def test_all_disagree_one(self) -> None:
        img = _make_traj(np.array([[1, 0]], dtype=float))
        rea = _make_traj(np.array([[0, 1]], dtype=float))
        batch_i = [img] * 4
        batch_r = [rea] * 4
        assert batch_violation_rate(batch_i, batch_r, _KM, horizon=1) == pytest.approx(1.0)

    def test_size_mismatch_raises(self) -> None:
        acts = np.array([[1, 0]], dtype=float)
        with pytest.raises(ValueError, match="Batch sizes"):
            batch_violation_rate(
                [_make_traj(acts)] * 3,
                [_make_traj(acts)] * 4,
                _KM, horizon=1,
            )

    def test_empty_batch_zero(self) -> None:
        assert batch_violation_rate([], [], _KM, horizon=5) == pytest.approx(0.0)


class TestPerStepDisagreementRate:
    def test_shape(self) -> None:
        acts_i = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
        acts_r = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        batch_i = [_make_traj(acts_i)] * 3
        batch_r = [_make_traj(acts_r)] * 3
        rates = per_step_disagreement_rate(batch_i, batch_r, _KM)
        assert rates.shape == (3,)

    def test_size_mismatch_raises(self) -> None:
        acts = np.array([[1, 0]], dtype=float)
        with pytest.raises(ValueError):
            per_step_disagreement_rate(
                [_make_traj(acts)], [_make_traj(acts), _make_traj(acts)], _KM
            )
