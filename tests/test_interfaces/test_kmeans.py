"""Tests for KMeansActionDiscretizer interface helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nms.interfaces.kmeans_interface import (
    KMeansActionDiscretizer,
    fit_and_save,
    load_interface,
    predict_batch,
)


def _try_fit(n: int = 5, dim: int = 2) -> KMeansActionDiscretizer | None:
    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except ModuleNotFoundError:
        return None
    rng = np.random.default_rng(42)
    actions = rng.standard_normal((200, dim))
    km = KMeansActionDiscretizer(n_clusters=n, random_state=0)
    km.fit(actions)
    return km


class TestKMeansInterface:
    def test_not_fitted_predict_batch_raises(self) -> None:
        km = KMeansActionDiscretizer(n_clusters=5)
        with pytest.raises(RuntimeError, match="fit"):
            predict_batch(km, np.zeros((10, 2)))

    def test_fit_and_save_roundtrip(self, tmp_path: Path) -> None:
        km = _try_fit()
        if km is None:
            pytest.skip("scikit-learn not installed")
        save_path = tmp_path / "kmeans.npz"
        km.save(save_path)
        km2 = load_interface(save_path)
        assert km2.is_fitted
        assert km2._centroids is not None
        assert km2._centroids.shape[0] == 5

    def test_fit_and_save_helper(self, tmp_path: Path) -> None:
        try:
            from sklearn.cluster import KMeans  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("scikit-learn not installed")
        rng = np.random.default_rng(0)
        actions = rng.standard_normal((100, 3))
        km = fit_and_save(actions, tmp_path / "test", n_clusters=4)
        assert km.is_fitted
        assert (tmp_path / "test.npz").exists()

    def test_predict_batch_shape(self) -> None:
        km = _try_fit(n=5, dim=2)
        if km is None:
            pytest.skip("scikit-learn not installed")
        actions = np.random.default_rng(0).standard_normal((20, 2))
        indices = predict_batch(km, actions)
        assert indices.shape == (20,)
        assert all(0 <= i < 5 for i in indices)

    def test_numpy_kmeans_fallback(self) -> None:
        """NumpyKMeans encode works without sklearn."""
        from nms.wm.dreamer_wrapper import KMeansActionDiscretizer, _NumpyKMeans
        centroids = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        km = KMeansActionDiscretizer(n_clusters=3)
        km._centroids = centroids
        km._kmeans = _NumpyKMeans(centroids)
        idx = km.encode(np.array([0.1, 0.05]))
        assert idx == 0  # nearest to [0,0]
