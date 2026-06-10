"""Tests for DreamerWrapper, DreamerNMSWrapper, KMeansActionDiscretizer.

All tests use numpy-only logic; scikit-learn is needed for KMeans-fit tests
(those are guarded with ``pytest.importorskip`` inside the test bodies).
dreamerv3 is NOT required — the wrapper runs in stub mode when it's absent.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from nms.wm.dreamer_wrapper import DreamerNMSWrapper, DreamerWrapper, KMeansActionDiscretizer

# ── helpers ───────────────────────────────────────────────────────────────────

def _try_fitted_kmeans(
    n_clusters: int = 5,
    action_dim: int = 2,
) -> KMeansActionDiscretizer | None:
    """Return a fitted KMeansActionDiscretizer, or None if sklearn is absent."""
    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except ModuleNotFoundError:
        return None

    rng = np.random.default_rng(0)
    actions = rng.standard_normal((200, action_dim))
    km = KMeansActionDiscretizer(n_clusters=n_clusters, random_state=0)
    km.fit(actions)
    return km


class _MockEnv:
    """Lightweight mock gymnasium environment."""

    class _Space:
        shape = (2,)
        low = np.array([-1.0, -1.0])
        high = np.array([1.0, 1.0])

    observation_space = _Space()
    action_space = _Space()

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(4, dtype=np.float32), 1.0, False, False, {}

    def close(self) -> None:
        pass


# ── KMeansActionDiscretizer — no-fit behaviour (no sklearn needed) ────────────

class TestKMeansNotFitted:
    def test_not_fitted_raises_encode(self) -> None:
        km = KMeansActionDiscretizer(n_clusters=5)
        with pytest.raises(RuntimeError, match="fit"):
            km.encode(np.zeros(2))

    def test_not_fitted_raises_decode(self) -> None:
        km = KMeansActionDiscretizer(n_clusters=5)
        with pytest.raises(RuntimeError, match="fit"):
            km.decode(0)

    def test_is_fitted_false_before_fit(self) -> None:
        km = KMeansActionDiscretizer(n_clusters=5)
        assert not km.is_fitted


# ── KMeansActionDiscretizer — sklearn-dependent tests ────────────────────────

class TestKMeansFitted:
    """These tests require scikit-learn; skipped automatically when absent."""

    def _km(self, n: int = 5) -> KMeansActionDiscretizer:
        km = _try_fitted_kmeans(n_clusters=n)
        if km is None:
            pytest.skip("scikit-learn not installed")
        return km

    def test_fit_marks_fitted(self) -> None:
        km = self._km(5)
        assert km.is_fitted

    def test_encode_decode_roundtrip(self) -> None:
        """encode then decode should return the centroid closest to the input."""
        km = self._km(5)
        action = np.array([0.5, -0.3])
        idx = km.encode(action)
        decoded = km.decode(idx)
        assert decoded.shape == (2,)
        assert 0 <= idx < 5

    def test_n_centroids(self) -> None:
        km = self._km(7)
        assert km._centroids is not None
        assert km._centroids.shape[0] == 7

    def test_decode_batch(self) -> None:
        km = self._km(5)
        indices = np.array([0, 1, 2], dtype=np.intp)
        batch = km.decode_batch(indices)
        assert batch.shape == (3, 2)

    def test_1d_actions_accepted(self) -> None:
        """1-D action arrays should be broadcast to (N, 1) during fit."""
        km = KMeansActionDiscretizer(n_clusters=3, random_state=0)
        if _try_fitted_kmeans() is None:
            pytest.skip("scikit-learn not installed")
        actions = np.random.default_rng(0).standard_normal(100)
        km.fit(actions)
        assert km.is_fitted


# ── DreamerNMSWrapper ─────────────────────────────────────────────────────────

class TestDreamerNMSWrapper:
    """Tests that run WITHOUT dreamerv3 (stub mode) but WITH a fitted kmeans."""

    def _make_wrapper(self, n_clusters: int = 5) -> DreamerNMSWrapper:
        km = _try_fitted_kmeans(n_clusters=n_clusters)
        if km is None:
            pytest.skip("scikit-learn not installed — cannot fit k-means")
        return DreamerNMSWrapper(
            env=_MockEnv(),
            dreamer_cfg={},
            kmeans=km,
            ensemble_k=3,
        )

    def test_unfitted_kmeans_raises(self) -> None:
        km = KMeansActionDiscretizer(n_clusters=5)
        with pytest.raises(ValueError, match="fitted"):
            DreamerNMSWrapper(_MockEnv(), {}, km)

    def test_ensemble_k_zero_raises(self) -> None:
        km = _try_fitted_kmeans()
        if km is None:
            pytest.skip("scikit-learn not installed")
        with pytest.raises(ValueError, match="ensemble_k"):
            DreamerNMSWrapper(_MockEnv(), {}, km, ensemble_k=0)

    def test_reset_returns_obs(self) -> None:
        wrapper = self._make_wrapper()
        obs, info = wrapper.reset()
        assert isinstance(obs, np.ndarray)

    def test_act_returns_continuous_action(self) -> None:
        wrapper = self._make_wrapper(n_clusters=5)
        wrapper.reset()
        latent = np.zeros(4, dtype=np.float32)
        action = wrapper.act(latent)
        assert action.ndim == 1
        assert action.shape == (2,)

    def test_imagine_stub_returns_list(self) -> None:
        wrapper = self._make_wrapper()
        wrapper.reset()
        latent = np.zeros(4, dtype=np.float32)
        imagined = wrapper.imagine(latent, horizon=5)
        assert len(imagined) == 5
        assert all(isinstance(s, np.ndarray) for s in imagined)

    def test_violation_estimator_none_returns_nan(self) -> None:
        wrapper = self._make_wrapper()
        wrapper.reset()
        wrapper._latent = np.zeros(4)
        import warnings
        with warnings.catch_warnings(record=True):
            result = wrapper.estimate_violation_rate(nu=0.1, sigma=0.3, horizon=5)
        assert np.isnan(result)

    def test_violation_estimator_registered(self) -> None:
        wrapper = self._make_wrapper()
        wrapper.reset()
        wrapper._latent = np.zeros(4)

        def fake_estimator(
            latents: list[np.ndarray], nu: float, sigma: float
        ) -> float:
            return 0.15

        wrapper.register_violation_estimator(fake_estimator)
        rate = wrapper.estimate_violation_rate(nu=0.1, sigma=0.3, horizon=5, n_imagined=10)
        assert rate == pytest.approx(0.15)


# ── DreamerWrapper (Phase A stub mode) ───────────────────────────────────────

class TestDreamerWrapper:
    """Tests for the Phase A DreamerWrapper in stub mode (no dreamerv3-torch)."""

    def _make_wrapper(self, seed: int = 0) -> DreamerWrapper:
        env = _MockEnv()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = DreamerWrapper(config_path="fake_config.yaml", env=env, seed=seed)
        return w

    def test_get_set_temperature(self) -> None:
        w = self._make_wrapper()
        assert w.get_actor_temperature() == pytest.approx(1.0)
        w.set_actor_temperature(0.3)
        assert w.get_actor_temperature() == pytest.approx(0.3)

    def test_sample_action_shape(self) -> None:
        w = self._make_wrapper()
        obs = np.zeros(4, dtype=np.float32)
        action = w.sample_action(obs)
        assert action.shape == (2,)

    def test_rollout_returns_correct_keys(self) -> None:
        w = self._make_wrapper()
        env = _MockEnv()
        traj = w.rollout(env, horizon=5, seed=0)
        assert set(traj.keys()) == {"obs", "actions", "rewards"}
        assert traj["actions"].shape[1] == 2   # action_dim
        assert traj["rewards"].ndim == 1

    def test_rollout_obs_ndim(self) -> None:
        w = self._make_wrapper()
        env = _MockEnv()
        traj = w.rollout(env, horizon=5)
        assert traj["obs"].ndim == 2

    def test_rollout_imagined_shape(self) -> None:
        w = self._make_wrapper(seed=42)
        obs = np.zeros(4, dtype=np.float32)
        result = w.rollout_imagined(obs, horizon=15, sigma=0.3, seed=0)
        assert "actions" in result
        assert result["actions"].shape == (15, 2)

    def test_rollout_realized_shape(self) -> None:
        w = self._make_wrapper(seed=7)
        env = _MockEnv()
        result = w.rollout_realized(env, horizon=15, sigma=0.5, seed=1)
        assert "actions" in result
        assert result["actions"].ndim == 2

    def test_train_stub_no_error(self) -> None:
        """train() in stub mode should not raise, just warn."""
        w = self._make_wrapper()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            w.train(total_steps=10, log_dir="/tmp/test_dreamer")

    def test_rssm_posterior_stub_shape(self) -> None:
        w = self._make_wrapper()
        obs = np.zeros(4, dtype=np.float32)
        mu, var = w.rssm_posterior(obs)
        assert mu.ndim == 1
        assert var.ndim == 1
        assert mu.shape == var.shape

    def test_seed_reproducibility(self) -> None:
        """Two wrappers with the same seed should produce identical actions."""
        obs = np.zeros(4, dtype=np.float32)
        w1 = self._make_wrapper(seed=99)
        w2 = self._make_wrapper(seed=99)
        a1 = w1.sample_action(obs)
        a2 = w2.sample_action(obs)
        np.testing.assert_array_equal(a1, a2)
