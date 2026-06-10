"""Tests for OOD environment wrappers (Phase 8/9).

Uses mock environments so dm_control is not required.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nms.envs.ood_wrappers import MassFrictionPerturbWrapper


# ── Mock environment ──────────────────────────────────────────────────────────

class _MockActionSpace:
    shape = (2,)
    low = -np.ones(2, dtype=np.float32)
    high = np.ones(2, dtype=np.float32)


class _MockObsSpace:
    shape = (4,)


class _MockEnv:
    """Minimal mock without physics (MassFrictionPerturbWrapper should skip silently)."""

    def __init__(self) -> None:
        self.observation_space = _MockObsSpace()
        self.action_space = _MockActionSpace()
        self.metadata = {}
        self.render_mode = None
        self._step_count = 0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        return np.ones(4, dtype=np.float32) * 0.5, 1.0, self._step_count >= 5, False, {}

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


class _MockPhysicsModel:
    def __init__(self) -> None:
        self.body_mass = np.array([0.0, 5.0, 2.0, 1.0])
        self.geom_friction = np.array([
            [1.0, 0.1, 0.01],
            [0.5, 0.1, 0.01],
            [0.5, 0.1, 0.01],
        ])


class _MockPhysics:
    def __init__(self) -> None:
        self.model = _MockPhysicsModel()


class _MockEnvWithPhysics(_MockEnv):
    """Mock env that exposes a physics attribute (simulates dm_control)."""

    def __init__(self) -> None:
        super().__init__()
        self.physics = _MockPhysics()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMassFrictionPerturbWrapper:
    """Tests for MassFrictionPerturbWrapper."""

    def test_identity_delta_zero(self) -> None:
        """delta=0 should produce no change."""
        env = _MockEnvWithPhysics()
        orig_masses = env.physics.model.body_mass.copy()
        orig_frictions = env.physics.model.geom_friction.copy()

        wrapped = MassFrictionPerturbWrapper(env, delta=0.0)

        np.testing.assert_array_equal(env.physics.model.body_mass, orig_masses)
        np.testing.assert_array_equal(env.physics.model.geom_friction, orig_frictions)

    def test_delta_scales_masses(self) -> None:
        """delta > 0 should multiply all masses by (1 + delta)."""
        env = _MockEnvWithPhysics()
        orig_masses = env.physics.model.body_mass.copy()

        wrapped = MassFrictionPerturbWrapper(env, delta=0.5)

        expected = orig_masses * 1.5
        np.testing.assert_allclose(env.physics.model.body_mass, expected, rtol=1e-5)

    def test_delta_reduces_friction(self) -> None:
        """delta > 0 should multiply geom_friction[:, 0] by max(1-delta, 0.05)."""
        env = _MockEnvWithPhysics()
        orig_friction0 = env.physics.model.geom_friction[:, 0].copy()

        wrapped = MassFrictionPerturbWrapper(env, delta=0.3)

        expected = np.clip(orig_friction0 * 0.7, 0.05, None)
        np.testing.assert_allclose(
            env.physics.model.geom_friction[:, 0], expected, rtol=1e-5
        )

    def test_friction_clamped_at_0_05(self) -> None:
        """Friction should never go below 0.05 regardless of delta."""
        env = _MockEnvWithPhysics()
        wrapped = MassFrictionPerturbWrapper(env, delta=0.99)  # 99% reduction

        assert np.all(env.physics.model.geom_friction[:, 0] >= 0.05)

    def test_negative_delta_raises(self) -> None:
        """delta < 0 is invalid."""
        env = _MockEnvWithPhysics()
        with pytest.raises(ValueError, match="delta must be"):
            MassFrictionPerturbWrapper(env, delta=-0.1)

    def test_no_physics_is_silent(self) -> None:
        """Env without physics attribute should not raise — silent pass-through."""
        env = _MockEnv()  # no physics
        wrapped = MassFrictionPerturbWrapper(env, delta=0.5)
        # Should not raise; just behave as pass-through
        obs, info = wrapped.reset()
        assert obs.shape == (4,)

    def test_reset_passthrough(self) -> None:
        """reset() / step() should pass through to the wrapped env."""
        env = _MockEnvWithPhysics()
        wrapped = MassFrictionPerturbWrapper(env, delta=0.2)

        obs, info = wrapped.reset()
        assert obs.shape == (4,)

        obs2, r, term, trunc, _ = wrapped.step(np.zeros(2))
        assert obs2.shape == (4,)
        assert isinstance(r, float)

    def test_obs_action_spaces_forwarded(self) -> None:
        """Wrapped env exposes original observation and action spaces."""
        env = _MockEnvWithPhysics()
        wrapped = MassFrictionPerturbWrapper(env, delta=0.3)

        assert wrapped.observation_space.shape == (4,)
        assert wrapped.action_space.shape == (2,)

    def test_delta_1_sets_mass_2x(self) -> None:
        """delta=1.0 should double the mass."""
        env = _MockEnvWithPhysics()
        orig_masses = env.physics.model.body_mass.copy()
        wrapped = MassFrictionPerturbWrapper(env, delta=1.0)

        expected = orig_masses * 2.0
        np.testing.assert_allclose(env.physics.model.body_mass, expected, rtol=1e-5)

    def test_getattr_delegation(self) -> None:
        """__getattr__ should delegate unknown attrs to the wrapped env."""
        env = _MockEnvWithPhysics()
        env.custom_attr = "hello"  # type: ignore[attr-defined]
        wrapped = MassFrictionPerturbWrapper(env, delta=0.1)

        assert wrapped.custom_attr == "hello"  # type: ignore[attr-defined]
