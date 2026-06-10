"""Tests for stochastic DM Control wrappers (Brief B Phase A).

These tests use a lightweight mock environment so gymnasium/dm_control
are not required to run the test suite.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nms.envs.stochastic_dm_control import (
    DynamicsShiftWrapper,
    ObservationNoiseWrapper,
    PhysicsParameterShiftWrapper,
    RewardStochasticityWrapper,
    StochasticDynamicsWrapper,
)

# ── Mock environment ──────────────────────────────────────────────────────────

class _MockActionSpace:
    """Mock Box-like action space with low/high bounds."""

    def __init__(self, n: int = 2) -> None:
        self.shape = (n,)
        self.low = -np.ones(n, dtype=np.float32)
        self.high = np.ones(n, dtype=np.float32)


class _MockObsSpace:
    shape = (4,)


class _MockEnv:
    """Minimal gymnasium-compatible mock environment."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 2) -> None:
        self.observation_space = _MockObsSpace()
        self.action_space = _MockActionSpace(action_dim)
        self._step_count = 0
        self._obs_dim = obs_dim

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = np.ones(self._obs_dim, dtype=np.float32) * 0.5
        reward = 1.0
        terminated = self._step_count >= 5
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        pass


# ── ObservationNoiseWrapper ───────────────────────────────────────────────────

class TestObservationNoiseWrapper:
    def test_sigma_zero_no_noise(self) -> None:
        """sigma=0 must return the exact observation unchanged."""
        env = ObservationNoiseWrapper(_MockEnv(), sigma=0.0)
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, np.zeros(4, dtype=np.float32))

    def test_sigma_positive_adds_noise(self) -> None:
        """sigma>0 should change the observation."""
        env = ObservationNoiseWrapper(_MockEnv(), sigma=1.0, seed=0)
        obs, _ = env.reset()
        # All-zero base obs + noise should generally be nonzero
        assert not np.allclose(obs, 0.0), "Noise should perturb the observation"

    def test_observation_dtype_preserved(self) -> None:
        """Wrapper must preserve the original observation dtype."""
        env = ObservationNoiseWrapper(_MockEnv(), sigma=0.1, seed=1)
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_obs_noise(self) -> None:
        """Noise is also applied in step()."""
        env = ObservationNoiseWrapper(_MockEnv(), sigma=0.0, seed=2)
        env.reset()
        obs_noiseless, _, _, _, _ = env.step(np.zeros(2))

        env2 = ObservationNoiseWrapper(_MockEnv(), sigma=1.0, seed=3)
        env2.reset()
        obs_noisy, _, _, _, _ = env2.step(np.zeros(2))
        assert not np.allclose(obs_noiseless, obs_noisy)

    def test_sigma_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be"):
            ObservationNoiseWrapper(_MockEnv(), sigma=-0.1)


# ── RewardStochasticityWrapper ────────────────────────────────────────────────

class TestRewardStochasticityWrapper:
    def test_sigma_zero_exact_reward(self) -> None:
        env = RewardStochasticityWrapper(_MockEnv(), sigma=0.0)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(2))
        assert reward == pytest.approx(1.0)

    def test_sigma_positive_perturbs_reward(self) -> None:
        rewards = []
        for seed in range(20):
            env = RewardStochasticityWrapper(_MockEnv(), sigma=1.0, seed=seed)
            env.reset()
            _, r, _, _, _ = env.step(np.zeros(2))
            rewards.append(r)
        assert np.std(rewards) > 0.1, "Reward noise should produce variance"

    def test_reward_is_float(self) -> None:
        env = RewardStochasticityWrapper(_MockEnv(), sigma=0.5, seed=99)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(2))
        assert isinstance(reward, float)


# ── DynamicsShiftWrapper ──────────────────────────────────────────────────────

class TestDynamicsShiftWrapper:
    def test_sigma_zero_no_shift(self) -> None:
        """sigma=0 should pass the action through unchanged (same reward)."""
        base = _MockEnv()
        env = DynamicsShiftWrapper(base, sigma=0.0, seed=5)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(2))
        assert reward == pytest.approx(1.0)

    def test_clip_to_action_bounds(self) -> None:
        """Large sigma should not produce out-of-bounds actions (clipped)."""
        env = DynamicsShiftWrapper(_MockEnv(), sigma=100.0, seed=7)
        env.reset()
        # Step should not raise even with large dynamics shift
        env.step(np.ones(2))

    def test_action_space_forwarded(self) -> None:
        env = DynamicsShiftWrapper(_MockEnv(), sigma=0.1)
        assert env.action_space is not None


# ── StochasticDynamicsWrapper ─────────────────────────────────────────────────

class TestStochasticDynamicsWrapper:
    def test_sigma_zero_deterministic(self) -> None:
        """sigma=0 everywhere should recover the base environment exactly."""
        env = StochasticDynamicsWrapper(_MockEnv(), sigma=0.0)
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, np.zeros(4, dtype=np.float32))
        _, reward, _, _, _ = env.step(np.zeros(2))
        assert reward == pytest.approx(1.0)

    def test_per_component_override(self) -> None:
        """sigma_obs=0 should suppress observation noise even if sigma>0."""
        env = StochasticDynamicsWrapper(
            _MockEnv(), sigma=1.0, sigma_obs=0.0, sigma_reward=0.0, seed=11
        )
        obs_base = np.zeros(4, dtype=np.float32)
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, obs_base)

    def test_reward_noise_active(self) -> None:
        """sigma_reward>0 should perturb the reward."""
        rewards = []
        for seed in range(20):
            env = StochasticDynamicsWrapper(
                _MockEnv(), sigma=0.0, sigma_reward=1.0, seed=seed
            )
            env.reset()
            _, r, _, _, _ = env.step(np.zeros(2))
            rewards.append(r)
        assert np.std(rewards) > 0.1

    def test_obs_noise_active(self) -> None:
        """sigma_obs>0 should perturb observations on reset."""
        env_noisy = StochasticDynamicsWrapper(
            _MockEnv(), sigma=0.0, sigma_obs=1.0, seed=42
        )
        obs, _ = env_noisy.reset()
        assert not np.allclose(obs, 0.0)

    def test_reproduces_with_same_seed(self) -> None:
        """Same seed → same trajectory."""
        def run(seed: int) -> list[float]:
            env = StochasticDynamicsWrapper(_MockEnv(), sigma=0.5, seed=seed)
            env.reset()
            rs = []
            for _ in range(3):
                _, r, done, _, _ = env.step(np.zeros(2))
                rs.append(r)
                if done:
                    break
            return rs

        assert run(0) == run(0)
        assert run(0) != run(1)

    def test_sigma_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be"):
            StochasticDynamicsWrapper(_MockEnv(), sigma=-0.5)


# ── PhysicsParameterShiftWrapper ──────────────────────────────────────────────

class TestPhysicsParameterShiftWrapper:
    """Tests for the OOD physics-parameter wrapper (Phase D)."""

    def test_wraps_env_without_physics(self) -> None:
        """Env without 'physics' attribute should be wrapped silently."""
        env = _MockEnv()
        wrapped = PhysicsParameterShiftWrapper(env, mass_mult=2.0)
        obs, info = wrapped.reset()
        assert obs is not None

    def test_step_passthrough(self) -> None:
        env = _MockEnv()
        wrapped = PhysicsParameterShiftWrapper(env)
        wrapped.reset()
        obs, reward, term, trunc, _ = wrapped.step(np.zeros(2))
        assert obs.shape == (4,)
        assert isinstance(reward, float)

    def test_spaces_forwarded(self) -> None:
        env = _MockEnv()
        wrapped = PhysicsParameterShiftWrapper(env, friction_mult=0.5)
        assert wrapped.observation_space is env.observation_space
        assert wrapped.action_space is env.action_space

    def test_with_mock_physics(self) -> None:
        """Env with a mock 'physics' attribute — multipliers applied in __init__."""
        class _Model:
            body_mass = np.array([1.0, 2.0, 3.0])
            geom_friction = np.array([[0.4], [0.8]])
            dof_damping = np.array([0.1, 0.2])

        class _Named:
            model = _Model()

        class _Physics:
            named = _Named()

        class _MockPhysicsEnv(_MockEnv):
            _phys = _Physics()

            @property
            def physics(self) -> Any:
                return self._phys

            @property
            def unwrapped(self) -> Any:
                return self

        env = _MockPhysicsEnv()
        original_mass = env.physics.named.model.body_mass.copy()
        PhysicsParameterShiftWrapper(env, mass_mult=2.0)
        # mass should have been doubled in-place
        np.testing.assert_allclose(
            env.physics.named.model.body_mass, original_mass * 2.0
        )
