"""Stochastic environment wrappers for DM Control suite (Brief B Phase A).

Four wrapper types are provided, each parameterised by σ (stochasticity level):

  ObservationNoiseWrapper
    Adds i.i.d. Gaussian noise N(0, σ²) to every observation component.

  RewardStochasticityWrapper
    Adds i.i.d. Gaussian noise N(0, σ²) to the scalar reward.

  DynamicsShiftWrapper
    Multiplies the action by (1 + ε·σ) at each step where ε ~ N(0, 1).
    Models distribution shift in the physical dynamics.

  StochasticDynamicsWrapper
    Full-package wrapper: composes ObservationNoise + RewardStochasticity +
    DynamicsShift in one call.

All wrappers follow the gymnasium.Env interface and preserve the observation /
action / reward dtypes of the wrapped environment.

Usage
-----
    import dm_control.suite as suite
    from nms.envs.stochastic_dm_control import StochasticDynamicsWrapper

    base_env = suite.load("cheetah", "run")
    noisy_env = StochasticDynamicsWrapper(base_env, sigma=0.1, seed=42)
    obs, info = noisy_env.reset()

Notes
-----
  * Wrappers use np.random.default_rng (never np.random.seed).
  * σ = 0 recovers the original deterministic environment exactly.
  * The wrappers do NOT use gymnasium's built-in seed mechanism for their
    internal RNG — they maintain their own Generator for reproducibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Defer the gymnasium import so this module can be imported even when
# gymnasium is not installed (e.g., during unit tests that mock the env).
try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover
    gym = None  # type: ignore[assignment]


class _BaseStochasticWrapper:
    """Minimal gymnasium-compatible wrapper base."""

    def __init__(self, env: Any, sigma: float, seed: int | None = None) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be ≥ 0, got {sigma}")
        self.env = env
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

        # Forward gymnasium-required attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    # ── pass-through delegation ───────────────────────────────────────────────

    def reset(self, **kwargs: Any) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        return self.env.reset(**kwargs)  # type: ignore[return-value]

    def step(self, action: Any) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        return self.env.step(action)  # type: ignore[return-value]

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        # Fall back to wrapped env for any attribute not defined here.
        return getattr(self.env, name)


class ObservationNoiseWrapper(_BaseStochasticWrapper):
    """Adds i.i.d. Gaussian noise to observations.

    obs_noisy = obs + N(0, σ²·I)

    Parameters
    ----------
    env   : gymnasium-compatible environment.
    sigma : Standard deviation of observation noise (σ = 0 → no noise).
    seed  : Optional RNG seed for reproducibility.
    """

    def reset(self, **kwargs: Any) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info

    def step(self, action: Any) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, terminated, truncated, info

    def _add_noise(self, obs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.sigma == 0.0:
            return obs
        noise = self._rng.normal(0.0, self.sigma, size=np.asarray(obs).shape)
        return (np.asarray(obs) + noise).astype(obs.dtype)


class RewardStochasticityWrapper(_BaseStochasticWrapper):
    """Adds i.i.d. Gaussian noise to the scalar reward.

    r_noisy = r + N(0, σ²)

    Parameters
    ----------
    env   : gymnasium-compatible environment.
    sigma : Standard deviation of reward noise (σ = 0 → no noise).
    seed  : Optional RNG seed for reproducibility.
    """

    def step(self, action: Any) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.sigma > 0.0:
            reward = float(reward) + float(self._rng.normal(0.0, self.sigma))
        return obs, float(reward), terminated, truncated, info


class DynamicsShiftWrapper(_BaseStochasticWrapper):
    """Multiplicative action-space noise modelling dynamics shift.

    a_applied = a * (1 + ε·σ),   ε ~ N(0, 1)

    The action is clipped to the action_space bounds after perturbation.

    Parameters
    ----------
    env   : gymnasium-compatible environment.
    sigma : Scale of the multiplicative dynamics noise (σ = 0 → no shift).
    seed  : Optional RNG seed for reproducibility.
    """

    def step(self, action: Any) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        if self.sigma > 0.0:
            action = np.asarray(action, dtype=float)
            epsilon = self._rng.normal(0.0, 1.0, size=action.shape)
            action = action * (1.0 + epsilon * self.sigma)
            if hasattr(self.action_space, "low"):
                action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)  # type: ignore[return-value]


class PhysicsParameterShiftWrapper:
    """For Phase D OOD testing: perturb DM Control physics parameters at env-init time.

    Multiplies body masses, geom friction and DOF damping by the supplied
    factors.  This modifies the underlying ``dm_control`` physics object
    **in-place**, so the effect persists for the lifetime of the wrapped env.

    Falls back gracefully (no-op) when the wrapped env does not expose a
    ``physics`` attribute (e.g. mocked envs, non-dm_control backends).

    Parameters
    ----------
    env           : gymnasium-compatible environment wrapping a dm_control task.
    mass_mult     : Multiplicative factor for all body masses (default 1.0 = no change).
    friction_mult : Multiplicative factor for geom friction coefficients.
    damping_mult  : Multiplicative factor for DOF damping coefficients.
    """

    def __init__(
        self,
        env: Any,
        mass_mult: float = 1.0,
        friction_mult: float = 1.0,
        damping_mult: float = 1.0,
    ) -> None:
        self.env = env
        self.mass_mult = float(mass_mult)
        self.friction_mult = float(friction_mult)
        self.damping_mult = float(damping_mult)

        # Apply parameter perturbations immediately
        physics = getattr(getattr(env, "unwrapped", env), "physics", None)
        if physics is not None:
            try:
                physics.named.model.body_mass[:] *= self.mass_mult
                physics.named.model.geom_friction[:, 0] *= self.friction_mult
                physics.named.model.dof_damping[:] *= self.damping_mult
            except (AttributeError, IndexError):
                pass  # non-standard physics interface — skip silently

        # Forward gymnasium-required attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, **kwargs: Any) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        return self.env.reset(**kwargs)  # type: ignore[return-value]

    def step(
        self, action: Any
    ) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        return self.env.step(action)  # type: ignore[return-value]

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_stochastic_dmc_env(
    suite: str,
    task: str,
    sigma_obs: float = 0.0,
    sigma_reward: float = 0.0,
    sigma_dynamics: float = 0.0,
    seed: int | None = None,
) -> StochasticDynamicsWrapper:
    """Factory: build a dm_control env wrapped in StochasticDynamicsWrapper.

    Handles dm_control → gymnasium bridging via shimmy (preferred) or raw env.

    Parameters
    ----------
    suite           : dm_control suite name (e.g. "cheetah").
    task            : dm_control task name (e.g. "run").
    sigma_obs       : Observation noise std.
    sigma_reward    : Reward noise std.
    sigma_dynamics  : Dynamics shift scale.
    seed            : Optional RNG seed.

    Returns
    -------
    StochasticDynamicsWrapper wrapping the requested dm_control task.

    Raises
    ------
    ImportError if dm_control is not installed.
    """
    try:
        import dm_control.suite as dmc_suite  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "dm_control is required. pip install dm_control"
        ) from exc

    raw_env = dmc_suite.load(suite, task)

    try:
        from shimmy import DMControlCompatibilityV0  # type: ignore[import-not-found]
        base_env = DMControlCompatibilityV0(raw_env)
    except ModuleNotFoundError:
        import warnings
        warnings.warn(
            "shimmy not installed — using dm_control env directly. "
            "pip install 'shimmy[dm-control]>=2.0' for full gymnasium compat.",
            stacklevel=2,
        )
        base_env = raw_env  # type: ignore[assignment]

    return StochasticDynamicsWrapper(
        base_env,
        sigma_obs=sigma_obs,
        sigma_reward=sigma_reward,
        sigma_dynamics=sigma_dynamics,
        seed=seed,
    )


class StochasticDynamicsWrapper(_BaseStochasticWrapper):
    """Full stochastic wrapper: observation noise + reward noise + dynamics shift.

    Equivalent to composing::

        env
        → DynamicsShiftWrapper(sigma_dynamics)
        → ObservationNoiseWrapper(sigma_obs)
        → RewardStochasticityWrapper(sigma_reward)

    For convenience, a single ``sigma`` parameter scales all three noise types
    equally.  Use ``sigma_obs``, ``sigma_reward``, ``sigma_dynamics`` to set
    them independently.

    Parameters
    ----------
    env             : gymnasium-compatible environment.
    sigma           : Default noise level for all components (overridden by
                      per-component kwargs).
    sigma_obs       : Observation noise std (overrides sigma).
    sigma_reward    : Reward noise std (overrides sigma).
    sigma_dynamics  : Dynamics shift scale (overrides sigma).
    seed            : Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        env: Any,
        sigma: float = 0.0,
        sigma_obs: float | None = None,
        sigma_reward: float | None = None,
        sigma_dynamics: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env, sigma=sigma, seed=seed)
        self._sigma_obs = float(sigma_obs if sigma_obs is not None else sigma)
        self._sigma_reward = float(sigma_reward if sigma_reward is not None else sigma)
        self._sigma_dynamics = float(sigma_dynamics if sigma_dynamics is not None else sigma)

    def reset(self, **kwargs: Any) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._obs_noise(obs), info

    def step(self, action: Any) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:
        # 1. Apply dynamics shift
        if self._sigma_dynamics > 0.0:
            action = np.asarray(action, dtype=float)
            eps = self._rng.normal(0.0, 1.0, size=action.shape)
            action = action * (1.0 + eps * self._sigma_dynamics)
            if hasattr(self.action_space, "low"):
                action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. Add observation noise
        obs = self._obs_noise(obs)

        # 3. Add reward noise
        if self._sigma_reward > 0.0:
            reward = float(reward) + float(self._rng.normal(0.0, self._sigma_reward))

        return obs, float(reward), terminated, truncated, info

    def _obs_noise(self, obs: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self._sigma_obs == 0.0:
            return obs
        noise = self._rng.normal(0.0, self._sigma_obs, size=np.asarray(obs).shape)
        return (np.asarray(obs) + noise).astype(obs.dtype)
