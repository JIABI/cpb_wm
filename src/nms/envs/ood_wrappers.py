"""Out-of-distribution (OOD) environment wrappers for Exp 2 Phase 8/9.

Provides wrappers that perturb the underlying dm_control physics parameters
to simulate distribution shift at test time.  All wrappers are applied
**once at construction** — the model parameters persist through resets
because dm_control resets state (qpos/qvel) but does not reinitialize the
MJCF model.

Classes
-------
MassFrictionPerturbWrapper
    Scales all body masses by (1 + δ) and all geom friction coefficients by
    max(1 − δ, 0.05).  δ = 0 → identity (no perturbation).
    Larger δ makes the body heavier and more slippery simultaneously.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class MassFrictionPerturbWrapper:
    """OOD wrapper: scale body mass by (1+δ), friction by max(1−δ, 0.05).

    Applies the perturbation to the underlying dm_control physics model
    **once at construction time**.  The change persists for the lifetime
    of the wrapper because dm_control does not reinitialise model params
    on reset().

    Parameters
    ----------
    env   : gymnasium-compatible environment wrapping a dm_control task.
    delta : Perturbation magnitude (0 = identity, 1 = double mass + near-zero
            friction).  Must be ≥ 0.

    Notes
    -----
    *  If ``env`` does not expose a ``physics`` attribute (e.g. mock envs),
       the wrapper behaves as a pure pass-through.
    *  Friction is clamped to ≥ 0.05 so the physics solver remains stable.
    """

    def __init__(self, env: Any, delta: float) -> None:
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        self.env = env
        self.delta = float(delta)

        # Forward gymnasium-required attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

        # Apply parameter perturbations at construction time
        self._apply_perturbation()

    def _apply_perturbation(self) -> None:
        """Modify dm_control physics model parameters in-place."""
        if self.delta == 0.0:
            return  # identity — nothing to do

        mass_mult = 1.0 + self.delta
        friction_mult = max(1.0 - self.delta, 0.05)

        # Try multiple access patterns for dm_control physics
        physics = self._get_physics()
        if physics is None:
            return  # non-dm_control env — silent pass-through

        try:
            model = physics.model if hasattr(physics, "model") else physics.named.model
            # Scale body masses
            if hasattr(model, "body_mass"):
                model.body_mass[:] = np.asarray(model.body_mass) * mass_mult
            # Scale geom friction (first column = sliding friction)
            if hasattr(model, "geom_friction"):
                model.geom_friction[:, 0] = np.clip(
                    np.asarray(model.geom_friction[:, 0]) * friction_mult, 0.05, None
                )
            print(
                f"[MassFrictionPerturbWrapper] δ={self.delta:.2f}: "
                f"mass ×{mass_mult:.2f}, friction ×{friction_mult:.4f}"
            )
        except (AttributeError, IndexError, TypeError) as exc:
            import warnings
            warnings.warn(
                f"MassFrictionPerturbWrapper: could not apply perturbation: {exc}",
                stacklevel=2,
            )

    def _get_physics(self) -> Any:
        """Extract dm_control physics object, traversing wrapper chain."""
        # Walk the wrapper chain (env → env.env → env.env.env → ...)
        # until we find an object with a `physics` attribute.
        seen = set()
        candidate: Any = self.env
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            # Check for physics directly
            physics = getattr(candidate, "physics", None)
            if physics is not None:
                return physics
            # shimmy DmControlCompatibilityV0 stores raw env as _env
            inner = getattr(candidate, "_env", None)
            if inner is not None and id(inner) not in seen:
                phys2 = getattr(inner, "physics", None)
                if phys2 is not None:
                    return phys2
            # Try gymnasium wrapper chain: .env
            candidate = getattr(candidate, "env", None)
        return None

    # ── gymnasium pass-through ────────────────────────────────────────────────

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
