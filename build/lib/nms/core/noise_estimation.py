from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class AleatoricEstimator(Protocol):
    """Estimator interface for aleatoric noise level û_alea."""

    def estimate(self, history: npt.NDArray[Any]) -> float: ...


class NoiseResponseMap(Protocol):
    """Interface for unconstrained noise-response map m(ν)."""

    def __call__(self, nu: float) -> float: ...


def linear_response(c: float = 1.0) -> NoiseResponseMap:
    """Linear response map m(ν)=cν."""
    return lambda nu: c * nu


def sqrt_response(c: float = 1.0) -> NoiseResponseMap:
    """Square-root response map m(ν)=c*sqrt(max(ν,0))."""
    return lambda nu: c * np.sqrt(max(nu, 0.0))
