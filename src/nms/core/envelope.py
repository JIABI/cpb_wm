from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bandwidth:
    """Certified stochasticity bandwidth B_{H,α}(ν) = [sigma_min, sigma_max].

    sigma_min == 0 means zero stochasticity is admissible (no Necessary Stochasticity).
    sigma_min > 0  means Theorem 1 Necessary Stochasticity is active.
    is_empty()     means no feasible sigma exists under (H, α, ν).

    is_quasiconvex=False and n_components>1 indicate a disconnected sublevel set —
    the quasi-convexity assumption (Sec 5.1) is violated for this triple.

    left_censored=True  means sigma_min == grid_min  (lower endpoint not confirmed).
    right_censored=True means sigma_max == grid_max  (upper endpoint not confirmed;
        report as "σ_max ≥ grid_max" rather than "σ_max = grid_max").

    components stores all admissible intervals as (lo, hi) pairs so callers can
    project onto the full union when n_components > 1.
    """

    sigma_min: float
    sigma_max: float
    nu: float
    horizon: int
    alpha: float
    is_quasiconvex: bool = True
    n_components: int = 1
    left_censored: bool = False
    right_censored: bool = False
    components: tuple[tuple[float, float], ...] = ()

    def is_empty(self) -> bool:
        return self.sigma_min > self.sigma_max

    def contains(self, sigma: float) -> bool:
        return self.sigma_min <= sigma <= self.sigma_max


class ViolationFn(Protocol):
    """Violation indicator interface."""

    def __call__(self, trajectory: npt.NDArray[Any], horizon: int) -> int: ...


def estimate_bandwidth(
    sigma_grid: npt.NDArray[Any],
    risk_estimator: Callable[[float], float],
    nu: float,
    horizon: int,
    alpha: float,
) -> Bandwidth:
    """Estimate the certified bandwidth from a sigma grid and empirical risk curve.

    Algorithm
    ---------
    1. Mark each grid point admissible iff risk_estimator(sigma) <= alpha.
    2. Split into maximal consecutive runs (connected components).
    3. Return the *largest* run as the bandwidth interval.
       Disconnected → is_quasiconvex=False, warning logged.
    4. Set left_censored / right_censored flags when the interval abuts the
       grid boundary (upper endpoint not empirically confirmed).
    5. Return empty Bandwidth (sigma_min > sigma_max) when nothing is admissible.
    """
    sorted_grid = np.sort(np.asarray(sigma_grid, dtype=float))
    grid_min = float(sorted_grid[0])
    grid_max = float(sorted_grid[-1])

    admissible = np.array(
        [risk_estimator(float(s)) <= alpha for s in sorted_grid], dtype=bool
    )

    if not np.any(admissible):
        return Bandwidth(
            1.0, 0.0, nu=nu, horizon=horizon, alpha=alpha,
            is_quasiconvex=True, n_components=0,
            left_censored=False, right_censored=False, components=(),
        )

    idx = np.flatnonzero(admissible)
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1)

    n_components = len(runs)
    is_quasiconvex = n_components == 1
    if not is_quasiconvex:
        LOGGER.warning(
            "Non-contiguous admissible sublevel set (n_components=%d) "
            "for nu=%.4f, horizon=%d, alpha=%.4f — returning largest component.",
            n_components, nu, horizon, alpha,
        )

    comps: tuple[tuple[float, float], ...] = tuple(
        (float(sorted_grid[r[0]]), float(sorted_grid[r[-1]])) for r in runs
    )

    largest = max(runs, key=len)
    sigma_min = float(sorted_grid[largest[0]])
    sigma_max = float(sorted_grid[largest[-1]])

    return Bandwidth(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        nu=nu,
        horizon=horizon,
        alpha=alpha,
        is_quasiconvex=is_quasiconvex,
        n_components=n_components,
        left_censored=sigma_min <= grid_min + 1e-12,
        right_censored=sigma_max >= grid_max - 1e-12,
        components=comps,
    )
