from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import beta  # type: ignore[import-untyped,unused-ignore]


def conformal_sublevel_endpoints(
    sigmas: npt.NDArray[Any],
    violations: npt.NDArray[Any],  # shape (n_sigma, n_calibration); 0/1
    alpha: float,
) -> tuple[float, float]:
    """Estimate admissible sigma endpoints from calibrated violation samples."""
    sigmas = np.asarray(sigmas, dtype=float)
    violations = np.asarray(violations, dtype=int)
    if violations.ndim != 2 or violations.shape[0] != sigmas.shape[0]:
        raise ValueError("violations must have shape (n_sigma, n_calibration)")

    n = violations.shape[1]
    upper_bounds = []
    for row in violations:
        k = int(np.sum(row))
        ub = 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))
        upper_bounds.append(ub)
    admissible = np.asarray(upper_bounds) <= alpha
    if not np.any(admissible):
        return (1.0, 0.0)
    idx = np.flatnonzero(admissible)
    return float(sigmas[idx[0]]), float(sigmas[idx[-1]])
