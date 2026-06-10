from __future__ import annotations

import numpy as np

from .envelope import Bandwidth


class BandwidthEmptyError(Exception):
    """Raised when projection is attempted on an empty bandwidth."""


def project_onto_bandwidth(m_nu: float, bandwidth: Bandwidth) -> float:
    """NMS projection operator: σ^NMS = clip(m(ν), σ_min, σ_max)."""
    if bandwidth.is_empty():
        raise BandwidthEmptyError("No feasible sigma: bandwidth is empty.")
    return float(np.clip(m_nu, bandwidth.sigma_min, bandwidth.sigma_max))


def project_onto_components(
    m_nu: float,
    components: tuple[tuple[float, float], ...],
) -> tuple[float, int]:
    """Project m_nu onto the union of admissible intervals.

    Each component is a (lo, hi) pair.  For each component the nearest
    feasible point is: clip(m_nu, lo, hi).  We choose the component whose
    nearest point is closest to m_nu.  Ties are broken by component width
    (largest interval wins — most stable choice).

    Parameters
    ----------
    m_nu       : Target value to project (e.g. noise-response map output).
    components : Admissible intervals as ``((lo₀, hi₀), (lo₁, hi₁), …)``.

    Returns
    -------
    (projected_value, chosen_component_idx)

    Raises
    ------
    BandwidthEmptyError
        If ``components`` is empty.
    """
    if not components:
        raise BandwidthEmptyError("No admissible components to project onto.")

    best_val = float("nan")
    best_idx = 0
    best_dist = float("inf")
    best_width = -1.0

    for i, (lo, hi) in enumerate(components):
        proj = float(np.clip(m_nu, lo, hi))
        dist = abs(m_nu - proj)
        width = hi - lo
        # Prefer smaller distance; break ties by larger width
        if dist < best_dist - 1e-14 or (
            abs(dist - best_dist) <= 1e-14 and width > best_width
        ):
            best_val = proj
            best_idx = i
            best_dist = dist
            best_width = width

    return best_val, best_idx
