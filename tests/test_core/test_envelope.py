import numpy as np
import pytest

from nms.core.envelope import Bandwidth, estimate_bandwidth


def test_bandwidth_helpers() -> None:
    bw = Bandwidth(0.1, 0.2, nu=0.1, horizon=10, alpha=0.1)
    assert not bw.is_empty()
    assert bw.contains(0.15)
    assert not bw.contains(0.3)


def test_bandwidth_defaults() -> None:
    """New fields have defaults so old construction still works."""
    bw = Bandwidth(0.0, 0.5, nu=0.2, horizon=100, alpha=0.1)
    assert bw.is_quasiconvex is True
    assert bw.n_components == 1
    assert bw.left_censored is False
    assert bw.right_censored is False
    assert bw.components == ()


def test_estimate_bandwidth_contiguous() -> None:
    """Connected admissible set — endpoints match, quasiconvex flag True."""
    grid = np.array([0.0, 0.1, 0.2, 0.3])
    risks = {0.0: 0.2, 0.1: 0.05, 0.2: 0.09, 0.3: 0.12}
    bw = estimate_bandwidth(grid, lambda s: risks[round(s, 1)], nu=0.2, horizon=50, alpha=0.1)
    assert bw.sigma_min == pytest.approx(0.1)
    assert bw.sigma_max == pytest.approx(0.2)
    assert bw.is_quasiconvex is True
    assert bw.n_components == 1
    # Not at grid boundary → not censored
    assert bw.left_censored is False
    assert bw.right_censored is False
    # components tuple stores the single interval
    assert len(bw.components) == 1
    assert bw.components[0][0] == pytest.approx(0.1)
    assert bw.components[0][1] == pytest.approx(0.2)


def test_estimate_bandwidth_empty() -> None:
    """No admissible sigma — returns empty bandwidth, n_components=0."""
    grid = np.array([0.0, 0.1, 0.2])
    bw = estimate_bandwidth(grid, lambda _s: 0.5, nu=0.1, horizon=10, alpha=0.1)
    assert bw.is_empty()
    assert bw.n_components == 0
    assert bw.components == ()
    assert bw.left_censored is False
    assert bw.right_censored is False


def test_estimate_bandwidth_non_contiguous() -> None:
    """Disconnected admissible set — returns the *largest* component.

    Grid:  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    Risks: admissible at indices 0 (run of 1) and 2,3,4 (run of 3).
    The larger run is [0.2, 0.3, 0.4].
    """
    grid = np.linspace(0.0, 0.5, 6)  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    risk_map = {
        0.0: 0.05,   # admissible  (component A — length 1)
        0.1: 0.20,   # not admissible
        0.2: 0.08,   # admissible  (component B — length 3)
        0.3: 0.09,   # admissible
        0.4: 0.07,   # admissible
        0.5: 0.30,   # not admissible
    }

    def risk_fn(s: float) -> float:
        return risk_map[round(s, 1)]

    bw = estimate_bandwidth(grid, risk_fn, nu=0.3, horizon=50, alpha=0.1)

    # Largest component is [0.2, 0.4]
    assert bw.sigma_min == pytest.approx(0.2)
    assert bw.sigma_max == pytest.approx(0.4)
    assert bw.is_quasiconvex is False
    assert bw.n_components == 2
    # Both components stored
    assert len(bw.components) == 2
    # Component A: [0.0, 0.0]; Component B: [0.2, 0.4] — order by run index
    comp_lows = {c[0] for c in bw.components}
    assert 0.0 in comp_lows or pytest.approx(0.0) in comp_lows
    assert any(abs(c[0] - 0.2) < 1e-9 for c in bw.components)


def test_estimate_bandwidth_left_censored() -> None:
    """sigma_min == grid_min → left_censored=True."""
    grid = np.array([0.0, 0.1, 0.2, 0.3])
    # admissible from 0.0 (grid_min) onward
    risks = {0.0: 0.05, 0.1: 0.05, 0.2: 0.15, 0.3: 0.30}
    bw = estimate_bandwidth(grid, lambda s: risks[round(s, 1)], nu=0.1, horizon=10, alpha=0.1)
    assert bw.left_censored is True
    assert bw.right_censored is False


def test_estimate_bandwidth_right_censored() -> None:
    """sigma_max == grid_max → right_censored=True."""
    grid = np.array([0.0, 0.1, 0.2, 0.3])
    # admissible all the way to 0.3 (grid_max)
    risks = {0.0: 0.30, 0.1: 0.05, 0.2: 0.05, 0.3: 0.05}
    bw = estimate_bandwidth(grid, lambda s: risks[round(s, 1)], nu=0.1, horizon=10, alpha=0.1)
    assert bw.right_censored is True
    assert bw.left_censored is False
