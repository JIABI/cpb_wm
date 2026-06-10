import pytest

from nms.core.envelope import Bandwidth
from nms.core.projection import (
    BandwidthEmptyError,
    project_onto_bandwidth,
    project_onto_components,
)


def test_projection_inside_bandwidth() -> None:
    bw = Bandwidth(0.05, 0.2, nu=0.1, horizon=10, alpha=0.1)
    assert project_onto_bandwidth(0.1, bw) == pytest.approx(0.1)


def test_projection_clips_upper_and_lower() -> None:
    bw = Bandwidth(0.05, 0.2, nu=0.1, horizon=10, alpha=0.1)
    assert project_onto_bandwidth(0.3, bw) == pytest.approx(0.2)
    assert project_onto_bandwidth(0.01, bw) == pytest.approx(0.05)


def test_projection_raises_on_empty_bandwidth() -> None:
    bw = Bandwidth(0.3, 0.2, nu=0.1, horizon=10, alpha=0.1)
    with pytest.raises(BandwidthEmptyError):
        project_onto_bandwidth(0.1, bw)


# ── project_onto_components ──────────────────────────────────────────────────

def test_project_onto_components_single_interval() -> None:
    """Single component behaves exactly like project_onto_bandwidth."""
    comps = ((0.1, 0.4),)
    val, idx = project_onto_components(0.25, comps)
    assert val == pytest.approx(0.25)
    assert idx == 0


def test_project_onto_components_clips_to_single() -> None:
    comps = ((0.1, 0.4),)
    val, idx = project_onto_components(0.9, comps)
    assert val == pytest.approx(0.4)
    assert idx == 0

    val2, idx2 = project_onto_components(0.0, comps)
    assert val2 == pytest.approx(0.1)
    assert idx2 == 0


def test_project_onto_components_chooses_nearest() -> None:
    """Two disjoint intervals — project picks the nearest one."""
    # intervals: [0.0, 0.2] and [0.6, 0.8]
    comps = ((0.0, 0.2), (0.6, 0.8))
    # target = 0.5 → dist to [0.0,0.2] is 0.3, dist to [0.6,0.8] is 0.1
    val, idx = project_onto_components(0.5, comps)
    assert val == pytest.approx(0.6)
    assert idx == 1


def test_project_onto_components_tiebreak_by_width() -> None:
    """On equidistant target, prefer the wider component."""
    # [0.0, 0.1] width=0.1  and  [0.3, 0.5] width=0.2
    # target = 0.2 → dist to [0.0,0.1] is 0.1, dist to [0.3,0.5] is 0.1 (tie)
    comps = ((0.0, 0.1), (0.3, 0.5))
    val, idx = project_onto_components(0.2, comps)
    assert idx == 1          # wider component wins
    assert val == pytest.approx(0.3)


def test_project_onto_components_target_inside_one() -> None:
    """Target inside one of two intervals → zero distance to that one."""
    comps = ((0.0, 0.2), (0.5, 0.8))
    val, idx = project_onto_components(0.7, comps)
    assert val == pytest.approx(0.7)
    assert idx == 1


def test_project_onto_components_empty_raises() -> None:
    with pytest.raises(BandwidthEmptyError):
        project_onto_components(0.5, ())
