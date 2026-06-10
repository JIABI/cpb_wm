from nms.metrics.violation import violation_indicator


def test_violation_indicator() -> None:
    assert violation_indicator([1, 2], lambda _traj, _h: 1, horizon=10) == 1
