from nms.metrics.recovery import recovery_time


def test_recovery_time() -> None:
    traj = [1.0, 1.0, 0.2, 0.4, 0.95]
    assert recovery_time(traj, perturb_step=2, baseline_value=1.0, tolerance=0.1) == 2
