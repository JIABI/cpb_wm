from nms.metrics.certified_horizon import certified_horizon


def test_certified_horizon() -> None:
    trajectories = [0, 1, 2, 3]

    def viol_fn(traj, h):
        return int(traj < h)

    assert certified_horizon(trajectories, viol_fn, alpha=0.5, max_h=5) == 2
