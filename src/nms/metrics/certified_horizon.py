from __future__ import annotations


def certified_horizon(trajectories, viol_fn, alpha: float, max_h: int) -> int:
    """Largest H with violation rate <= alpha."""
    best_h = 0
    n = len(trajectories)
    if n == 0:
        return 0
    for h in range(1, max_h + 1):
        rate = sum(int(viol_fn(traj, h)) for traj in trajectories) / n
        if rate <= alpha:
            best_h = h
        else:
            break
    return best_h
