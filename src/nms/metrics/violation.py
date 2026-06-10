from __future__ import annotations


def violation_indicator(trajectory, viol_fn, horizon: int) -> int:
    """Viol_H indicator on one trajectory."""
    return int(viol_fn(trajectory, horizon))
