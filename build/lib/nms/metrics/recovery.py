from __future__ import annotations

import numpy as np


def recovery_time(
    trajectory, perturb_step: int, baseline_value: float, tolerance: float
) -> int:
    """Steps needed after perturbation to recover into baseline ± tolerance."""
    arr = np.asarray(trajectory, dtype=float)
    target_min = baseline_value - tolerance
    target_max = baseline_value + tolerance
    for idx in range(perturb_step + 1, arr.size):
        if target_min <= arr[idx] <= target_max:
            return idx - perturb_step
    return -1
