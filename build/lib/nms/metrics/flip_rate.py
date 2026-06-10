from __future__ import annotations

import numpy as np


def interface_flip_rate(interface_seq: np.ndarray) -> float:
    """P(I(a_t) != I(a_{t-1})) on a single trajectory."""
    seq = np.asarray(interface_seq)
    if seq.size < 2:
        return 0.0
    return float(np.mean(seq[1:] != seq[:-1]))
