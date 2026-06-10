import numpy as np

from nms.metrics.flip_rate import interface_flip_rate


def test_interface_flip_rate() -> None:
    seq = np.array([0, 1, 1, 0])
    assert interface_flip_rate(seq) == 2 / 3
