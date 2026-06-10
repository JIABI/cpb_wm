"""World model wrappers for NMS integration.

Phase A  — DreamerV3 baseline wrapper + stochastic environment wrappers.
Phase B  — NMS infrastructure (k-means interface, violation detection, CPB).
Phase 2.5 — EnsembleNuEstimator: state-dependent ν̂ from K-member ensemble.
"""
from nms.wm._dreamerv3_path import (  # noqa: F401
    DREAMERV3_AVAILABLE,
    DREAMERV3_COMMIT_SHA,
    DREAMERV3_PATH,
)
