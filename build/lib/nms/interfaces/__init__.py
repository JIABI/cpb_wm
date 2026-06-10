"""Abstract interfaces for NMS world model integration.

Exports
-------
KMeansActionDiscretizer : k-means discretizer mapping continuous → discrete actions.
fit_and_save            : Fit and persist a KMeansActionDiscretizer.
load_interface          : Load a persisted KMeansActionDiscretizer.
predict_batch           : Batch encode continuous actions to discrete indices.
"""
from nms.interfaces.kmeans_interface import (
    KMeansActionDiscretizer,
    fit_and_save,
    load_interface,
    predict_batch,
)

__all__ = [
    "KMeansActionDiscretizer",
    "fit_and_save",
    "load_interface",
    "predict_batch",
]
