"""KMeans action discretizer interface for Exp 2.

Thin re-export and extension of KMeansActionDiscretizer from dreamer_wrapper,
adding batch predict and persistence helpers used by fit_interface.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from nms.wm.dreamer_wrapper import KMeansActionDiscretizer

__all__ = ["KMeansActionDiscretizer", "fit_and_save", "load_interface"]


def fit_and_save(
    actions: npt.NDArray[np.float64],
    save_path: str | Path,
    n_clusters: int = 20,
    random_state: int = 42,
) -> KMeansActionDiscretizer:
    """Fit a KMeansActionDiscretizer on actions and persist to disk.

    Parameters
    ----------
    actions      : (N, action_dim) array of collected exploration actions.
    save_path    : Path prefix (without extension) for the .npz file.
    n_clusters   : Number of k-means centroids.
    random_state : Seed for sklearn KMeans.

    Returns
    -------
    Fitted KMeansActionDiscretizer.
    """
    km = KMeansActionDiscretizer(n_clusters=n_clusters, random_state=random_state)
    km.fit(actions)
    path = Path(str(save_path) if not str(save_path).endswith(".npz") else save_path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    km.save(path)
    return km


def load_interface(path: str | Path) -> KMeansActionDiscretizer:
    """Load a KMeansActionDiscretizer from a .npz file."""
    return KMeansActionDiscretizer.load(path)


def predict_batch(
    km: KMeansActionDiscretizer,
    actions: npt.NDArray[np.float64],
) -> npt.NDArray[np.intp]:
    """Encode a batch of continuous actions → discrete indices.

    Parameters
    ----------
    km      : Fitted KMeansActionDiscretizer.
    actions : (N, action_dim) float array.

    Returns
    -------
    (N,) int array of centroid indices.
    """
    if not km.is_fitted:
        raise RuntimeError("KMeansActionDiscretizer is not fitted.")
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[:, np.newaxis]
    return np.array([km.encode(a) for a in actions], dtype=np.intp)
