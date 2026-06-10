"""World-model violation metrics for Exp 2 (Phase B+).

Provides:

  imagined_realized_disagreement(imagined, realized, kmeans) → float
    Fraction of steps where imagined and realized discrete actions differ.

  per_step_disagreement_rate(imagined_batch, realized_batch, kmeans) → ndarray
    Per-step disagreement rates averaged over a batch of trajectory pairs.

  violation_indicator_wm(imagined, realized, kmeans, horizon) → int
    Binary indicator: 1 if any step within horizon has disagreement.

  batch_violation_rate(imagined_batch, realized_batch, kmeans, horizon) → float
    Empirical R_H over a batch of trajectory pairs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _encode_batch(
    actions: npt.NDArray[Any],
    kmeans: Any,
) -> npt.NDArray[np.intp]:
    """Encode (H, act_dim) continuous actions → (H,) discrete indices."""
    if actions.ndim == 1:
        actions = actions[:, np.newaxis]
    return np.array([kmeans.encode(a) for a in actions], dtype=np.intp)


def imagined_realized_disagreement(
    imagined: dict[str, npt.NDArray[Any]],
    realized: dict[str, npt.NDArray[Any]],
    kmeans: Any,
) -> float:
    """Fraction of steps where imagined and realized discrete actions differ.

    Parameters
    ----------
    imagined : dict with ``actions`` key, shape (H, act_dim).
    realized : dict with ``actions`` key, shape (H, act_dim).
    kmeans   : Fitted KMeansActionDiscretizer.

    Returns
    -------
    Float ∈ [0, 1]: fraction of disagreeing steps.
    """
    img_acts = np.asarray(imagined["actions"])
    rea_acts = np.asarray(realized["actions"])

    if img_acts.ndim == 1:
        img_acts = img_acts[:, np.newaxis]
    if rea_acts.ndim == 1:
        rea_acts = rea_acts[:, np.newaxis]

    h = min(len(img_acts), len(rea_acts))
    if h == 0:
        return 0.0

    img_idx = _encode_batch(img_acts[:h], kmeans)
    rea_idx = _encode_batch(rea_acts[:h], kmeans)
    return float(np.mean(img_idx != rea_idx))


def per_step_disagreement_rate(
    imagined_batch: list[dict[str, npt.NDArray[Any]]],
    realized_batch: list[dict[str, npt.NDArray[Any]]],
    kmeans: Any,
) -> npt.NDArray[np.float64]:
    """Per-step disagreement rates averaged over a batch of trajectory pairs.

    Parameters
    ----------
    imagined_batch : List of N imagined trajectory dicts (``actions`` key).
    realized_batch : List of N realized trajectory dicts (``actions`` key).
    kmeans         : Fitted KMeansActionDiscretizer.

    Returns
    -------
    (H,) float array of per-step disagreement rates.
    """
    if len(imagined_batch) != len(realized_batch):
        raise ValueError("imagined_batch and realized_batch must have the same length.")
    if not imagined_batch:
        return np.zeros(0, dtype=np.float64)

    # Determine horizon from first pair
    h = min(
        len(imagined_batch[0]["actions"]),
        len(realized_batch[0]["actions"]),
    )
    step_disag = np.zeros((len(imagined_batch), h), dtype=np.float64)

    for n, (img, rea) in enumerate(zip(imagined_batch, realized_batch, strict=False)):
        img_acts = np.asarray(img["actions"])
        rea_acts = np.asarray(rea["actions"])
        if img_acts.ndim == 1:
            img_acts = img_acts[:, np.newaxis]
        if rea_acts.ndim == 1:
            rea_acts = rea_acts[:, np.newaxis]
        hn = min(h, len(img_acts), len(rea_acts))
        img_idx = _encode_batch(img_acts[:hn], kmeans)
        rea_idx = _encode_batch(rea_acts[:hn], kmeans)
        step_disag[n, :hn] = (img_idx != rea_idx).astype(float)

    return step_disag.mean(axis=0)


def violation_indicator_wm(
    imagined: dict[str, npt.NDArray[Any]],
    realized: dict[str, npt.NDArray[Any]],
    kmeans: Any,
    horizon: int,
) -> int:
    """Binary indicator: 1 if any step ≤ horizon has imagined/realized disagreement.

    Parameters
    ----------
    imagined : dict with ``actions`` key.
    realized : dict with ``actions`` key.
    kmeans   : Fitted KMeansActionDiscretizer.
    horizon  : Max step to check (H).

    Returns
    -------
    0 or 1.
    """
    img_acts = np.asarray(imagined["actions"])[:horizon]
    rea_acts = np.asarray(realized["actions"])[:horizon]

    if img_acts.ndim == 1:
        img_acts = img_acts[:, np.newaxis]
    if rea_acts.ndim == 1:
        rea_acts = rea_acts[:, np.newaxis]

    h = min(len(img_acts), len(rea_acts))
    if h == 0:
        return 0

    img_idx = _encode_batch(img_acts[:h], kmeans)
    rea_idx = _encode_batch(rea_acts[:h], kmeans)
    return int(np.any(img_idx != rea_idx))


def batch_violation_rate(
    imagined_batch: list[dict[str, npt.NDArray[Any]]],
    realized_batch: list[dict[str, npt.NDArray[Any]]],
    kmeans: Any,
    horizon: int,
) -> float:
    """Empirical R_H over a batch of trajectory pairs.

    R_H = (1/N) Σ_n 1[any step ≤ H disagrees]

    Parameters
    ----------
    imagined_batch : List of N imagined trajectory dicts.
    realized_batch : List of N realized trajectory dicts.
    kmeans         : Fitted KMeansActionDiscretizer.
    horizon        : Max step to check.

    Returns
    -------
    Float ∈ [0, 1].
    """
    if len(imagined_batch) != len(realized_batch):
        raise ValueError("Batch sizes must match.")
    if not imagined_batch:
        return 0.0
    indicators = [
        violation_indicator_wm(img, rea, kmeans, horizon)
        for img, rea in zip(imagined_batch, realized_batch, strict=False)
    ]
    return float(np.mean(indicators))
