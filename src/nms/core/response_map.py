"""Empirical response-map fitting for the NMS noise-response function m(ν).

Phase B of the v4 brief:
  - Given (ν_train, σ*_train) pairs, fit a parametric m: ν → σ* function.
  - Supported forms:
      "linear"  σ* = c·ν            (through origin)
      "sqrt"    σ* = c·√ν           (through origin)
      "log"     σ* = c·log(1+ν)     (through origin)
      "affine"  σ* = a + c·ν        (intercept + slope; best for
                                      games with a non-zero baseline
                                      forgiveness even at ν=0)
  - Returns a FittedResponseMap with R² on held-out test set.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

FormType = Literal["linear", "sqrt", "log", "affine"]

_FEATURES: dict[str, Callable[[float], float]] = {
    "linear": lambda nu: nu,
    "sqrt": lambda nu: float(np.sqrt(nu)),
    "log": lambda nu: float(np.log1p(nu)),
}


@dataclass
class FittedResponseMap:
    """Fitted noise-response map m: ν → σ*.

    Attributes
    ----------
    form        : Functional form ("linear" | "sqrt" | "log" | "affine").
    params      : Fitted coefficient(s).
                  - 1-tuple (c,)    for through-origin forms (linear/sqrt/log).
                  - 2-tuple (a, c)  for "affine": m(ν) = a + c·ν.
    r2_train    : R² on training ν values (for debugging only).
    r2_test     : R² on held-out test ν values (primary quality metric).
    train_nus   : ν values used for fitting.
    test_nus    : ν values used for evaluation.
    """

    form: str
    params: tuple[float, ...]
    r2_train: float
    r2_test: float
    train_nus: tuple[float, ...]
    test_nus: tuple[float, ...]

    def __call__(self, nu: float) -> float:
        """Evaluate m(ν)."""
        if self.form == "affine":
            a, c = self.params
            return float(a + c * nu)
        feat_fn = _FEATURES[self.form]
        return float(self.params[0] * feat_fn(nu))


def _r2(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Coefficient of determination R² = 1 − SS_res / SS_tot."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot < 1e-14:
        return 1.0 if ss_res < 1e-14 else 0.0
    return 1.0 - ss_res / ss_tot


def _fit_single_param(
    form: str,
    nu_train: npt.NDArray[np.float64],
    sigma_star_train: npt.NDArray[np.float64],
) -> float:
    """Fit c in σ* = c · feat(ν) via least-squares (no intercept)."""
    feat_fn = _FEATURES[form]
    feats = np.asarray([feat_fn(float(n)) for n in nu_train], dtype=np.float64)
    # c = (X'X)^{-1} X'y  with X = feats[:,None], y = sigma_star
    denom = float(np.dot(feats, feats))
    if denom < 1e-14:
        return 0.0
    return float(np.dot(feats, sigma_star_train) / denom)


def _fit_affine_params(
    nu_train: npt.NDArray[np.float64],
    sigma_star_train: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Fit (a, c) in σ* = a + c·ν via OLS with intercept.

    Returns
    -------
    (a, c) : intercept and slope.  Falls back to (mean, 0.0) when the
             design matrix is rank-deficient (e.g. only one training point).
    """
    n = len(nu_train)
    if n < 2:  # not enough points for two parameters — use mean as constant
        return float(np.mean(sigma_star_train)), 0.0
    # design matrix: [1 | ν]
    design = np.column_stack([np.ones(n, dtype=np.float64), nu_train])
    # normal equations: (D'D) θ = D'y
    gram = design.T @ design
    rhs = design.T @ sigma_star_train
    try:
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return float(np.mean(sigma_star_train)), 0.0
    return float(theta[0]), float(theta[1])


def fit_linear_response(
    nu_train: npt.NDArray[np.float64],
    sigma_star_train: npt.NDArray[np.float64],
    nu_test: npt.NDArray[np.float64] | None = None,
    sigma_star_test: npt.NDArray[np.float64] | None = None,
) -> FittedResponseMap:
    """Fit the linear response map σ* = c·ν.

    Parameters
    ----------
    nu_train, sigma_star_train : Training data.
    nu_test, sigma_star_test   : Optional held-out evaluation data.
                                 If None, test R² equals train R².
    """
    return fit_response_map(
        "linear", nu_train, sigma_star_train, nu_test, sigma_star_test
    )


def fit_response_map(
    form: FormType,
    nu_train: npt.NDArray[np.float64],
    sigma_star_train: npt.NDArray[np.float64],
    nu_test: npt.NDArray[np.float64] | None = None,
    sigma_star_test: npt.NDArray[np.float64] | None = None,
) -> FittedResponseMap:
    """Fit a response map of the given functional form.

    Parameters
    ----------
    form                       : "linear" | "sqrt" | "log" | "affine"
    nu_train, sigma_star_train : Training data (arrays of equal length ≥ 1).
    nu_test, sigma_star_test   : Optional held-out data; required together.

    Returns
    -------
    FittedResponseMap  — callable as m(ν) → σ*.
    """
    valid_forms = set(_FEATURES) | {"affine"}
    if form not in valid_forms:
        raise ValueError(f"Unknown form {form!r}; choose from {sorted(valid_forms)}")

    nu_tr = np.asarray(nu_train, dtype=np.float64)
    ss_tr = np.asarray(sigma_star_train, dtype=np.float64)
    if nu_tr.shape != ss_tr.shape or nu_tr.ndim != 1 or len(nu_tr) == 0:
        raise ValueError(
            "nu_train and sigma_star_train must be 1-D arrays of equal non-zero length."
        )

    # ── Fit ──────────────────────────────────────────────────────────────────
    if form == "affine":
        a, c = _fit_affine_params(nu_tr, ss_tr)
        params: tuple[float, ...] = (a, c)
        pred_train = np.asarray([a + c * float(n) for n in nu_tr], dtype=np.float64)
    else:
        c_val = _fit_single_param(form, nu_tr, ss_tr)
        params = (float(c_val),)
        feat_fn = _FEATURES[form]
        pred_train = np.asarray([c_val * feat_fn(float(n)) for n in nu_tr], dtype=np.float64)

    r2_tr = _r2(ss_tr, pred_train)

    # ── Evaluate on test set ─────────────────────────────────────────────────
    if nu_test is not None and sigma_star_test is not None:
        nu_te = np.asarray(nu_test, dtype=np.float64)
        ss_te = np.asarray(sigma_star_test, dtype=np.float64)
        if form == "affine":
            a2, c2 = params
            pred_test = np.asarray([a2 + c2 * float(n) for n in nu_te], dtype=np.float64)
        else:
            feat_fn2 = _FEATURES[form]
            pred_test = np.asarray(
                [params[0] * feat_fn2(float(n)) for n in nu_te], dtype=np.float64
            )
        r2_te = _r2(ss_te, pred_test)
        test_nus_tuple = tuple(float(n) for n in nu_te)
    else:
        r2_te = r2_tr
        test_nus_tuple = ()

    return FittedResponseMap(
        form=form,
        params=params,
        r2_train=r2_tr,
        r2_test=r2_te,
        train_nus=tuple(float(n) for n in nu_tr),
        test_nus=test_nus_tuple,
    )


def best_fit_response_map(
    nu_train: npt.NDArray[np.float64],
    sigma_star_train: npt.NDArray[np.float64],
    nu_test: npt.NDArray[np.float64] | None = None,
    sigma_star_test: npt.NDArray[np.float64] | None = None,
) -> FittedResponseMap:
    """Fit all forms and return the one with highest test R².

    Forms tried: linear, sqrt, log (through-origin) and affine (a + c·ν).
    Falls back to train R² when no test data is provided.
    """
    forms: list[FormType] = ["linear", "sqrt", "log", "affine"]
    candidates = [
        fit_response_map(f, nu_train, sigma_star_train, nu_test, sigma_star_test)
        for f in forms
    ]
    # Prefer test R²; fall back to train R²
    return max(candidates, key=lambda m: m.r2_test)


# ── Alternative response-map models ──────────────────────────────────────────


@dataclass
class AnalyticResponseMap:
    """Analytic reference target m̂(ν) = c·ν from Lemma 1's adversarial limit.

    This map is *not* fitted to empirical σ*_cert data.  It serves as the
    theoretical reference for the CPB projection step.  The safety guarantee
    of CPB derives from projection onto B_{H,α}(ν), not from m̂ accuracy —
    so a moderate mismatch between m̂ and the true response curve does not
    break the budget constraint (see Fig. B/C in the paper).

    The slope ``c`` is either set analytically (default 1.0 — Lemma 1's
    adversarial limit) or calibrated from held-out conditions via
    :meth:`calibrate`.  This is the environment-specific instantiation of
    the "c" in Lemma 1's σ* = c·ν + O(ν²) expansion.

    Attributes
    ----------
    c : Slope coefficient (default 1.0).
    """

    c: float = 1.0

    def __call__(self, nu: float) -> float:
        """Evaluate m̂(ν) = c·ν."""
        return float(self.c * nu)

    @classmethod
    def calibrate(
        cls,
        cal_summary: pd.DataFrame,
        cal_bandwidth_raw: pd.DataFrame,
        c_grid: npt.NDArray[np.float64] | None = None,
    ) -> AnalyticResponseMap:
        """Find c minimising mean normalised regret on calibration conditions.

        The objective is::

            c_cal = argmin_c  mean_{ν ∈ cal} NormRegret(c, ν)

        where::

            NormRegret(c, ν) =
                [V*_cert(ν) − V(clip(c·ν, σ_min, σ_max))]
                / max(V*_cert(ν) − V(σ=0), 1e-6)

        Parameters
        ----------
        cal_summary        : DataFrame with columns ``nu``, ``sigma_min``,
                             ``sigma_max``, ``value_at_star`` (output of
                             ``plot_bandwidth.py``).
        cal_bandwidth_raw  : Raw bandwidth CSV (multiple seeds per σ); must
                             have columns ``nu``, ``sigma``, ``mean_return``.
        c_grid             : 1-D array of candidate c values to evaluate.
                             Defaults to ``np.linspace(0.0, 10.0, 201)``.

        Returns
        -------
        AnalyticResponseMap
            New instance with ``c`` set to the calibrated value.
        """
        if c_grid is None:
            c_grid = np.linspace(0.0, 10.0, 201)

        # Pre-compute per-ν return curves (averaged over seeds)
        return_curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for nu_val, grp in cal_bandwidth_raw.groupby("nu"):
            agg = (
                grp.groupby("sigma")["mean_return"]
                .mean()
                .reset_index()
                .sort_values("sigma")
            )
            return_curves[float(nu_val)] = (
                agg["sigma"].to_numpy(dtype=np.float64),
                agg["mean_return"].to_numpy(dtype=np.float64),
            )

        best_c: float = 1.0
        best_loss: float = float("inf")

        for c_val in c_grid:
            losses: list[float] = []
            for _, row in cal_summary.iterrows():
                nu = float(row["nu"])
                sigma_min = float(row["sigma_min"])
                sigma_max = float(row["sigma_max"])
                v_cert = float(row["value_at_star"])

                if nu not in return_curves:
                    continue
                sigs, vals = return_curves[nu]

                sigma_cpb = float(np.clip(c_val * nu, sigma_min, sigma_max))
                v_cpb = float(np.interp(sigma_cpb, sigs, vals))
                v_base = float(np.interp(0.0, sigs, vals))
                denom = max(v_cert - v_base, 1e-6)
                losses.append((v_cert - v_cpb) / denom)

            if not losses:
                continue
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_loss = loss
                best_c = float(c_val)

        return cls(c=best_c)


@dataclass
class OracleResponseMap:
    """Oracle response map m̂(ν) = σ*_cert(ν) — perfect knowledge baseline.

    Interpolates certified optimal stochasticity σ*_cert(ν) directly from a
    lookup table.  Represents an *upper bound* on what any learned predictor
    can achieve; used for ablation studies comparing CPB performance against
    the theoretical ceiling.

    Attributes
    ----------
    nu_table         : ν grid points (strictly increasing).
    sigma_star_table : Corresponding σ*_cert values.
    """

    nu_table: tuple[float, ...]
    sigma_star_table: tuple[float, ...]

    def __call__(self, nu: float) -> float:
        """Evaluate m̂(ν) via linear interpolation of the lookup table."""
        return float(
            np.interp(
                nu,
                np.asarray(self.nu_table, dtype=np.float64),
                np.asarray(self.sigma_star_table, dtype=np.float64),
            )
        )


class LearnedConditionalMap:
    """Stub for a context-conditional learned response map m̂(ν | context).

    The full implementation would train a predictor (e.g. a neural network)
    mapping game-context features together with ν to σ*_cert.  This stub
    exists to reserve the interface; the logic is intentionally left
    unimplemented.

    Raises
    ------
    NotImplementedError
        Always — until the learned predictor is plugged in.
    """

    def __call__(self, nu: float) -> float:  # noqa: ARG002
        raise NotImplementedError(
            "LearnedConditionalMap is a stub. "
            "Implement a context-conditional predictor before use."
        )
