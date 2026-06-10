"""Exp 1: generate Fig 1 (certified bandwidth hero) and Fig 2 (theory panels).

Fig 1 — R_H(σ; ν) curves for each ν, with α threshold, bandwidth shading,
         certified optimal σ*, and NMS-projected σ^NMS marked.

Fig 2 — Up to four theory-validation panels:
  (a) σ_min(ν) and σ_max(ν) vs ν  — Necessary Stochasticity (Theorem 1).
  (b) Empirical σ*(ν) vs fitted m(ν) — Lemma 1 (Phase B train/test split).
  (c) V(σ^NMS) vs V(σ*)            — Theorem 2: projection accuracy.
      Skipped automatically when all bandwidths are right-censored
      (use --skip-projection-panel to force-skip).
  (d) Certified vs unconstrained σ* — gap analysis.

Also writes exp1_summary.csv alongside the output figures.

Usage — Phase B (mixture data):
    python -m experiments.exp1_noisy_games.plot_bandwidth \
        --csv results/runs/exp1b_full_bandwidth/bandwidth.csv \
        --out-fig1 results/figures/exp1b_bandwidth_hero.pdf \
        --out-fig2 results/figures/exp1b_lemma1_proj.pdf \
        --horizon 200 --alpha 0.1 \
        --train-nus 0.05 0.15 0.30 \
        --test-nus 0.10 0.20 \
        --response-form best

Usage — self-play baseline (right-censored; skip projection panel):
    python -m experiments.exp1_noisy_games.plot_bandwidth \
        --csv results/runs/exp1a_clean_lower_bound/bandwidth.csv \
        --out-fig1 results/figures/exp1a_bandwidth.pdf \
        --out-fig2 results/figures/exp1a_lemma1.pdf \
        --skip-projection-panel
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nms.core.envelope import estimate_bandwidth
from nms.core.projection import BandwidthEmptyError, project_onto_bandwidth
from nms.core.response_map import (
    AnalyticResponseMap,
    OracleResponseMap,
    best_fit_response_map,
    fit_response_map,
)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average violation_rate and mean_return over seeds for each (nu, sigma)."""
    agg_cols: dict[str, object] = {
        "violation_rate": ("violation_rate", "mean"),
        "violation_se": ("violation_rate", "sem"),
        "mean_return": ("mean_return", "mean"),
        "return_se": ("mean_return", "sem"),
    }
    # exploitation_rate is optional (only present in Phase A CSVs)
    if "exploitation_rate" in df.columns:
        agg_cols["exploitation_rate"] = ("exploitation_rate", "mean")

    return (
        df.groupby(["nu", "sigma"])
        .agg(**agg_cols)  # type: ignore[arg-type]
        .reset_index()
    )


def bandwidth_from_curve(
    sigma_arr: np.ndarray,
    risk_arr: np.ndarray,
    nu: float,
    horizon: int,
    alpha: float,
):
    """Construct a Bandwidth from a discrete (sigma, risk) table."""
    lookup = {float(s): float(r) for s, r in zip(sigma_arr, risk_arr, strict=False)}

    def risk_fn(s: float) -> float:
        return lookup[min(lookup, key=lambda k: abs(k - s))]

    return estimate_bandwidth(sigma_arr, risk_fn, nu=nu, horizon=horizon, alpha=alpha)


def certified_optimum(
    sigmas: np.ndarray,
    risks: np.ndarray,
    returns: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Return (σ*_certified, V*_certified): best return within admissible set.

    σ*_certified = argmax V(σ) s.t. R_H(σ) ≤ α

    Falls back to (nan, nan) if no admissible σ exists.
    """
    admissible = risks <= alpha
    if not admissible.any():
        return float("nan"), float("nan")
    adm_returns = np.where(admissible, returns, -np.inf)
    best_idx = int(np.argmax(adm_returns))
    return float(sigmas[best_idx]), float(returns[best_idx])


def _try_fit_response_map(
    summary: pd.DataFrame,
    train_nus: list[float],
    test_nus: list[float],
    form: str,
) -> object:
    """Fit response map; return FittedResponseMap or None with fallback warnings.

    Falls back to None when:
    - No train ν values in summary
    - All sigma_star_certified are NaN (empty bandwidths) or grid-censored
    """
    train_set = set(train_nus)
    test_set = set(test_nus)

    train_df = summary[
        summary["nu"].apply(lambda n: any(abs(n - t) < 1e-9 for t in train_set))
    ].dropna(subset=["sigma_star_certified"])

    # Check if all certified optima are grid-censored (hit sigma_max)
    all_censored = (
        train_df["right_censored"].all()
        if "right_censored" in train_df.columns and len(train_df) > 0
        else False
    )

    if len(train_df) == 0:
        warnings.warn(
            "No train ν values found in summary — skipping Phase B response map fitting. "
            "Check --train-nus values match the ν values in the CSV.",
            stacklevel=3,
        )
        return None

    if all_censored:
        warnings.warn(
            "All sigma_star_certified values are grid-censored (right_censored=True). "
            "Phase B response map fit may be unreliable — using c=1.0 as placeholder. "
            "Run with --skip-projection-panel to suppress panel (c).",
            stacklevel=3,
        )

    test_df = summary[
        summary["nu"].apply(lambda n: any(abs(n - t) < 1e-9 for t in test_set))
    ].dropna(subset=["sigma_star_certified"])

    nu_tr = train_df["nu"].to_numpy()
    ss_tr = train_df["sigma_star_certified"].to_numpy()
    nu_te = test_df["nu"].to_numpy() if len(test_df) > 0 else None
    ss_te = test_df["sigma_star_certified"].to_numpy() if len(test_df) > 0 else None

    if form == "best":
        return best_fit_response_map(nu_tr, ss_tr, nu_te, ss_te)  # type: ignore[arg-type]
    return fit_response_map(form, nu_tr, ss_tr, nu_te, ss_te)  # type: ignore[arg-type]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 1 bandwidth figures.")
    ap.add_argument(
        "--csv", type=Path,
        default=Path("results/runs/exp1b_full_bandwidth/bandwidth.csv"),
        help="Input bandwidth CSV (default: exp1b mixture data).",
    )
    ap.add_argument(
        "--out-fig1", type=Path,
        default=Path("results/figures/exp1b_bandwidth_hero.pdf"),
        help="Output path for Fig 1 (violation-rate hero).",
    )
    ap.add_argument(
        "--out-fig2", type=Path,
        default=Path("results/figures/exp1b_lemma1_proj.pdf"),
        help="Output path for Fig 2 (theory panels).",
    )
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    # Phase B: response map fitting
    ap.add_argument(
        "--train-nus", nargs="+", type=float,
        default=[0.05, 0.15, 0.30],
        metavar="NU",
        help="ν values used for fitting the response map (Phase B train set).",
    )
    ap.add_argument(
        "--test-nus", nargs="+", type=float,
        default=[0.10, 0.20],
        metavar="NU",
        help="ν values used for evaluating the response map (Phase B test set).",
    )
    ap.add_argument(
        "--response-form",
        default="best",
        choices=["linear", "sqrt", "log", "affine", "best"],
        help=(
            "Functional form for the fitted response map m(ν).  "
            "'best' (default) auto-selects the form with highest test R²."
        ),
    )
    ap.add_argument(
        "--response-model",
        default="fitted",
        choices=["fitted", "analytic", "oracle"],
        help=(
            "Which response-map model to use for CPB projection.  "
            "'fitted' (default): empirically fitted via --response-form.  "
            "'analytic': AnalyticResponseMap m̂(ν)=ν (Lemma 1 reference).  "
            "'oracle': OracleResponseMap interpolating empirical σ*_cert."
        ),
    )
    ap.add_argument(
        "--c-cal-json",
        type=Path,
        default=Path("results/runs/exp1b_full_bandwidth/c_cal.json"),
        help="Path to c_cal.json (output of calibrate_c.py). "
             "If the file exists, c_cal columns are added to exp1_summary.csv.",
    )
    # Panel control
    ap.add_argument(
        "--skip-projection-panel",
        action="store_true",
        help=(
            "Skip Fig 2 panel (c) — Theorem 2 projection accuracy. "
            "Use when sigma_star_certified is grid-censored "
            "(e.g., clean self-play where σ_max = grid_max for all ν)."
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Filter to target horizon (relevant when CSV contains multiple horizons)
    if "horizon" in df.columns:
        df = df[df["horizon"] == args.horizon]
    agg = aggregate(df)
    nus = sorted(agg["nu"].unique())

    # ------------------------------------------------------------------ Fig 1
    fig1, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(nus)))
    summary_rows = []
    # Store return/risk curves for post-loop interpolation
    nu_return_curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    nu_risk_curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for color, nu in zip(colors, nus, strict=False):
        sub = agg[agg["nu"] == nu].sort_values("sigma")
        sigmas = sub["sigma"].to_numpy()
        risks = sub["violation_rate"].to_numpy()
        returns = sub["mean_return"].to_numpy()
        nu_return_curves[float(nu)] = (sigmas, returns)
        nu_risk_curves[float(nu)] = (sigmas, risks)

        ax.plot(sigmas, risks, color=color, lw=2, label=f"ν={nu:.2f}")

        bw = bandwidth_from_curve(sigmas, risks, nu, args.horizon, args.alpha)

        # Certified optimum σ*
        sigma_star_cert, value_star_cert = certified_optimum(
            sigmas, risks, returns, args.alpha
        )
        # Unconstrained optimum (ignores bandwidth)
        sigma_star_unc = float(sigmas[np.argmax(returns)])
        value_star_unc = float(returns.max())

        # NMS projection: use σ*_cert as representative m(ν) proxy
        try:
            sigma_nms = project_onto_bandwidth(sigma_star_cert, bw) \
                if np.isfinite(sigma_star_cert) else float("nan")
        except BandwidthEmptyError:
            sigma_nms = float("nan")

        value_at_nms = (
            float(np.interp(sigma_nms, sigmas, returns))
            if np.isfinite(sigma_nms)
            else float("nan")
        )

        if not bw.is_empty():
            ax.axvspan(bw.sigma_min, bw.sigma_max, color=color, alpha=0.07)
            ax.axvline(sigma_nms, color=color, lw=0.8, ls=":", alpha=0.8)
            if np.isfinite(sigma_star_cert):
                ax.axvline(sigma_star_cert, color=color, lw=1.2, ls="--", alpha=0.6)

        summary_rows.append({
            "nu": nu,
            "sigma_min": bw.sigma_min,
            "sigma_max": bw.sigma_max,
            "is_quasiconvex": bw.is_quasiconvex,
            "n_components": bw.n_components,
            "left_censored": bw.left_censored,
            "right_censored": bw.right_censored,
            "sigma_star_certified": sigma_star_cert,
            "sigma_star_unconstrained": sigma_star_unc,
            "sigma_nms": sigma_nms,
            "value_at_nms": value_at_nms,
            "value_at_star": value_star_cert,
            "value_at_star_unc": value_star_unc,
            "gap_to_certified": float(
                abs(sigma_nms - sigma_star_cert)
                if np.isfinite(sigma_nms) and np.isfinite(sigma_star_cert)
                else float("nan")
            ),
            "gap_to_unconstrained": float(
                abs(sigma_nms - sigma_star_unc)
                if np.isfinite(sigma_nms)
                else float("nan")
            ),
        })

    ax.axhline(args.alpha, color="black", ls="--", lw=1, label=f"α={args.alpha}")
    ax.set_xlabel("internal stochasticity σ  (forgiveness probability)")
    ax.set_ylabel(r"$R_H(\sigma;\,\nu)$ — violation rate")
    ax.set_title("Certified Stochasticity Bandwidth  ·  Noisy IPD")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(-0.02, 1.05)
    fig1.tight_layout()
    args.out_fig1.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.out_fig1, dpi=200)
    plt.close(fig1)
    print(f"Saved Fig 1 → {args.out_fig1}")

    # ------------------------------------------------------------------ Fig 2
    summary = pd.DataFrame(summary_rows)

    # Decide whether panel (c) is meaningful
    all_rc = bool(summary["right_censored"].all())
    skip_proj = args.skip_projection_panel or all_rc
    if all_rc and not args.skip_projection_panel:
        print(
            "WARNING: right_censored=True on all ν — "
            "skipping Theorem 2 projection accuracy panel (c). "
            "Pass --skip-projection-panel to suppress this message."
        )

    # Phase B train/test split flags
    train_set = set(args.train_nus)
    test_set = set(args.test_nus)
    summary["is_train_nu"] = summary["nu"].apply(
        lambda n: any(abs(n - t) < 1e-9 for t in train_set)
    )

    fitted_map = _try_fit_response_map(
        summary, args.train_nus, args.test_nus, args.response_form
    )

    # Override with analytic or oracle model when requested
    if args.response_model == "analytic":
        fitted_map = AnalyticResponseMap(c=1.0)
        print("Using AnalyticResponseMap: m̂(ν) = ν  (Lemma 1 reference target)")
    elif args.response_model == "oracle":
        cert_df = summary.dropna(subset=["sigma_star_certified"])
        if len(cert_df) > 0:
            fitted_map = OracleResponseMap(
                nu_table=tuple(float(n) for n in cert_df["nu"]),
                sigma_star_table=tuple(
                    float(s) for s in cert_df["sigma_star_certified"]
                ),
            )
            print("Using OracleResponseMap: m̂(ν) = σ*_cert(ν)  (perfect-knowledge baseline)")
        else:
            print("WARNING: oracle model requested but no σ*_cert data — falling back to fitted.")
    # "fitted" (default): keep fitted_map as-is

    if fitted_map is not None:
        summary["m_nu"] = summary["nu"].apply(fitted_map)  # type: ignore[arg-type]
        summary["response_map_r2"] = fitted_map.r2_test  # type: ignore[union-attr]

        # Recompute sigma_nms using the fitted map
        def _reproject(row: pd.Series) -> float:  # type: ignore[type-arg]
            if np.isnan(row["sigma_min"]) or row["sigma_min"] > row["sigma_max"]:
                return float("nan")
            m_val = float(fitted_map(float(row["nu"])))  # type: ignore[misc]
            lo, hi = float(row["sigma_min"]), float(row["sigma_max"])
            return float(np.clip(m_val, lo, hi))

        summary["sigma_nms_fitted"] = summary.apply(_reproject, axis=1)

        # Compute V(σ^NMS_fitted) using stored return curves
        def _v_at_fitted(row: pd.Series) -> float:  # type: ignore[type-arg]
            s = float(row["sigma_nms_fitted"])
            if not np.isfinite(s):
                return float("nan")
            nu_key = float(row["nu"])
            if nu_key not in nu_return_curves:
                return float("nan")
            sg, ret = nu_return_curves[nu_key]
            return float(np.interp(s, sg, ret))

        summary["value_at_nms_fitted"] = summary.apply(_v_at_fitted, axis=1)
    else:
        summary["m_nu"] = float("nan")
        summary["response_map_r2"] = float("nan")
        summary["sigma_nms_fitted"] = float("nan")
        summary["value_at_nms_fitted"] = float("nan")

    # ── Task D: robustness columns (CPB with naive linear map m̂(ν) = ν) ─────
    # sigma_cpb_naive = project(ν, B(ν)) = clip(ν, σ_min, σ_max)
    def _sigma_cpb_naive(row: pd.Series) -> float:  # type: ignore[type-arg]
        lo, hi = float(row["sigma_min"]), float(row["sigma_max"])
        if not np.isfinite(lo) or lo > hi:
            return float("nan")
        return float(np.clip(float(row["nu"]), lo, hi))

    summary["m_reference_naive"] = summary["nu"].astype(float)
    summary["sigma_cpb_naive"] = summary.apply(_sigma_cpb_naive, axis=1)

    def _interp_nu(curves: dict[float, tuple[np.ndarray, np.ndarray]],
                   row: pd.Series, sigma_col: str) -> float:  # type: ignore[type-arg]
        s = float(row[sigma_col])
        if not np.isfinite(s):
            return float("nan")
        nu_key = float(row["nu"])
        if nu_key not in curves:
            return float("nan")
        sg, vals = curves[nu_key]
        return float(np.interp(s, sg, vals))

    summary["value_cpb_naive"] = summary.apply(
        lambda r: _interp_nu(nu_return_curves, r, "sigma_cpb_naive"), axis=1
    )
    summary["violation_cpb_naive"] = summary.apply(
        lambda r: _interp_nu(nu_risk_curves, r, "sigma_cpb_naive"), axis=1
    )

    # V(σ=0) as baseline for normalized regret
    def _v_at_sigma0(nu_key: float) -> float:
        if nu_key not in nu_return_curves:
            return float("nan")
        sg, ret = nu_return_curves[nu_key]
        return float(np.interp(0.0, sg, ret))

    summary["value_at_sigma0"] = summary["nu"].apply(_v_at_sigma0)
    summary["regret_to_certified_naive"] = (
        summary["value_at_star"] - summary["value_cpb_naive"]
    )
    denom_regret = (summary["value_at_star"] - summary["value_at_sigma0"]).clip(lower=1e-6)
    summary["normalized_regret_naive"] = summary["regret_to_certified_naive"] / denom_regret
    summary["is_safe_cpb_naive"] = summary["violation_cpb_naive"] <= args.alpha

    # ── c_cal columns (loaded from JSON if available) ─────────────────────────
    c_cal_value: float | None = None
    if hasattr(args, "c_cal_json") and Path(args.c_cal_json).exists():
        with open(args.c_cal_json) as _fp:
            c_cal_value = float(json.load(_fp)["c_cal"])
        print(f"Loaded c_cal = {c_cal_value:.3f} for summary CSV columns")

    if c_cal_value is not None:

        def _sigma_cpb_ccal(row: pd.Series) -> float:  # type: ignore[type-arg]
            lo, hi = float(row["sigma_min"]), float(row["sigma_max"])
            if not np.isfinite(lo) or lo > hi:
                return float("nan")
            return float(np.clip(c_cal_value * float(row["nu"]), lo, hi))  # type: ignore[operator]

        summary["c_cal"] = c_cal_value
        summary["sigma_cpb_with_c_cal"] = summary.apply(_sigma_cpb_ccal, axis=1)
        summary["value_cpb_with_c_cal"] = summary.apply(
            lambda r: _interp_nu(nu_return_curves, r, "sigma_cpb_with_c_cal"), axis=1
        )
        summary["violation_cpb_with_c_cal"] = summary.apply(
            lambda r: _interp_nu(nu_risk_curves, r, "sigma_cpb_with_c_cal"), axis=1
        )
        summary["regret_with_c_cal"] = (
            summary["value_at_star"] - summary["value_cpb_with_c_cal"]
        )
        denom_ccal = (summary["value_at_star"] - summary["value_at_sigma0"]).clip(lower=1e-6)
        summary["normalized_regret_with_c_cal"] = (
            summary["regret_with_c_cal"] / denom_ccal
        )
        summary["is_safe_cpb_with_c_cal"] = (
            summary["violation_cpb_with_c_cal"] <= args.alpha
        )
    else:
        for col in [
            "c_cal", "sigma_cpb_with_c_cal", "value_cpb_with_c_cal",
            "violation_cpb_with_c_cal", "regret_with_c_cal",
            "normalized_regret_with_c_cal", "is_safe_cpb_with_c_cal",
        ]:
            summary[col] = float("nan")

    # Number of panels: 4 normally, 3 when projection is skipped
    n_panels = 3 if skip_proj else 4
    fig2, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]  # type: ignore[assignment]

    # Panel (a): Necessary Stochasticity
    ax0 = axes[0]
    valid_bw = summary[~(summary["sigma_min"] > summary["sigma_max"])]
    if len(valid_bw) > 0:
        ax0.plot(valid_bw["nu"], valid_bw["sigma_min"], "o-",
                 label=r"$\sigma_{\min}(\nu)$")
        ax0.plot(valid_bw["nu"], valid_bw["sigma_max"], "s-",
                 label=r"$\sigma_{\max}(\nu)$")
        ax0.fill_between(valid_bw["nu"], valid_bw["sigma_min"], valid_bw["sigma_max"],
                         alpha=0.15, label="bandwidth")
    ax0.set_xlabel("noise rate ν")
    ax0.set_ylabel("σ")
    ax0.set_title("Two-sided certified bandwidth\n(spiral + exploit pressure)")
    ax0.legend(fontsize=8)

    # Panel (b): Phase B — fitted response map
    ax1 = axes[1]
    cert_valid = summary.dropna(subset=["sigma_star_certified"])
    if len(cert_valid) > 0:
        ax1.plot(cert_valid["nu"], cert_valid["sigma_star_certified"], "o-",
                 label=r"certified $\sigma^*(\nu)$")
    if fitted_map is not None:
        nu_plot = np.linspace(float(summary["nu"].min()), float(summary["nu"].max()), 200)
        ax1.plot(nu_plot, [fitted_map(float(n)) for n in nu_plot], "k--",  # type: ignore[arg-type]
                 label=r"$\hat{m}(\nu)$ analytic reference")
    # Highlight train / test points
    tr = summary[summary["is_train_nu"]].dropna(subset=["sigma_star_certified"])
    te = summary[
        ~summary["is_train_nu"]
        & summary["nu"].apply(lambda n: any(abs(n - t) < 1e-9 for t in test_set))
    ].dropna(subset=["sigma_star_certified"])
    if len(tr) > 0:
        ax1.scatter(tr["nu"], tr["sigma_star_certified"], marker="D", zorder=5,
                    label="train ν")
    if len(te) > 0:
        ax1.scatter(te["nu"], te["sigma_star_certified"], marker="*", s=80, zorder=5,
                    label="test ν")
    ax1.set_xlabel("noise rate ν")
    ax1.set_ylabel("σ")
    ax1.set_title(
        r"Analytic Reference Target $\hat{m}(\nu) = c_{\mathrm{cal}}\cdot\nu$"
        "\n(Lemma 1; c calibrated on held-out ν)"
    )
    ax1.legend(fontsize=7)

    # Panel (c): Theorem 2 — projection accuracy  [optional]
    # Shows V(σ^NMS_fitted) vs V(σ*_cert) — gap measures cost of using m(ν) proxy.
    if not skip_proj:
        ax2 = axes[2]
        proj_col = "value_at_nms_fitted" if "value_at_nms_fitted" in summary.columns \
            else "value_at_nms"
        proj_valid = summary.dropna(subset=["value_at_star", proj_col])
        if len(proj_valid) > 0:
            ax2.plot(proj_valid["nu"], proj_valid["value_at_star"], "o-",
                     label=r"$V(\sigma^*_{\rm cert})$")
            ax2.plot(proj_valid["nu"], proj_valid[proj_col], "s--",
                     label=r"$V(\sigma^{NMS}_{\rm fitted})$")
        ax2.set_xlabel("noise rate ν")
        ax2.set_ylabel("mean return")
        ax2.set_title(
            r"Projected analytic reference"
            "\n"
            r"($\sigma^{\mathrm{CPB}}$ vs $\sigma^\star_{\mathrm{cert}}$)"
        )
        ax2.legend(fontsize=8)

        # Panel (d): certified vs unconstrained gap
        ax3 = axes[3]
    else:
        # Panel (c) is actually the gap panel when projection is skipped
        ax3 = axes[2]

    cert_valid2 = summary.dropna(subset=["sigma_star_certified", "sigma_star_unconstrained"])
    if len(cert_valid2) > 0:
        ax3.plot(cert_valid2["nu"], cert_valid2["sigma_star_certified"], "o-",
                 label=r"$\sigma^*_{\rm cert}$")
        ax3.plot(cert_valid2["nu"], cert_valid2["sigma_star_unconstrained"], "s--",
                 label=r"$\sigma^*_{\rm unc}$")
        ax3.fill_between(
            cert_valid2["nu"],
            cert_valid2["sigma_star_certified"],
            cert_valid2["sigma_star_unconstrained"],
            alpha=0.15, label="gap",
        )
    ax3.set_xlabel("noise rate ν")
    ax3.set_ylabel("σ")
    ax3.set_title("Cert. vs unconstrained optimum")
    ax3.legend(fontsize=8)

    # Report quasi-convexity violations
    n_nonqc = int((summary["n_components"] > 1).sum())
    if n_nonqc > 0:
        fig2.text(
            0.5, 0.01,
            f"Note: {n_nonqc}/{len(summary)} ν values show non-contiguous "
            "admissible sets (quasi-convexity violated).",
            ha="center", fontsize=8, color="red",
        )

    fig2.tight_layout()
    fig2.savefig(args.out_fig2, dpi=200)
    plt.close(fig2)
    print(f"Saved Fig 2 → {args.out_fig2}")

    # -------------------------------------------------------- summary CSV + print
    summary_path = args.out_fig1.parent / "exp1_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary → {summary_path}")

    # -------------------------------------------------------- milestone printout
    print("\n=== Exp 1 Milestone Summary ===")
    cols = [
        "nu", "sigma_min", "sigma_max",
        "sigma_star_certified", "sigma_star_unconstrained",
        "sigma_nms",
        "value_at_star", "value_at_nms",
        "left_censored", "right_censored",
        "is_quasiconvex", "n_components",
    ]
    available = [c for c in cols if c in summary.columns]
    print(summary[available].to_string(index=False))

    # Right-censored check (Phase A acceptance criterion)
    rc_count = int(summary["right_censored"].sum())
    total_nu = len(summary)
    print(f"\nRight-censored bandwidth count: {rc_count}/{total_nu}")
    if rc_count == total_nu:
        warnings.warn(
            "All bandwidths are right-censored (σ_max == grid_max). "
            "The U-shaped curves have not emerged. "
            "ROOT CAUSE: with spiral_len ≤ 5 and 20% AllD, the mutual_defection_spiral "
            "fires in AllD episodes at ALL σ values (AllD always defects, creating (D,D) runs), "
            "setting a violation floor of allD_fraction > α=0.1 for all σ. "
            "FIX: use --spiral-len 15 --exploit-len 15 so short AllD-induced runs don't count.",
            stacklevel=2,
        )

    # Quasi-convexity violation rate
    qc_rate = float((summary["n_components"] > 1).mean())
    print(f"Quasi-convexity violation rate: {qc_rate:.2%}")

    # Theorem 2: V(σ^NMS_fitted) / V(σ*_cert) gap
    # Uses the fitted response map σ^NMS = project(m(ν), B(ν)) for a meaningful gap.
    if not skip_proj and "value_at_nms_fitted" in summary.columns:
        valid = summary.dropna(subset=["value_at_nms_fitted", "value_at_star"])
        if len(valid) > 0:
            denom = valid["value_at_star"].abs().clip(lower=1e-9)
            gaps = (valid["value_at_star"] - valid["value_at_nms_fitted"]) / denom
            print(f"Theorem 2 — V(σ^NMS_fitted)/V(σ*) relative gap "
                  f"[σ^NMS=project(m(ν),B)]: "
                  f"mean={float(gaps.mean()):.3f}, max={float(gaps.max()):.3f}")

    # Necessary Stochasticity (σ_min > 0 for large ν)
    ns_present = summary[summary["sigma_min"] > 1e-6]
    if len(ns_present) > 0:
        nu0 = float(ns_present["nu"].min())
        print(f"Theorem 1 — Necessary Stochasticity activates at ν ≥ {nu0:.3f}")
    else:
        warnings.warn(
            "sigma_min == 0 for all ν. Necessary Stochasticity not observed. "
            "Try spiral_len=3, horizon=500, or reconsider the violation definition.",
            stacklevel=2,
        )

    # Phase B: response map quality
    if fitted_map is not None:
        fm = fitted_map  # type: ignore[union-attr]
        if fm.form == "affine":
            params_str = f"a={fm.params[0]:.4f}, c={fm.params[1]:.4f}"
        else:
            params_str = f"c={fm.params[0]:.4f}"
        print(
            f"\nPhase B — fitted m(ν) form={fm.form}, {params_str}, "
            f"R²_train={fm.r2_train:.3f}, R²_test={fm.r2_test:.3f}"
        )


if __name__ == "__main__":
    main()
