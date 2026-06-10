"""Exp 1 Phase C: 2D Certified Imagination Region (CIR) heatmap.

For a fixed ν the CIR is A_α = {(H, σ) : R_H(σ; ν) ≤ α}.

This script reads the multi-horizon sweep CSV produced by run_bandwidth.py
(with --horizons H1 H2 ...) and plots:
  - A heatmap of R_H(σ; ν) over the (H, σ) grid.
  - A contour line at R = α delineating the admissible region boundary.
  - Horizontal markers for σ_min(H) and σ_max(H) at each H.

Usage:
    python -m experiments.exp1_noisy_games.plot_cir \
        --csv results/runs/exp1_noisy_games/bandwidth_cir.csv \
        --nu 0.1 \
        --alpha 0.1 \
        --out results/figures/exp1_cir_nu0.1.pdf

    # Overlay multiple ν values:
    python -m experiments.exp1_noisy_games.plot_cir \
        --csv results/runs/exp1_noisy_games/bandwidth_cir.csv \
        --nu 0.05 0.1 0.2 \
        --alpha 0.1 \
        --out results/figures/exp1_cir_multi.pdf
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nms.core.envelope import estimate_bandwidth


def _aggregate_one(df: pd.DataFrame, nu: float, horizon: int) -> pd.DataFrame:
    """Return (sigma, violation_rate) table for a fixed (nu, horizon)."""
    sub = df[(np.abs(df["nu"] - nu) < 1e-9) & (df["horizon"] == horizon)]
    if sub.empty:
        return pd.DataFrame(columns=["sigma", "violation_rate"])
    return (
        sub.groupby("sigma")
        .agg(violation_rate=("violation_rate", "mean"))
        .reset_index()
        .sort_values("sigma")
    )


def plot_cir_heatmap(
    df: pd.DataFrame,
    nu_fixed: float,
    alpha: float,
    out_path: Path,
) -> None:
    """Plot a single-ν CIR heatmap over (H, σ).

    Parameters
    ----------
    df        : Multi-horizon sweep DataFrame (columns: nu, horizon, sigma,
                violation_rate, seed, ...).
    nu_fixed  : The ν value to slice on.
    alpha     : Violation rate threshold α.
    out_path  : Path to save the PDF/PNG figure.
    """
    horizons = sorted(df["horizon"].unique())
    sigmas_all = sorted(df["sigma"].unique())

    # Build the (H, σ) matrix
    h_n = len(horizons)
    s_n = len(sigmas_all)
    rate_matrix = np.full((h_n, s_n), np.nan)
    sigma_min_vec = np.full(h_n, np.nan)
    sigma_max_vec = np.full(h_n, np.nan)

    sigma_arr = np.asarray(sigmas_all, dtype=float)

    for i, h in enumerate(horizons):
        sub = _aggregate_one(df, nu_fixed, h)
        if sub.empty:
            continue
        # Map sigma → column index
        for _, row in sub.iterrows():
            j_candidates = np.where(np.abs(sigma_arr - float(row["sigma"])) < 1e-9)[0]
            if len(j_candidates) == 0:
                continue
            j = int(j_candidates[0])
            rate_matrix[i, j] = float(row["violation_rate"])

        # Compute bandwidth for this (ν, H)
        risk_arr = sub["violation_rate"].to_numpy()
        s_arr = sub["sigma"].to_numpy()
        lookup = dict(zip(s_arr.tolist(), risk_arr.tolist(), strict=False))

        def risk_fn(s: float, lk: dict[float, float] = lookup) -> float:
            return lk[min(lk, key=lambda k: abs(k - s))]

        bw = estimate_bandwidth(s_arr, risk_fn, nu=nu_fixed, horizon=h, alpha=alpha)
        if not bw.is_empty():
            sigma_min_vec[i] = bw.sigma_min
            sigma_max_vec[i] = bw.sigma_max

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    sigma_mesh = np.asarray(sigmas_all, dtype=float)
    horizon_mesh = np.asarray(horizons, dtype=float)

    im = ax.pcolormesh(
        sigma_mesh, horizon_mesh, rate_matrix,
        cmap="RdYlGn_r", vmin=0.0, vmax=1.0, shading="nearest",
    )
    fig.colorbar(im, ax=ax, label=r"$R_H(\sigma;\nu)$")

    # α-contour line (boundary of admissible region)
    if not np.all(np.isnan(rate_matrix)):
        try:
            ax.contour(
                sigma_mesh, horizon_mesh, rate_matrix,
                levels=[alpha],
                colors=["white"],
                linewidths=2,
                linestyles="--",
            )
        except Exception:  # noqa: BLE001
            pass  # contour may fail if matrix has too many nans

    # σ_min / σ_max curves
    valid = ~np.isnan(sigma_min_vec)
    if valid.any():
        ax.plot(sigma_min_vec[valid], horizon_mesh[valid],
                "w-o", ms=4, lw=1.5, label=r"$\sigma_{\min}(H)$")
    valid_max = ~np.isnan(sigma_max_vec)
    if valid_max.any():
        ax.plot(sigma_max_vec[valid_max], horizon_mesh[valid_max],
                "ws", ms=4, lw=1.5, linestyle="--",
                label=r"$\sigma_{\max}(H)$")

    ax.set_xlabel("stochasticity σ")
    ax.set_ylabel("horizon H")
    ax.set_title(
        rf"Certified Imagination Region $A_{{\alpha}}$  ·  ν={nu_fixed:.3f}, α={alpha}"
    )
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved CIR heatmap → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 1 CIR heatmap(s).")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument(
        "--nu", nargs="+", type=float, required=True,
        metavar="NU",
        help="ν value(s) to plot; one figure per ν.",
    )
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--out", type=Path, required=True,
        help=(
            "Output path for the first ν.  Additional ν values are saved as "
            "<stem>_nu<N><suffix>."
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "horizon" not in df.columns:
        warnings.warn(
            "CSV does not contain a 'horizon' column. "
            "Did you run run_bandwidth.py with --horizons H1 H2 ...?",
            stacklevel=2,
        )
        return

    for i, nu in enumerate(args.nu):
        if i == 0:
            out_path = args.out
        else:
            out_path = args.out.with_stem(f"{args.out.stem}_nu{nu:.3f}".replace(".", "p"))
        plot_cir_heatmap(df, nu_fixed=nu, alpha=args.alpha, out_path=out_path)


if __name__ == "__main__":
    main()
