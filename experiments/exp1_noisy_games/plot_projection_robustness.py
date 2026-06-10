"""Fig B — Projection Robustness: safety preserved under m̂ misspecification.

Two-panel figure:
  (a) V(σ) curves for each ν with markers for σ*_unc, σ*_cert, and σ^CPB_{m̂=c·ν}.
      Shows that projection onto B(ν) guarantees R_H ≤ α regardless of c.
  (b) Normalized regret bar chart: (V* - V^CPB) / (V* - V(σ=0)) for each ν × c.
      Safety (all violations ≤ α) is the primary acceptance criterion.

Usage:
    python -m experiments.exp1_noisy_games.plot_projection_robustness \\
        --summary results/figures/exp1_summary.csv \\
        --bandwidth results/runs/exp1b_full_bandwidth/bandwidth.csv \\
        --out results/figures/exp1b_projection_robustness.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    }
)

_DEFAULT_C_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]

# Colour palette (consistent with other exp1 figures)
_NU_COLORS = {
    0.00: "#1f77b4",
    0.05: "#ff7f0e",
    0.10: "#2ca02c",
    0.15: "#d62728",
    0.20: "#9467bd",
    0.30: "#8c564b",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average violation_rate and mean_return over seeds."""
    return (
        df.groupby(["nu", "sigma"])
        .agg(
            violation_rate=("violation_rate", "mean"),
            mean_return=("mean_return", "mean"),
        )
        .reset_index()
    )


def _load_curves(
    bw_csv: Path, horizon: int
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {ν: (sigmas, returns, violations)} from bandwidth CSV."""
    df = pd.read_csv(bw_csv)
    if "horizon" in df.columns:
        df = df[df["horizon"] == horizon]
    agg = _aggregate(df)
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for nu, grp in agg.groupby("nu"):
        grp = grp.sort_values("sigma")
        curves[float(nu)] = (
            grp["sigma"].to_numpy(),
            grp["mean_return"].to_numpy(),
            grp["violation_rate"].to_numpy(),
        )
    return curves


def _cpb_sigma(nu: float, c: float, sigma_min: float, sigma_max: float) -> float:
    """Compute σ^CPB = clip(c·ν, σ_min, σ_max)."""
    if not np.isfinite(sigma_min) or sigma_min > sigma_max:
        return float("nan")
    return float(np.clip(c * nu, sigma_min, sigma_max))


def _interp(sigmas: np.ndarray, vals: np.ndarray, s: float) -> float:
    if not np.isfinite(s):
        return float("nan")
    return float(np.interp(s, sigmas, vals))


def _normalized_regret(
    v_star: float, v_cpb: float, v_sigma0: float
) -> float:
    denom = max(v_star - v_sigma0, 1e-6)
    return (v_star - v_cpb) / denom


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_projection_robustness(
    summary: pd.DataFrame,
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    c_values: list[float],
    alpha: float,
    out_path: Path,
    c_cal: float | None = None,
) -> None:
    """Generate the two-panel projection-robustness figure."""
    nus = sorted(curves.keys())
    # Ensure c_cal is present in the bar-chart c-values (for comparison)
    all_c = sorted(set(c_values) | ({round(c_cal, 6)} if c_cal is not None else set()))
    c_cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(all_c)))

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # ── Panel (a): V(σ) curves + CPB markers ─────────────────────────────────
    for nu in nus:
        if nu not in curves:
            continue
        sigmas, returns, _ = curves[nu]
        color = _NU_COLORS.get(nu, "gray")
        ax_v.plot(sigmas, returns, color=color, lw=1.5, alpha=0.7,
                  label=rf"$\nu$={nu:.2f}")

        row = summary[np.abs(summary["nu"] - nu) < 1e-9]
        if row.empty:
            continue
        row = row.iloc[0]

        sigma_cert = float(row["sigma_star_certified"])
        sigma_unc = float(row["sigma_star_unconstrained"])
        sigma_min = float(row["sigma_min"])
        sigma_max = float(row["sigma_max"])

        # σ*_cert marker (filled diamond)
        if np.isfinite(sigma_cert):
            v_cert = _interp(sigmas, returns, sigma_cert)
            ax_v.scatter([sigma_cert], [v_cert], color=color,
                         marker="D", s=40, zorder=5)

        # σ*_unc marker (open square)
        if np.isfinite(sigma_unc):
            v_unc = _interp(sigmas, returns, sigma_unc)
            ax_v.scatter([sigma_unc], [v_unc], color=color,
                         marker="s", s=30, zorder=4, facecolors="none",
                         edgecolors=color, linewidths=1.2)

        # σ^CPB for c=1.0 (cross marker — analytic baseline)
        s_cpb1 = _cpb_sigma(nu, 1.0, sigma_min, sigma_max)
        if np.isfinite(s_cpb1):
            v_cpb1 = _interp(sigmas, returns, s_cpb1)
            ax_v.scatter([s_cpb1], [v_cpb1], color=color,
                         marker="x", s=50, zorder=6, linewidths=1.5)

        # σ^CPB for c=c_cal (star marker — calibrated default)
        if c_cal is not None:
            s_cpb_cal = _cpb_sigma(nu, c_cal, sigma_min, sigma_max)
            if np.isfinite(s_cpb_cal):
                v_cpb_cal = _interp(sigmas, returns, s_cpb_cal)
                ax_v.scatter([s_cpb_cal], [v_cpb_cal], color="red",
                             marker="*", s=80, zorder=7, linewidths=1.5)

    # Legend patches
    from matplotlib.lines import Line2D
    c_cal_str = (rf"$c_{{\rm cal}}$={c_cal:.2f}" if c_cal is not None
                 else "c_cal (N/A)")
    legend_handles = [
        Line2D([0], [0], color="gray", lw=1.5, label=r"$V(\sigma;\nu)$"),
        Line2D([0], [0], marker="D", color="gray", lw=0, markersize=6,
               label=r"$\sigma^\star_{\mathrm{cert}}$ (certified opt.)"),
        Line2D([0], [0], marker="s", color="gray", lw=0, markersize=6,
               mfc="none", mew=1.2,
               label=r"$\sigma^\star_{\mathrm{unc}}$ (unconstrained opt.)"),
        Line2D([0], [0], marker="x", color="gray", lw=0, markersize=6,
               mew=1.5,
               label=r"$\sigma^{\mathrm{CPB}}$ ($c=1$, analytic baseline)"),
        Line2D([0], [0], marker="*", color="red", lw=0, markersize=8,
               label=rf"$\sigma^{{\mathrm{{CPB}}}}$ ({c_cal_str}, default)"),
    ]
    ax_v.legend(handles=legend_handles, fontsize=7, loc="upper right")
    ax_v.set_xlabel(r"internal stochasticity $\sigma$")
    ax_v.set_ylabel(r"mean return $V(\sigma;\nu)$")
    ax_v.set_title(
        r"(a) $V(\sigma;\nu)$ curves with $\sigma^\star$ and "
        r"$\sigma^{\mathrm{CPB}}$ markers"
    )

    # ── Panel (b): normalized regret bar chart ────────────────────────────────
    nu_arr = np.array(nus)
    n_nus = len(nu_arr)
    n_c = len(all_c)
    bar_width = 0.8 / n_c
    offsets = np.linspace(-(n_c - 1) / 2, (n_c - 1) / 2, n_c) * bar_width

    all_violations: list[float] = []
    c_cal_regrets: list[float] = []   # mean regret at c=c_cal, for acceptance check

    for ci, (c_val, c_color) in enumerate(zip(all_c, c_cmap, strict=False)):
        regrets = []
        is_c_cal = c_cal is not None and abs(c_val - c_cal) < 1e-6
        for nu in nus:
            if nu not in curves:
                regrets.append(float("nan"))
                continue
            sigmas, returns, violations = curves[nu]
            row = summary[np.abs(summary["nu"] - nu) < 1e-9]
            if row.empty:
                regrets.append(float("nan"))
                continue
            row = row.iloc[0]
            sigma_min = float(row["sigma_min"])
            sigma_max = float(row["sigma_max"])

            s_cpb = _cpb_sigma(nu, c_val, sigma_min, sigma_max)
            v_cpb = _interp(sigmas, returns, s_cpb)
            v_star = float(row["value_at_star"])
            v0 = _interp(sigmas, returns, 0.0)
            reg = _normalized_regret(v_star, v_cpb, v0)
            regrets.append(reg)

            viol = _interp(sigmas, violations, s_cpb)
            all_violations.append(viol)
            if is_c_cal and np.isfinite(reg):
                c_cal_regrets.append(reg)

        xs = np.arange(n_nus) + offsets[ci]
        edge_color = "red" if is_c_cal else "none"
        edge_lw = 2.0 if is_c_cal else 0.0
        label = (
            rf"$c_{{\rm cal}}$={c_val:.2f} (default)"
            if is_c_cal else f"c={c_val:.1f}"
        )
        ax_r.bar(xs, regrets, width=bar_width, color=c_color, alpha=0.85,
                 label=label, edgecolor=edge_color, linewidth=edge_lw)

    ax_r.axhline(0.10, color="red", ls="--", lw=1.5, label="10% regret ref. (c_cal target)")
    ax_r.axhline(0.15, color="darkorange", ls="--", lw=1.0, label="15% regret ref.")
    ax_r.set_xticks(np.arange(n_nus))
    ax_r.set_xticklabels([f"{n:.2f}" for n in nus])
    ax_r.set_xlabel("noise rate ν")
    ax_r.set_ylabel(
        r"normalized regret $\frac{V^\star - V^{\mathrm{CPB}}}{V^\star - V(0)}$"
    )
    ax_r.set_title(r"(b) Normalized regret vs. $c$ in $\hat{m}(\nu)=c\cdot\nu$")
    ax_r.legend(fontsize=7, ncol=2)
    ax_r.set_ylim(bottom=0.0)

    # ── Safety + regret summary ───────────────────────────────────────────────
    n_safe = sum(v <= alpha for v in all_violations if np.isfinite(v))
    n_total = sum(np.isfinite(v) for v in all_violations)
    print(f"\n=== Projection Robustness Safety Check (α={alpha}) ===")
    print(f"  Safe (violation ≤ α): {n_safe}/{n_total}  ({100*n_safe/max(n_total,1):.0f}%)")
    if n_safe < n_total:
        print("  WARNING: some σ^CPB points violate the budget!")
    else:
        print("  ✓ All CPB projections satisfy R_H ≤ α across all c values.")
    if c_cal_regrets:
        mean_reg = float(np.mean(c_cal_regrets))
        print(f"  Mean normalised regret at c_cal={c_cal:.3f}: {mean_reg:.3f}")
        if mean_reg <= 0.10:
            print("  ✓ c_cal mean regret ≤ 10% — acceptance criterion met.")
        else:
            print("  NOTE: c_cal mean regret > 10%.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved projection-robustness figure → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Fig B: projection robustness across m̂ misspecification."
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("results/figures/exp1_summary.csv"),
        help="Path to exp1_summary.csv (output of plot_bandwidth.py).",
    )
    ap.add_argument(
        "--bandwidth",
        type=Path,
        default=Path("results/runs/exp1b_full_bandwidth/bandwidth.csv"),
        help="Path to raw bandwidth CSV (exp1b).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures/exp1b_projection_robustness.pdf"),
        help="Output PDF path.",
    )
    ap.add_argument(
        "--horizon", type=int, default=200,
        help="Horizon to filter from bandwidth CSV.",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.1,
        help="Violation rate threshold α.",
    )
    ap.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=_DEFAULT_C_VALUES,
        metavar="C",
        help="Ablation c values for m̂(ν)=c·ν shown in panel (b).",
    )
    ap.add_argument(
        "--c-cal-json",
        type=Path,
        default=Path("results/runs/exp1b_full_bandwidth/c_cal.json"),
        help="Path to c_cal.json (output of calibrate_c.py). "
             "If the file exists, c_cal is highlighted as the default.",
    )
    args = ap.parse_args()

    c_cal: float | None = None
    if args.c_cal_json.exists():
        with open(args.c_cal_json) as fp:
            c_cal = float(json.load(fp)["c_cal"])
        print(f"Loaded c_cal = {c_cal:.3f} from {args.c_cal_json}")
    else:
        print(f"NOTE: {args.c_cal_json} not found — c_cal markers omitted.")

    summary = pd.read_csv(args.summary)
    curves = _load_curves(args.bandwidth, args.horizon)
    plot_projection_robustness(
        summary, curves, args.c_values, args.alpha, args.out, c_cal=c_cal
    )


if __name__ == "__main__":
    main()
