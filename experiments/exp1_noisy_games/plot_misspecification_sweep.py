"""Fig C — Misspecification Sweep: safety under varying m̂(ν) = c·ν slope.

Three-panel figure sweeping c ∈ {0, 0.5, 1, 2, 4, 8}:
  (a) Realized violation rate R_H(σ^CPB; ν) vs c for each ν.
      All lines must stay ≤ α (primary safety acceptance criterion).
  (b) Normalized regret (V* - V^CPB)/(V* - V(0)) vs c for each ν.
      Shows value cost of using the wrong c.
  (c) Projected σ^CPB = clip(c·ν, σ_min, σ_max) vs c for each ν.
      Shows which part of the bandwidth is used at each c.

When --c-cal-json is provided (or found at the default path), the calibrated
slope c_cal is added to the sweep and highlighted with a thick red vertical
line labelled "c_cal = X.XX (default)" on all three panels.

Acceptance criterion: all violations ≤ α=0.1 across all (c, ν) pairs.

Usage:
    python -m experiments.exp1_noisy_games.plot_misspecification_sweep \\
        --summary results/figures/exp1_summary.csv \\
        --bandwidth results/runs/exp1b_full_bandwidth/bandwidth.csv \\
        --out results/figures/exp1b_misspecification_sweep.pdf
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

_DEFAULT_C_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

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
    """Return {ν: (sigmas, returns, violations)}."""
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
    if not np.isfinite(sigma_min) or sigma_min > sigma_max:
        return float("nan")
    return float(np.clip(c * nu, sigma_min, sigma_max))


def _interp(sigmas: np.ndarray, vals: np.ndarray, s: float) -> float:
    if not np.isfinite(s):
        return float("nan")
    return float(np.interp(s, sigmas, vals))


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_misspecification_sweep(
    summary: pd.DataFrame,
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    c_values: list[float],
    alpha: float,
    out_path: Path,
    c_cal: float | None = None,
) -> None:
    nus = sorted(curves.keys())

    # Merge c_cal into the sweep array (sorted, deduplicated)
    all_c = sorted(set(c_values) | ({round(c_cal, 6)} if c_cal is not None else set()))
    c_arr = np.array(all_c)

    # Build result arrays: shape (len(nus), len(all_c))
    n_c = len(all_c)
    violations = np.full((len(nus), n_c), np.nan)
    regrets = np.full((len(nus), n_c), np.nan)
    cpb_sigmas = np.full((len(nus), n_c), np.nan)

    for ni, nu in enumerate(nus):
        if nu not in curves:
            continue
        sigmas, returns, viols = curves[nu]
        row = summary[np.abs(summary["nu"] - nu) < 1e-9]
        if row.empty:
            continue
        row = row.iloc[0]
        sigma_min = float(row["sigma_min"])
        sigma_max = float(row["sigma_max"])
        v_star = float(row["value_at_star"])
        v0 = _interp(sigmas, returns, 0.0)

        for ci, c_val in enumerate(all_c):
            s_cpb = _cpb_sigma(nu, c_val, sigma_min, sigma_max)
            cpb_sigmas[ni, ci] = s_cpb
            violations[ni, ci] = _interp(sigmas, viols, s_cpb)
            v_cpb = _interp(sigmas, returns, s_cpb)
            denom = max(v_star - v0, 1e-6)
            regrets[ni, ci] = (v_star - v_cpb) / denom

    # Safety check
    all_safe = bool(np.all(violations[np.isfinite(violations)] <= alpha))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ax_viol, ax_reg, ax_sig = axes

    for ni, nu in enumerate(nus):
        color = _NU_COLORS.get(nu, "gray")
        label = rf"$\nu$={nu:.2f}"

        # Panel (a): violation vs c
        ax_viol.plot(c_arr, violations[ni], "o-", color=color, lw=1.5,
                     markersize=4, label=label)

        # Panel (b): regret vs c
        ax_reg.plot(c_arr, regrets[ni], "o-", color=color, lw=1.5,
                    markersize=4, label=label)

        # Panel (c): σ^CPB vs c, with bandwidth shading
        ax_sig.plot(c_arr, cpb_sigmas[ni], "o-", color=color, lw=1.5,
                    markersize=4, label=label)
        row = summary[np.abs(summary["nu"] - nu) < 1e-9]
        if not row.empty:
            smin = float(row.iloc[0]["sigma_min"])
            smax = float(row.iloc[0]["sigma_max"])
            ax_sig.axhspan(smin, smax, color=color, alpha=0.07)

    # c_cal vertical marker on all three panels
    if c_cal is not None:
        c_cal_label = rf"$c_{{\rm cal}}$={c_cal:.2f} (default)"
        for ax in (ax_viol, ax_reg, ax_sig):
            ax.axvline(c_cal, color="red", lw=2.5, ls="-",
                       label=c_cal_label, zorder=10)
            c_cal_label = "_nolegend_"   # only label once per panel group

    # Panel (a) decorations
    ax_viol.axhline(alpha, color="black", ls="--", lw=1.2,
                    label=rf"$\alpha$={alpha}")
    ax_viol.set_xlabel(r"slope $c$ in $\hat{m}(\nu)=c\cdot\nu$")
    ax_viol.set_ylabel(r"realized $R_H(\sigma^{\mathrm{CPB}};\nu)$")
    ax_viol.set_title(
        "(a) Violation rate vs. c\n"
        r"(all lines must stay $\leq\alpha$)"
    )
    ax_viol.legend(fontsize=7, ncol=2)
    # Show base c-values as ticks; c_cal gets its own tick if it falls between them
    ax_viol.set_xticks(sorted(set(c_values) | ({c_cal} if c_cal is not None else set())))
    ax_viol.set_xticklabels(
        [f"{v:.2f}" if v == c_cal else f"{v:g}"
         for v in sorted(set(c_values) | ({c_cal} if c_cal is not None else set()))],
        rotation=45, ha="right", fontsize=7,
    )
    # Color the safe region
    ax_viol.axhspan(0, alpha, color="limegreen", alpha=0.08, zorder=0)

    # Safety annotation
    safety_text = "✓ All safe" if all_safe else "✗ Safety violated!"
    safety_color = "darkgreen" if all_safe else "red"
    ax_viol.text(
        0.97, 0.97, safety_text,
        transform=ax_viol.transAxes, ha="right", va="top",
        fontsize=9, color=safety_color,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Panel (b) decorations
    ax_reg.axhline(0.15, color="darkorange", ls="--", lw=1.0, label="15% ref.")
    ax_reg.set_xlabel(r"slope $c$ in $\hat{m}(\nu)=c\cdot\nu$")
    ax_reg.set_ylabel(
        r"normalized regret $\frac{V^\star-V^{\mathrm{CPB}}}{V^\star-V(0)}$"
    )
    ax_reg.set_title("(b) Normalized regret vs. c")
    ax_reg.legend(fontsize=7, ncol=2)
    ax_reg.set_xticks(sorted(set(c_values) | ({c_cal} if c_cal is not None else set())))
    ax_reg.set_xticklabels(
        [f"{v:.2f}" if v == c_cal else f"{v:g}"
         for v in sorted(set(c_values) | ({c_cal} if c_cal is not None else set()))],
        rotation=45, ha="right", fontsize=7,
    )
    ax_reg.set_ylim(bottom=0.0)

    # Panel (c) decorations
    ax_sig.set_xlabel(r"slope $c$ in $\hat{m}(\nu)=c\cdot\nu$")
    ax_sig.set_ylabel(
        r"$\sigma^{\mathrm{CPB}} = "
        r"\mathrm{clip}(c\cdot\nu,\,\sigma_{\min},\sigma_{\max})$"
    )
    ax_sig.set_title(
        r"(c) Projected $\sigma^{\mathrm{CPB}}$ vs. $c$"
        "\n(shading = bandwidth B(ν))"
    )
    ax_sig.legend(fontsize=7, ncol=2)
    ax_sig.set_xticks(sorted(set(c_values) | ({c_cal} if c_cal is not None else set())))
    ax_sig.set_xticklabels(
        [f"{v:.2f}" if v == c_cal else f"{v:g}"
         for v in sorted(set(c_values) | ({c_cal} if c_cal is not None else set()))],
        rotation=45, ha="right", fontsize=7,
    )

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n=== Misspecification Sweep: Violation Rates ===")
    c_label_row = [
        f"{c:>6.2f}{'*' if c_cal is not None and abs(c - c_cal) < 1e-6 else ' '}"
        for c in all_c
    ]
    hdr = "   ν  \\ c   " + "  ".join(c_label_row)
    print(hdr)
    print("-" * len(hdr))
    for ni, nu in enumerate(nus):
        row_str = f"  {nu:.2f}      " + "  ".join(
            f"{violations[ni, ci]:6.4f} " for ci in range(len(all_c))
        )
        print(row_str)
    if c_cal is not None:
        print(f"  (* marks c_cal = {c_cal:.3f})")

    n_unsafe = int(np.sum(violations[np.isfinite(violations)] > alpha))
    n_total = int(np.sum(np.isfinite(violations)))
    print(f"\nSafe (≤ α={alpha}): {n_total - n_unsafe}/{n_total}")
    if n_unsafe > 0:
        print("  WARNING: some (ν, c) pairs exceed the violation budget!")
    else:
        print("  ✓ Safety preserved for all c × ν combinations.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved misspecification-sweep figure → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Fig C: misspecification sweep over c in m̂(ν)=c·ν."
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("results/figures/exp1_summary.csv"),
        help="Path to exp1_summary.csv.",
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
        default=Path("results/figures/exp1b_misspecification_sweep.pdf"),
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
        help="Ablation slope values c to sweep (m̂(ν)=c·ν).",
    )
    ap.add_argument(
        "--c-cal-json",
        type=Path,
        default=Path("results/runs/exp1b_full_bandwidth/c_cal.json"),
        help="Path to c_cal.json (output of calibrate_c.py). "
             "If the file exists, c_cal is added to the sweep and highlighted.",
    )
    args = ap.parse_args()

    # Load c_cal if JSON exists
    c_cal: float | None = None
    if args.c_cal_json.exists():
        with open(args.c_cal_json) as fp:
            c_cal = float(json.load(fp)["c_cal"])
        print(f"Loaded c_cal = {c_cal:.3f} from {args.c_cal_json}")
    else:
        print(f"NOTE: {args.c_cal_json} not found — c_cal line omitted.")

    summary = pd.read_csv(args.summary)
    curves = _load_curves(args.bandwidth, args.horizon)
    plot_misspecification_sweep(
        summary, curves, args.c_values, args.alpha, args.out, c_cal=c_cal
    )


if __name__ == "__main__":
    main()
