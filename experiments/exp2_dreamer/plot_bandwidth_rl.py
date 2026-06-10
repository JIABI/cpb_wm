"""Exp 2 Fig 3: NMS bandwidth curves for DreamerV3 (Phase B).

Plots R_H(σ; ν) vs σ for multiple ν values, with certified bandwidth
B_{H,α}(ν) marked.  Mirrors Fig 1 from Exp 1 but for the RL setting.

Usage::

    # Synthetic (no data required)
    python -m experiments.exp2_dreamer.plot_bandwidth_rl --synthetic

    # From real calibration data
    python -m experiments.exp2_dreamer.plot_bandwidth_rl \\
        --data results/runs/exp2_cheetah_run/seed0/bandwidth/bandwidth_curve.csv \\
        --bandwidth-json results/runs/exp2_cheetah_run/seed0/bandwidth/bandwidth.json \\
        --out results/figures/exp2_bandwidth_rl.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _make_synthetic_data(
    nu_values: list[float],
    sigma_grid: np.ndarray,
    alpha: float,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Generate synthetic R_H curves for plotting without real data."""
    rng = np.random.default_rng(seed)
    curves: dict[float, np.ndarray] = {}
    bandwidths: dict[float, float] = {}
    for nu in nu_values:
        # R_H increases with σ and nu
        r_h = 1 - np.exp(-3.0 * sigma_grid / (nu + 0.05)) + rng.normal(0, 0.02, len(sigma_grid))
        r_h = np.clip(r_h, 0, 1)
        curves[nu] = r_h
        # Certified bandwidth: largest σ with R_H ≤ α
        safe = sigma_grid[r_h <= alpha]
        bandwidths[nu] = float(safe[-1]) if len(safe) > 0 else 0.0
    return curves, bandwidths


def plot_bandwidth_rl(
    curves: dict[float, np.ndarray],
    sigma_grid: np.ndarray,
    bandwidths: dict[float, float],
    alpha: float,
    out_path: str | Path,
    title: str = "World-model violation rate $R_H(\\sigma; \\nu)$",
) -> None:
    nu_values = sorted(curves.keys())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(nu_values) - 1, 1)) for i in range(len(nu_values))]

    fig, ax = plt.subplots(figsize=(6, 4))

    for nu, color in zip(nu_values, colors, strict=False):
        r_h = curves[nu]
        bw = bandwidths.get(nu, 0.0)
        ax.plot(sigma_grid, r_h, color=color, lw=1.8, label=f"ν={nu:.2f}")
        ax.axvline(bw, color=color, lw=1.2, ls="--", alpha=0.7)

    ax.axhline(alpha, color="red", lw=1.5, ls=":", label=f"α={alpha}")
    ax.set_xlabel("σ (actor temperature)")
    ax.set_ylabel("$R_H(\\sigma; \\nu)$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 2 bandwidth (Fig 3).")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", type=Path, default=None,
                    help="bandwidth_curve.csv from calibrate_bandwidth.py")
    ap.add_argument("--bandwidth-json", type=Path, default=None)
    ap.add_argument("--nu-values", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/exp2_bandwidth_rl.pdf"))
    args = ap.parse_args()

    sigma_grid = np.linspace(0.0, 0.5, 51)

    if args.synthetic or args.data is None:
        print("[synthetic] Using synthetic data ...")
        curves, bandwidths = _make_synthetic_data(args.nu_values, sigma_grid, args.alpha, args.seed)
    else:
        import json

        import pandas as pd

        df = pd.read_csv(args.data)
        curves = {}
        for nu, grp in df.groupby("nu"):
            agg = grp.sort_values("sigma")
            curves[float(nu)] = np.interp(sigma_grid, agg["sigma"].values, agg["r_h"].values)

        if args.bandwidth_json and args.bandwidth_json.exists():
            bw_data = json.loads(args.bandwidth_json.read_text())
            bandwidths = {float(k): float(v) for k, v in bw_data["bandwidth"].items()}
        else:
            bandwidths = {}

    plot_bandwidth_rl(curves, sigma_grid, bandwidths, args.alpha, args.out)


if __name__ == "__main__":
    main()
