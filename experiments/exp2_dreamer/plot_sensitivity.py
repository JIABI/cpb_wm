"""Exp 2 Fig 7b: Sensitivity analysis — CPB bandwidth vs α, H (Phase E).

Plots how the certified bandwidth B_{H,α}(ν) changes with:
  - Panel (a): α ∈ {0.05, 0.10, 0.15, 0.20}
  - Panel (b): H ∈ {5, 10, 15, 25}

Usage::

    python -m experiments.exp2_dreamer.plot_sensitivity --synthetic
    python -m experiments.exp2_dreamer.plot_sensitivity \\
        --bandwidth-dir results/runs/exp2_sensitivity \\
        --out results/figures/exp2_sensitivity.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _make_synthetic_alpha(
    nu_values: list[float],
    alpha_values: list[float],
    seed: int,
) -> dict[float, list[float]]:
    """bandwidth[alpha] = list of bw per nu."""
    rng = np.random.default_rng(seed)
    data: dict[float, list[float]] = {}
    for alpha in alpha_values:
        bw_list = []
        for nu in nu_values:
            # Larger alpha → larger certified bandwidth
            bw = nu * (1.5 + 3.0 * alpha) + rng.normal(0, 0.005)
            bw_list.append(max(0.0, float(bw)))
        data[alpha] = bw_list
    return data


def _make_synthetic_horizon(
    nu_values: list[float],
    horizon_values: list[int],
    seed: int,
) -> dict[int, list[float]]:
    rng = np.random.default_rng(seed)
    data: dict[int, list[float]] = {}
    for h in horizon_values:
        bw_list = []
        for nu in nu_values:
            # Longer horizon → smaller bandwidth (harder to certify)
            bw = nu * 2.0 / (1.0 + 0.05 * h) + rng.normal(0, 0.005)
            bw_list.append(max(0.0, float(bw)))
        data[h] = bw_list
    return data


def plot_sensitivity(
    nu_values: list[float],
    alpha_data: dict[float, list[float]],
    horizon_data: dict[int, list[float]],
    out_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.cm.tab10

    # Panel (a): α sensitivity
    ax = axes[0]
    for i, (alpha, bws) in enumerate(sorted(alpha_data.items())):
        ax.plot(nu_values, bws, "o-", color=cmap(i % 10), lw=1.8, label=f"α={alpha:.2f}")
    ax.set_xlabel("ν")
    ax.set_ylabel("$B_{H,\\alpha}(\\nu)$")
    ax.set_title("Bandwidth vs violation threshold α")
    ax.legend(fontsize=8)

    # Panel (b): horizon sensitivity
    ax = axes[1]
    for i, (h, bws) in enumerate(sorted(horizon_data.items())):
        ax.plot(nu_values, bws, "s-", color=cmap(i % 10), lw=1.8, label=f"H={h}")
    ax.set_xlabel("ν")
    ax.set_title("Bandwidth vs imagination horizon H")
    ax.legend(fontsize=8)

    fig.suptitle("CPB bandwidth sensitivity analysis", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 2 sensitivity (Fig 7b).")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--bandwidth-dir", type=Path, default=None)
    ap.add_argument("--nu-values", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--alpha-values", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--horizon-values", type=int, nargs="+",
                    default=[5, 10, 15, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/exp2_sensitivity.pdf"))
    args = ap.parse_args()

    if args.synthetic or args.bandwidth_dir is None:
        print("[synthetic] Using synthetic sensitivity data ...")
        alpha_data = _make_synthetic_alpha(args.nu_values, args.alpha_values, args.seed)
        horizon_data = _make_synthetic_horizon(args.nu_values, args.horizon_values, args.seed)
    else:
        # Load from bandwidth sweep results
        import json
        alpha_data = {}
        horizon_data = {}
        for json_path in args.bandwidth_dir.glob("*.json"):
            info = json.loads(json_path.read_text())
            if "alpha" in info and "horizon" not in json_path.stem:
                alpha_val = float(info.get("alpha", 0.1))
                bw_dict = info.get("bandwidth", {})
                bw_list = [bw_dict.get(str(nu), 0.0) for nu in args.nu_values]
                alpha_data[alpha_val] = bw_list

    plot_sensitivity(args.nu_values, alpha_data, horizon_data, args.out)


if __name__ == "__main__":
    main()
