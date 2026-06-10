"""Plot ν̂ sensitivity analysis results (Proposition 1).

Generates Fig 3: realized violation rate and mean return as functions of
estimation bias b = (ν̂ − ν_true) / ν_true.

Panels:
  (a) Realized violation rate R_H(σ^CPB; ν_true) vs bias — should stay ≤ α when admissible.
  (b) Realized mean return V(σ^CPB) vs bias — robustness cost.

Usage:
    python -m experiments.exp1_noisy_games.plot_nuhat_sensitivity \
        --csv results/runs/exp1b_full_bandwidth/nu_hat_sensitivity.csv \
        --out results/figures/exp1b_nuhat_sensitivity.pdf \
        --alpha 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib style ─────────────────────────────────────────────────────────────
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

_NU_COLORS = {
    0.05: "#1f77b4",
    0.10: "#ff7f0e",
    0.15: "#2ca02c",
    0.20: "#d62728",
    0.30: "#9467bd",
}


def load_results(csv_path: Path, alpha: float) -> pd.DataFrame:
    """Load and annotate the nuhat sensitivity CSV."""
    df = pd.read_csv(csv_path)
    # Aggregate over seeds
    agg = (
        df.groupby(["nu_true", "bias"])
        .agg(
            realized_violation_rate=("realized_violation_rate", "mean"),
            violation_se=("realized_violation_rate", "sem"),
            realized_mean_return=("realized_mean_return", "mean"),
            return_se=("realized_mean_return", "sem"),
            is_admissible_at_true=("is_admissible_at_true", "mean"),
        )
        .reset_index()
    )
    agg["violates_budget"] = agg["realized_violation_rate"] > alpha
    return agg


def plot_sensitivity(
    df: pd.DataFrame,
    alpha: float,
    out_path: Path,
) -> None:
    """Generate the two-panel ν̂ sensitivity figure."""
    nus = sorted(df["nu_true"].unique())
    biases = sorted(df["bias"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    ax_viol, ax_ret = axes

    for nu in nus:
        sub = df[df["nu_true"] == nu].sort_values("bias")
        color = _NU_COLORS.get(float(nu), "gray")
        label = rf"$\nu$={nu}"

        # Panel (a): violation rate vs bias
        ax_viol.errorbar(
            sub["bias"],
            sub["realized_violation_rate"],
            yerr=sub["violation_se"],
            fmt="o-",
            color=color,
            label=label,
            capsize=3,
            linewidth=1.2,
            markersize=4,
        )
        # Mark inadmissible points
        inadm = sub[sub["is_admissible_at_true"] < 1.0]
        if not inadm.empty:
            ax_viol.scatter(
                inadm["bias"],
                inadm["realized_violation_rate"],
                color=color,
                marker="x",
                s=60,
                zorder=5,
            )

        # Panel (b): return vs bias
        ax_ret.errorbar(
            sub["bias"],
            sub["realized_mean_return"],
            yerr=sub["return_se"],
            fmt="o-",
            color=color,
            label=label,
            capsize=3,
            linewidth=1.2,
            markersize=4,
        )

    # Panel (a) decorations
    ax_viol.axhline(alpha, color="black", linestyle="--", linewidth=1.0, label=rf"$\alpha$={alpha}")
    ax_viol.set_xlabel(r"Estimation bias $b = (\hat{\nu} - \nu) / \nu$")
    ax_viol.set_ylabel(r"Realized $R_H(\sigma^{\mathrm{CPB}}; \nu)$")
    ax_viol.set_title(r"(a) Violation rate vs. $\hat{\nu}$ bias")
    ax_viol.legend(fontsize=6, ncol=2)
    ax_viol.set_xlim(min(biases) - 0.05, max(biases) + 0.05)
    ax_viol.set_xticks(biases)

    # Panel (b) decorations
    ax_ret.set_xlabel(r"Estimation bias $b = (\hat{\nu} - \nu) / \nu$")
    ax_ret.set_ylabel(r"Mean return $V(\sigma^{\mathrm{CPB}})$")
    ax_ret.set_title(r"(b) Return vs. $\hat{\nu}$ bias")
    ax_ret.legend(fontsize=6, ncol=2)
    ax_ret.set_xlim(min(biases) - 0.05, max(biases) + 0.05)
    ax_ret.set_xticks(biases)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ν̂ sensitivity figure → {out_path}")

    # Print summary table
    print("\n=== ν̂ Sensitivity Summary ===")
    pivot = df.pivot_table(
        index="nu_true",
        columns="bias",
        values="realized_violation_rate",
        aggfunc="mean",
    )
    print(pivot.to_string(float_format="{:.3f}".format))
    n_inadm = (df["is_admissible_at_true"] < 1.0).sum()
    n_over = (df["realized_violation_rate"] > alpha).sum()
    print(f"\nInadmissible (σ^CPB ∉ B(ν_true)): {n_inadm}/{len(df)}")
    print(f"Budget overrun (R_H > α={alpha}):     {n_over}/{len(df)}")

    # Fraction of budget overruns that are inadmissible
    over_rows = df[df["realized_violation_rate"] > alpha]
    inadm_given_over = (over_rows["is_admissible_at_true"] < 1.0).sum()
    if len(over_rows) > 0:
        print(
            f"Of overruns, inadmissible fraction: {inadm_given_over}/{len(over_rows)} "
            f"({100 * inadm_given_over / len(over_rows):.0f}%)"
        )
    else:
        print("No budget overruns above α.")

    # Bias robustness threshold
    by_bias = df.groupby("bias")["realized_violation_rate"].max()
    safe_biases = by_bias[by_bias <= alpha].index.tolist()
    if safe_biases:
        print(f"\nMax safe bias (all ν, max R_H ≤ α): {max(safe_biases):+.1f}")
    else:
        print("\nNo bias value is safe for all ν simultaneously (see per-ν breakdown).")

    # Numpy mean/std of violation rates for non-zero ν
    nz = df[df["nu_true"] > 0]
    mean_viol = nz["realized_violation_rate"].mean()
    max_viol = nz["realized_violation_rate"].max()
    print(
        f"\nProposition 1 check (ν > 0):"
        f" mean R_H={mean_viol:.3f}, max R_H={max_viol:.3f}  (α={alpha})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ν̂ sensitivity results.")
    ap.add_argument("--csv", required=True, type=Path, help="Path to nu_hat_sensitivity.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures/exp1b_nuhat_sensitivity.pdf"),
        help="Output PDF path.",
    )
    ap.add_argument("--alpha", type=float, default=0.1, help="Violation rate threshold α.")
    args = ap.parse_args()

    df = load_results(args.csv, args.alpha)
    plot_sensitivity(df, args.alpha, args.out)


if __name__ == "__main__":
    main()
