"""Exp 2 Table 2 + Fig 7: Ablation study plots (Phase E).

Generates both a grouped bar chart (Fig 7) and a LaTeX table (Table 2).

Usage::

    python -m experiments.exp2_dreamer.plot_ablation --synthetic
    python -m experiments.exp2_dreamer.plot_ablation \\
        --results results/runs/exp2_ablation/ablation_results.json \\
        --out results/figures/exp2_ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ABLATION_METHODS = ["cpb_full", "cpb_match_only", "cpb_cap_only", "dreamer_default"]
_NU_VALUES = [0.05, 0.10, 0.20]


def _make_synthetic(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for method in _ABLATION_METHODS:
        for nu in _NU_VALUES:
            if method == "cpb_full":
                mean_r = 100.0 * (1 - nu * 0.5) + rng.normal(0, 3)
            elif method == "cpb_match_only":
                mean_r = 100.0 * (1 - nu * 0.8) + rng.normal(0, 3)
            elif method == "cpb_cap_only":
                mean_r = 100.0 * (1 - nu * 1.0) + rng.normal(0, 4)
            else:
                mean_r = 100.0 * (1 - nu * 1.5) + rng.normal(0, 5)
            rows.append({"method": method, "nu": nu, "mean_return": float(mean_r),
                         "std_return": float(rng.uniform(3, 8))})
    return rows


def plot_ablation_bars(rows: list[dict], out_path: str | Path) -> None:
    nu_values = sorted(set(r["nu"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))

    fig, axes = plt.subplots(1, len(nu_values), figsize=(4 * len(nu_values), 4), sharey=True)
    if len(nu_values) == 1:
        axes = [axes]
    cmap = plt.cm.tab10
    bar_width = 0.7 / len(methods)

    for ax_i, nu in enumerate(nu_values):
        ax = axes[ax_i]
        nu_rows = {r["method"]: r for r in rows if abs(r["nu"] - nu) < 1e-6}
        for j, method in enumerate(methods):
            if method not in nu_rows:
                continue
            r = nu_rows[method]
            x = j * bar_width - (len(methods) - 1) * bar_width / 2
            ax.bar(x, r["mean_return"], width=bar_width * 0.9,
                   color=cmap(j % 10), label=method if ax_i == 0 else None,
                   yerr=r.get("std_return", 0), capsize=3)
        ax.set_title(f"ν={nu:.2f}")
        ax.set_xticks([])
        ax.set_xlabel("Method")
        if ax_i == 0:
            ax.set_ylabel("Mean return")

    axes[0].legend(fontsize=7)
    fig.suptitle("Ablation: CPB component contributions", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def make_latex_table(rows: list[dict], out_path: str | Path) -> None:
    nu_values = sorted(set(r["nu"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["\\begin{tabular}{l" + "c" * len(nu_values) + "}",
             "\\toprule",
             "Method & " + " & ".join(f"$\\nu={nu}$" for nu in nu_values) + " \\\\",
             "\\midrule"]
    for method in methods:
        vals = []
        for nu in nu_values:
            found = next((r for r in rows if r["method"] == method and abs(r["nu"] - nu) < 1e-6), None)  # noqa: E501
            vals.append(f"{found['mean_return']:.1f}" if found else "--")
        lines.append(method.replace("_", "\\_") + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(out_path).write_text("\n".join(lines))
    print(f"LaTeX table → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 2 ablation (Table 2 + Fig 7).")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/exp2_ablation"))
    args = ap.parse_args()

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.results is None:
        print("[synthetic] Using synthetic ablation data ...")
        rows = _make_synthetic(args.seed)
    else:
        rows = json.loads(args.results.read_text())

    plot_ablation_bars(rows, out_base.with_suffix(".pdf"))
    make_latex_table(rows, out_base.with_suffix(".tex"))


if __name__ == "__main__":
    main()
