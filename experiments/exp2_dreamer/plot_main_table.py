"""Exp 2 Table 1: Main results table across all 12 methods × 9 noise conditions.

Generates a LaTeX table and a heatmap PNG.

Usage::

    python -m experiments.exp2_dreamer.plot_main_table --synthetic
    python -m experiments.exp2_dreamer.plot_main_table \\
        --results-dir results/runs/exp2_main \\
        --out results/figures/exp2_main_table
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_METHODS = [
    "cpb_full", "cpb_match_only", "cpb_cap_only",
    "dreamer_default", "dreamer_fixed_001", "dreamer_fixed_01", "dreamer_fixed_05",
    "dreamer_autotuned", "ppo_entropy", "noisynet", "ensemble_disag", "rnd",
]
_NOISE_CONDITIONS = [
    "clean",
    "stoch_dyn_005", "stoch_dyn_010", "stoch_dyn_020",
    "obs_noise_010", "obs_noise_020", "obs_noise_030",
    "reward_stoch_010", "reward_stoch_030",
]


def _make_synthetic(seed: int = 0) -> dict[str, dict[str, float]]:
    """Generate synthetic mean-return table."""
    rng = np.random.default_rng(seed)
    data: dict[str, dict[str, float]] = {}
    for method in _METHODS:
        data[method] = {}
        for noise in _NOISE_CONDITIONS:
            # CPB methods perform better, especially under noise
            if method.startswith("cpb"):
                base = 80.0 + rng.normal(0, 5)
            elif method.startswith("dreamer"):
                base = 65.0 + rng.normal(0, 8)
            else:
                base = 55.0 + rng.normal(0, 10)
            # Add noise penalty
            noise_penalty = 0.0
            if "005" in noise:
                noise_penalty = 5
            elif "010" in noise:
                noise_penalty = 10
            elif "020" in noise:
                noise_penalty = 18
            elif "030" in noise:
                noise_penalty = 25
            # CPB mitigates noise penalty
            if method.startswith("cpb"):
                noise_penalty *= 0.4
            data[method][noise] = float(base - noise_penalty + rng.normal(0, 2))
    return data


def _make_heatmap(
    data: dict[str, dict[str, float]],
    out_path: str | Path,
) -> None:
    methods = list(data.keys())
    noises = _NOISE_CONDITIONS
    arr = np.array([[data[m].get(n, np.nan) for n in noises] for m in methods])

    fig, ax = plt.subplots(figsize=(len(noises) * 1.1, len(methods) * 0.5 + 1))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels(noises, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title("Mean Return (12 methods × 9 noise conditions)", fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean return")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(noises)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _make_latex_table(data: dict[str, dict[str, float]], out_path: str | Path) -> None:
    noises = _NOISE_CONDITIONS
    rows = []
    rows.append("\\begin{tabular}{l" + "c" * len(noises) + "}")
    rows.append("\\toprule")
    header = "Method & " + " & ".join(n.replace("_", "\\_") for n in noises) + " \\\\"
    rows.append(header)
    rows.append("\\midrule")
    for method, results in data.items():
        vals = [f"{results.get(n, 0):.1f}" for n in noises]
        rows.append(method.replace("_", "\\_") + " & " + " & ".join(vals) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    Path(out_path).write_text("\n".join(rows))
    print(f"LaTeX table → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot/generate Exp 2 main table (Table 1).")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/exp2_main_table"))
    args = ap.parse_args()

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.results_dir is None:
        print("[synthetic] Generating synthetic main table ...")
        data = _make_synthetic(seed=args.seed)
    else:
        data: dict[str, dict[str, float]] = {}
        for fpath in glob.glob(str(args.results_dir / "**" / "run_info.json"), recursive=True):
            info = json.loads(Path(fpath).read_text())
            method = info.get("method", "unknown")
            noise = "clean"  # infer from path
            for n in _NOISE_CONDITIONS:
                if n in str(fpath):
                    noise = n
                    break
            if method not in data:
                data[method] = {}
            data[method][noise] = float(info.get("mean_return_last_100", 0.0))

    _make_heatmap(data, out_base.with_suffix(".pdf"))
    _make_latex_table(data, out_base.with_suffix(".tex"))


if __name__ == "__main__":
    main()
