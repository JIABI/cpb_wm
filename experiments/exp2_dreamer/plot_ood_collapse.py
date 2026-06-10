"""Exp 2 Fig 5: OOD performance collapse under physics shifts (Phase D).

Plots mean return vs mass multiplier for each method.

Usage::

    python -m experiments.exp2_dreamer.plot_ood_collapse --synthetic
    python -m experiments.exp2_dreamer.plot_ood_collapse \\
        --results results/runs/exp2_ood/*/ood_results.json \\
        --out results/figures/exp2_ood_collapse.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_METHODS_TO_PLOT = ["cpb_full", "dreamer_default", "ppo_entropy", "rnd"]


def _make_synthetic(
    mass_mults: list[float],
    methods: list[str],
    seed: int,
) -> dict[str, dict[float, float]]:
    rng = np.random.default_rng(seed)
    data: dict[str, dict[float, float]] = {}
    for method in methods:
        data[method] = {}
        for m in mass_mults:
            deviation = abs(m - 1.0)
            if method.startswith("cpb"):
                decay = np.exp(-deviation * 1.0)
            elif method == "dreamer_default":
                decay = np.exp(-deviation * 2.0)
            else:
                decay = np.exp(-deviation * 1.5)
            data[method][m] = float(100.0 * decay + rng.normal(0, 3))
    return data


def plot_ood_collapse(
    data: dict[str, dict[float, float]],
    out_path: str | Path,
    title: str = "OOD performance under mass perturbation",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.tab10
    for i, (method, results) in enumerate(data.items()):
        mass_mults = sorted(results.keys())
        returns = [results[m] for m in mass_mults]
        ax.plot(mass_mults, returns, "o-", color=cmap(i % 10), lw=1.8, label=method)

    ax.axvline(1.0, color="gray", lw=1.0, ls="--", label="Training mass")
    ax.set_xlabel("Mass multiplier")
    ax.set_ylabel("Mean episodic return")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 2 OOD collapse (Fig 5).")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--mass-mults", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/exp2_ood_collapse.pdf"))
    args = ap.parse_args()

    if args.synthetic or not args.results:
        print("[synthetic] Using synthetic OOD data ...")
        data = _make_synthetic(args.mass_mults, _METHODS_TO_PLOT, args.seed)
    else:
        data: dict[str, dict[float, float]] = {}
        for pattern in args.results:
            for fpath in glob.glob(str(pattern)):
                info = json.loads(Path(fpath).read_text())
                method = info.get("method", Path(fpath).parent.name)
                if method not in data:
                    data[method] = {}
                for r in info.get("results", []):
                    data[method][float(r.get("mass_mult", 1.0))] = float(r.get("mean_return", 0.0))

    plot_ood_collapse(data, args.out)


if __name__ == "__main__":
    main()
