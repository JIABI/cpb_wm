"""Exp 2 Fig 4: Policy recovery curves after abrupt noise onset (Phase E).

Usage::

    python -m experiments.exp2_dreamer.plot_recovery --synthetic
    python -m experiments.exp2_dreamer.plot_recovery \\
        --results results/runs/exp2_recovery/*/recovery_results.json \\
        --out results/figures/exp2_recovery.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _make_synthetic(methods: list[str], horizon: int, switch_step: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_data = {}
    for _i, method in enumerate(methods):
        # CPB recovers faster
        if "cpb" in method:
            recovery_delay = int(rng.integers(20, 60))
        elif method == "dreamer_default":
            recovery_delay = int(rng.integers(80, 180))
        else:
            recovery_delay = int(rng.integers(50, 120))

        pre = rng.normal(1.0, 0.05, switch_step)
        drop = rng.normal(0.3, 0.1, recovery_delay)
        n_rec = horizon - switch_step - recovery_delay
        recover = np.linspace(0.3, 0.9, n_rec) + rng.normal(0, 0.05, n_rec)
        post = np.concatenate([drop, recover])
        post = post[: horizon - switch_step]
        rewards = np.concatenate([pre, post])
        all_data[method] = rewards
    return all_data


def plot_recovery(
    methods_data: dict[str, np.ndarray],
    switch_step: int,
    out_path: str | Path,
    alpha: float = 0.1,
    title: str = "Policy recovery after abrupt noise onset",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(methods_data))]

    for (method, rewards), color in zip(methods_data.items(), colors, strict=False):
        # Smooth with rolling average
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        t = np.arange(len(smoothed)) + window // 2
        ax.plot(t, smoothed, label=method, color=color, lw=1.8)

    ax.axvline(switch_step, color="black", lw=1.5, ls="--", label="Noise onset")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (rolling avg)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp 2 recovery (Fig 4).")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", nargs="*", default=[],
                    help="Glob of recovery_results.json files.")
    ap.add_argument("--switch-step", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/exp2_recovery.pdf"))
    args = ap.parse_args()

    _default_methods = ["cpb_full", "cpb_match_only", "dreamer_default", "ppo_entropy"]

    if args.synthetic or not args.results:
        print("[synthetic] Using synthetic recovery data ...")
        methods_data = _make_synthetic(_default_methods, args.horizon, args.switch_step, args.seed)
    else:
        methods_data = {}
        for pattern in args.results:
            for fpath in glob.glob(str(pattern)):
                data = json.loads(Path(fpath).read_text())
                method = data.get("method", Path(fpath).parent.name)
                if "mean_rewards" in data:
                    methods_data[method] = np.array(data["mean_rewards"])

    plot_recovery(methods_data, args.switch_step, args.out)


if __name__ == "__main__":
    main()
