"""Compute c_cal from held-out calibration conditions and save to JSON.

c_cal is the environment-specific slope in m̂(ν) = c·ν that minimises mean
normalised regret on a held-out set of noise levels.  It is a single scalar
calibration of the Lemma 1 analytic reference (not a new model class).

Outputs a JSON file with {c_cal, train_loss, test_loss, train_nus, test_nus}
for downstream plot scripts.

Usage:
    python -m experiments.exp1_noisy_games.calibrate_c \\
        --summary results/figures/exp1_summary.csv \\
        --bandwidth results/runs/exp1b_full_bandwidth/bandwidth.csv \\
        --train-nus 0.05 0.15 0.30 \\
        --test-nus 0.10 0.20 \\
        --out results/runs/exp1b_full_bandwidth/c_cal.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nms.core.response_map import AnalyticResponseMap


def _eval_loss(
    fitted: AnalyticResponseMap,
    eval_summary: pd.DataFrame,
    raw: pd.DataFrame,
) -> float:
    """Compute mean normalised regret for ``fitted`` on ``eval_summary``."""
    losses: list[float] = []
    for _, row in eval_summary.iterrows():
        nu = float(row["nu"])
        sigma_min = float(row["sigma_min"])
        sigma_max = float(row["sigma_max"])
        v_cert = float(row["value_at_star"])

        sub = raw[raw["nu"] == nu].sort_values("sigma")
        if sub.empty:
            continue
        agg = sub.groupby("sigma")["mean_return"].mean().reset_index().sort_values("sigma")
        sigs = agg["sigma"].to_numpy(dtype=np.float64)
        vals = agg["mean_return"].to_numpy(dtype=np.float64)

        sigma_cpb = float(np.clip(fitted.c * nu, sigma_min, sigma_max))
        v_cpb = float(np.interp(sigma_cpb, sigs, vals))
        v_base = float(np.interp(0.0, sigs, vals))
        denom = max(v_cert - v_base, 1e-6)
        losses.append((v_cert - v_cpb) / denom)

    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate single-scalar c for m̂(ν)=c·ν analytic reference."
    )
    ap.add_argument("--summary", required=True, type=Path,
                    help="exp1_summary.csv (output of plot_bandwidth.py).")
    ap.add_argument("--bandwidth", required=True, type=Path,
                    help="Raw bandwidth CSV (exp1b).")
    ap.add_argument("--train-nus", nargs="+", type=float, required=True,
                    metavar="NU", help="ν values used for calibration.")
    ap.add_argument("--test-nus", nargs="+", type=float, required=True,
                    metavar="NU", help="ν values used for evaluation.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output JSON: {c_cal, train_loss, test_loss, ...}")
    ap.add_argument("--horizon", type=int, default=200,
                    help="Horizon filter for bandwidth CSV.")
    ap.add_argument("--c-min", type=float, default=0.0)
    ap.add_argument("--c-max", type=float, default=10.0)
    ap.add_argument("--c-steps", type=int, default=201,
                    help="Resolution of c grid.")
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    raw = pd.read_csv(args.bandwidth)
    if "horizon" in raw.columns:
        raw = raw[raw["horizon"] == args.horizon]

    train_nus_set = set(args.train_nus)
    test_nus_set = set(args.test_nus)
    train_summary = summary[summary["nu"].apply(
        lambda n: any(abs(n - t) < 1e-9 for t in train_nus_set)
    )]
    test_summary = summary[summary["nu"].apply(
        lambda n: any(abs(n - t) < 1e-9 for t in test_nus_set)
    )]

    if train_summary.empty:
        raise ValueError(
            f"No rows found in summary for train_nus={args.train_nus}. "
            "Check --summary path and --train-nus values."
        )

    c_grid = np.linspace(args.c_min, args.c_max, args.c_steps)
    print(f"Calibrating c on ν = {sorted(train_nus_set)}  "
          f"(grid: {args.c_min}..{args.c_max}, {args.c_steps} steps) ...")
    fitted = AnalyticResponseMap.calibrate(train_summary, raw, c_grid=c_grid)

    train_loss = _eval_loss(fitted, train_summary, raw)
    test_loss = _eval_loss(fitted, test_summary, raw)

    result = {
        "c_cal": fitted.c,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_nus": args.train_nus,
        "test_nus": args.test_nus,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(result, fp, indent=2)

    print(
        f"c_cal = {fitted.c:.3f}  |  "
        f"train normalised regret = {train_loss:.3f}  |  "
        f"test = {test_loss:.3f}"
    )
    if test_loss > 0.20:
        print(
            "WARNING: test_loss > 0.20 — single-scalar calibration may be "
            "insufficient. Consider a ν-dependent model m(ν, ρ)."
        )
    elif test_loss > 0.15:
        print(
            "NOTE: test_loss is between 0.15 and 0.20. "
            "Acceptable but watch for higher-ν regret."
        )
    else:
        print("✓ test_loss < 0.15 — calibration criterion met.")


if __name__ == "__main__":
    main()
