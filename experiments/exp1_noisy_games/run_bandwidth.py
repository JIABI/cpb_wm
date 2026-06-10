"""Exp 1: sweep (nu, sigma, seed) on noisy IPD — Phase A/B/C v4.

Writes a tidy CSV that downstream plot_bandwidth.py and plot_cir.py use to
reconstruct R_H(σ; ν) curves, certified bandwidths, and theory panels.

Phase A: opponent mixture (e.g. "reciprocal:0.8,always_defect:0.2") +
         composite violation indicator ("spiral_or_exploit") → U-shaped curves.

Phase C: --horizons accepts a list; the outer loop sweeps H and appends a
         "horizon" column.  The H=200 row reuses Phase A data (run separately).

Usage — Phase A smoke run (~1 min CPU):
    python -m experiments.exp1_noisy_games.run_bandwidth \
        --noise-levels 0.0 0.1 0.2 \
        --sigma-min 0.0 --sigma-max 0.8 --sigma-n 11 \
        --horizons 50 --episodes 200 --seeds 3 \
        --opponent-mixture "reciprocal:0.8,always_defect:0.2" \
        --violation-mode spiral_or_exploit \
        --out results/runs/exp1_noisy_games/smoke_mixture.csv

Usage — self-play baseline (same format, different mixture):
    python -m experiments.exp1_noisy_games.run_bandwidth \
        --opponent-mixture "reciprocal:1.0" \
        --violation-mode spiral_only \
        --out results/runs/exp1_noisy_games/bandwidth_selfplay.csv

Usage — Phase C horizon sweep:
    python -m experiments.exp1_noisy_games.run_bandwidth \
        --noise-levels 0.1 0.2 \
        --horizons 50 100 200 400 \
        --opponent-mixture "reciprocal:0.8,always_defect:0.2" \
        --violation-mode spiral_or_exploit \
        --out results/runs/exp1_noisy_games/bandwidth_cir.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nms.envs.noisy_games import rollout_ipd_with_mixture
from nms.metrics.game_violation import (
    composite_violation,
    cooperation_rate,
    exploitation_spiral,
    false_retaliation_rate,
    mean_return,
)
from nms.policies.opponents import parse_mixture_spec
from nms.policies.reciprocal import SoftReciprocalPolicy
from nms.utils.logging import write_run_info
from nms.utils.seeds import set_global_seed


def run_one(
    nu: float,
    sigma: float,
    opp_spec: str,
    viol_mode: str,
    spiral_len: int,
    exploit_len: int,
    safe_threshold: float,
    episodes: int,
    horizon: int,
    seed: int,
) -> dict[str, float]:
    """Run ``episodes`` episodes for a single (nu, sigma, seed) configuration.

    When ``opp_spec`` is "reciprocal:1.0" the rollout is pure self-play
    (uses rollout_ipd for efficiency).  Otherwise uses rollout_ipd_with_mixture.

    Returns aggregated statistics across episodes.
    """
    rng = set_global_seed(seed)
    viols, exploit_rates, coops, falses, returns = [], [], [], [], []

    # Parse mixture once per (sigma, seed) combination.
    # "reciprocal:1.0" is the pure self-play baseline — all opponents are
    # SoftReciprocalPolicy(sigma), equivalent to the original rollout_ipd call.
    mixture = parse_mixture_spec(opp_spec, agent_sigma=sigma)

    for _ in range(episodes):
        p1 = SoftReciprocalPolicy(sigma)
        traj = rollout_ipd_with_mixture(p1, mixture, horizon, nu, rng)

        viols.append(composite_violation(
            traj, horizon,
            mode=viol_mode,
            spiral_len=spiral_len,
            exploit_len=exploit_len,
            safe_threshold=safe_threshold,
        ))
        exploit_rates.append(exploitation_spiral(traj, horizon, min_len=exploit_len))
        coops.append(cooperation_rate(traj, horizon))
        falses.append(false_retaliation_rate(traj, horizon))
        returns.append(mean_return(traj, horizon))

    return {
        "violation_rate": float(np.mean(viols)),
        "exploitation_rate": float(np.mean(exploit_rates)),
        "cooperation_rate": float(np.mean(coops)),
        "false_retaliation_rate": float(np.mean(falses)),
        "mean_return": float(np.mean(returns)),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Sweep (nu, sigma, seed, horizon) on noisy IPD — v4."
    )
    # Grid
    ap.add_argument(
        "--noise-levels", nargs="+", type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        metavar="NU",
        help="Noise (misperception) levels to sweep.",
    )
    ap.add_argument("--sigma-min", type=float, default=0.0)
    ap.add_argument("--sigma-max", type=float, default=0.8)
    ap.add_argument("--sigma-n", type=int, default=41,
                    help="Number of sigma grid points (inclusive).")
    # Phase C: horizon list
    ap.add_argument(
        "--horizons", nargs="+", type=int, default=[200],
        metavar="H",
        help="Episode lengths H to sweep (Phase C). Default: [200].",
    )
    # Episode / seed
    ap.add_argument("--episodes", type=int, default=2000,
                    help="Episodes per (nu, sigma, seed, horizon) combo.")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of random seeds.")
    # Violation settings
    ap.add_argument(
        "--spiral-len", type=int, default=5,
        help="Min consecutive (D,D) steps for mutual_defection_spiral.",
    )
    ap.add_argument(
        "--exploit-len", type=int, default=5,
        help="Min consecutive (C,D) steps for exploitation_spiral.",
    )
    ap.add_argument(
        "--safe-threshold", type=float, default=2.0,
        help="Mean payoff below this fires payoff_loss_violation.",
    )
    ap.add_argument(
        "--violation-mode",
        default="spiral_or_exploit",
        choices=["spiral_only", "exploit_only", "spiral_or_exploit",
                 "spiral_or_payoff_loss"],
        help="Which composite violation indicator to use.",
    )
    # Opponent
    ap.add_argument(
        "--opponent-mixture",
        default="reciprocal:1.0",
        help=(
            'Mixture spec, e.g. "reciprocal:0.8,always_defect:0.2". '
            '"reciprocal:1.0" → pure self-play baseline.'
        ),
    )
    # Misc
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="Violation rate threshold α (stored in CSV only).")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.sigma_n)
    rows = []
    total = (
        len(args.horizons)
        * len(args.noise_levels)
        * len(sigma_grid)
        * args.seeds
    )

    with tqdm(total=total, desc="Exp1 sweep") as pbar:
        for horizon in args.horizons:
            for nu in args.noise_levels:
                for sigma in sigma_grid:
                    for seed in range(args.seeds):
                        stats = run_one(
                            nu=nu,
                            sigma=float(sigma),
                            opp_spec=args.opponent_mixture,
                            viol_mode=args.violation_mode,
                            spiral_len=args.spiral_len,
                            exploit_len=args.exploit_len,
                            safe_threshold=args.safe_threshold,
                            episodes=args.episodes,
                            horizon=horizon,
                            seed=seed,
                        )
                        rows.append({
                            "game": "ipd",
                            "nu": nu,
                            "sigma": float(sigma),
                            "seed": seed,
                            "episodes": args.episodes,
                            "horizon": horizon,
                            "spiral_len": args.spiral_len,
                            "exploit_len": args.exploit_len,
                            "safe_threshold": args.safe_threshold,
                            "violation_mode": args.violation_mode,
                            "opponent_mixture": args.opponent_mixture,
                            "alpha": args.alpha,
                            **stats,
                        })
                        pbar.update(1)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    write_run_info(
        args.out.parent,
        json.dumps(vars(args), default=str, indent=2),
    )

    expected_rows = (
        len(args.horizons)
        * len(args.noise_levels)
        * args.sigma_n
        * args.seeds
    )
    print(f"Wrote {len(df)} rows (expected {expected_rows}) to {args.out}")


if __name__ == "__main__":
    main()
