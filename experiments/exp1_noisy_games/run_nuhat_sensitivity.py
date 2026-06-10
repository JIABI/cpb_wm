"""Exp 1 Phase C: ν̂ sensitivity analysis (Proposition 1).

For each (ν_true, bias) pair:
  1. Compute ν̂ = clip(ν_true * (1 + bias), 0, 1).
  2. Look up the certified bandwidth B_{H,α}(ν̂) from the pre-run sweep CSV
     (the same CSV produced by run_bandwidth.py at H=200 with mixture).
  3. Use the linear response map σ^CPB = clip(m(ν̂), σ_min(ν̂), σ_max(ν̂)).
  4. Roll out the agent with σ=σ^CPB against the true ν_true environment
     and record the realized violation rate and mean return.

Output: nu_hat_sensitivity.csv with columns
    nu_true, nu_hat, bias, sigma_cpb,
    realized_violation_rate, realized_mean_return,
    sigma_min_nuhat, sigma_max_nuhat, is_admissible_at_true

Usage:
    python -m experiments.exp1_noisy_games.run_nuhat_sensitivity \
        --csv results/runs/exp1_noisy_games/bandwidth_ipd.csv \
        --out results/runs/exp1_noisy_games/nu_hat_sensitivity.csv \
        --horizon 200 --alpha 0.1 \
        --nu-true 0.05 0.10 0.15 0.20 0.30 \
        --biases -0.5 -0.2 0.0 0.2 0.5 \
        --episodes 2000 --seeds 10 \
        --response-c 1.0 \
        --opponent-mixture "reciprocal:0.8,always_defect:0.2" \
        --violation-mode spiral_or_exploit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nms.core.envelope import Bandwidth, estimate_bandwidth
from nms.core.projection import BandwidthEmptyError, project_onto_bandwidth
from nms.envs.noisy_games import rollout_ipd_with_mixture
from nms.metrics.game_violation import composite_violation, mean_return
from nms.policies.opponents import parse_mixture_spec
from nms.policies.reciprocal import SoftReciprocalPolicy
from nms.utils.logging import write_run_info
from nms.utils.seeds import set_global_seed


def _build_bandwidth_lookup(
    df: pd.DataFrame,
    horizon: int,
    alpha: float,
) -> dict[float, Bandwidth]:
    """Build a {nu: Bandwidth} dict from the sweep CSV at a given horizon."""
    if "horizon" in df.columns:
        df = df[df["horizon"] == horizon]

    lookup: dict[float, Bandwidth] = {}
    agg = (
        df.groupby(["nu", "sigma"])
        .agg(violation_rate=("violation_rate", "mean"))
        .reset_index()
    )
    for nu, sub in agg.groupby("nu"):
        sub = sub.sort_values("sigma")
        sigma_arr = sub["sigma"].to_numpy()
        risk_arr = sub["violation_rate"].to_numpy()
        risk_map = dict(zip(sigma_arr.tolist(), risk_arr.tolist(), strict=False))

        def risk_fn(s: float, lk: dict[float, float] = risk_map) -> float:
            return lk[min(lk, key=lambda k: abs(k - s))]

        bw = estimate_bandwidth(sigma_arr, risk_fn, nu=float(nu),
                                horizon=horizon, alpha=alpha)
        lookup[float(nu)] = bw
    return lookup


def _nearest_nu(target: float, available: list[float]) -> float:
    """Return the nearest ν in available to target."""
    return min(available, key=lambda n: abs(n - target))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Proposition 1 ν̂ sensitivity analysis."
    )
    ap.add_argument("--csv", type=Path, required=True,
                    help="Pre-run bandwidth sweep CSV (from run_bandwidth.py).")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--nu-true", nargs="+", type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
        metavar="NU",
    )
    ap.add_argument(
        "--biases", nargs="+", type=float,
        default=[-0.5, -0.2, 0.0, 0.2, 0.5],
        metavar="BIAS",
        help="Relative bias: ν̂ = ν_true * (1 + bias), clipped to [0, 1].",
    )
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--response-c", type=float, default=1.0,
        help="Slope c for the linear response map m(ν̂) = c·ν̂.",
    )
    ap.add_argument(
        "--opponent-mixture", default="reciprocal:0.8,always_defect:0.2",
        help="Opponent mixture spec for rollout evaluation.",
    )
    ap.add_argument(
        "--violation-mode", default="spiral_or_exploit",
        choices=["spiral_only", "exploit_only",
                 "spiral_or_exploit", "spiral_or_payoff_loss"],
    )
    ap.add_argument("--spiral-len", type=int, default=15,
                    help="Min consecutive (D,D) steps for spiral check (default 15).")
    ap.add_argument("--exploit-len", type=int, default=15,
                    help="Min consecutive (C,D) steps for exploit check (default 15).")
    ap.add_argument("--safe-threshold", type=float, default=2.0)
    args = ap.parse_args()

    df_sweep = pd.read_csv(args.csv)
    bw_lookup = _build_bandwidth_lookup(df_sweep, args.horizon, args.alpha)
    available_nus = sorted(bw_lookup.keys())

    rows = []
    total = len(args.nu_true) * len(args.biases) * args.seeds
    with tqdm(total=total, desc="ν̂ sensitivity") as pbar:
        for nu_true in args.nu_true:
            for bias in args.biases:
                nu_hat = float(np.clip(nu_true * (1.0 + bias), 0.0, 1.0))
                # Nearest available ν̂ in the sweep
                nu_hat_nearest = _nearest_nu(nu_hat, available_nus)
                bw = bw_lookup[nu_hat_nearest]

                # σ^CPB = project(m(ν̂), B_{H,α}(ν̂))
                m_nu_hat = args.response_c * nu_hat
                try:
                    sigma_cpb = project_onto_bandwidth(m_nu_hat, bw)
                except BandwidthEmptyError:
                    sigma_cpb = float("nan")

                sigma_min_nuhat = bw.sigma_min
                sigma_max_nuhat = bw.sigma_max

                # Check if σ^CPB is admissible at the *true* ν
                nu_true_nearest = _nearest_nu(nu_true, available_nus)
                bw_true = bw_lookup[nu_true_nearest]
                is_admissible_at_true = (
                    bw_true.contains(sigma_cpb)
                    if np.isfinite(sigma_cpb)
                    else False
                )

                # Evaluate: roll out with σ=σ^CPB at true ν_true
                for seed in range(args.seeds):
                    if not np.isfinite(sigma_cpb):
                        rows.append({
                            "nu_true": nu_true,
                            "nu_hat": nu_hat,
                            "nu_hat_nearest": nu_hat_nearest,
                            "bias": bias,
                            "sigma_cpb": sigma_cpb,
                            "sigma_min_nuhat": sigma_min_nuhat,
                            "sigma_max_nuhat": sigma_max_nuhat,
                            "is_admissible_at_true": is_admissible_at_true,
                            "realized_violation_rate": float("nan"),
                            "realized_mean_return": float("nan"),
                            "seed": seed,
                        })
                        pbar.update(1)
                        continue

                    rng = set_global_seed(seed)
                    viols, rets = [], []
                    p1 = SoftReciprocalPolicy(sigma=sigma_cpb)
                    mixture = parse_mixture_spec(
                        args.opponent_mixture, agent_sigma=sigma_cpb
                    )
                    for _ in range(args.episodes):
                        traj = rollout_ipd_with_mixture(
                            p1, mixture, args.horizon, nu_true, rng
                        )
                        viols.append(composite_violation(
                            traj, args.horizon,
                            mode=args.violation_mode,
                            spiral_len=args.spiral_len,
                            exploit_len=args.exploit_len,
                            safe_threshold=args.safe_threshold,
                        ))
                        rets.append(mean_return(traj, args.horizon))

                    rows.append({
                        "nu_true": nu_true,
                        "nu_hat": nu_hat,
                        "nu_hat_nearest": nu_hat_nearest,
                        "bias": bias,
                        "sigma_cpb": sigma_cpb,
                        "sigma_min_nuhat": sigma_min_nuhat,
                        "sigma_max_nuhat": sigma_max_nuhat,
                        "is_admissible_at_true": is_admissible_at_true,
                        "realized_violation_rate": float(np.mean(viols)),
                        "realized_mean_return": float(np.mean(rets)),
                        "seed": seed,
                    })
                    pbar.update(1)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    write_run_info(
        args.out.parent,
        json.dumps(vars(args), default=str, indent=2),
    )

    print(f"Wrote {len(out_df)} rows to {args.out}")

    # Quick summary: mean violation rate by bias
    summary = (
        out_df.groupby("bias")
        .agg(
            mean_viol=("realized_violation_rate", "mean"),
            mean_ret=("realized_mean_return", "mean"),
        )
        .reset_index()
    )
    print("\n=== ν̂ Sensitivity Summary (by bias) ===")
    print(summary.to_string(index=False))
    inadmissible = out_df[~out_df["is_admissible_at_true"]]
    print(f"\nRows where σ^CPB is NOT admissible at true ν: "
          f"{len(inadmissible)}/{len(out_df)}")


if __name__ == "__main__":
    main()
