"""Exp 2 Phase E: Ablation study comparing CPB components.

Ablation conditions:
  full        : cpb_full (match + cap)
  match_only  : CPB matching only, no cap
  cap_only    : CPB cap only, no match calibration
  no_cpb      : dreamer_default (no CPB)

Usage::

    python -m experiments.exp2_dreamer.run_ablation \\
        --env cheetah-run \\
        --steps 200000 \\
        --nu-values 0.05 0.10 0.20 \\
        --seed 0 \\
        --out results/runs/exp2_ablation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_ABLATION_METHODS = ["cpb_full", "cpb_match_only", "cpb_cap_only", "dreamer_default"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 2 ablation study.")
    ap.add_argument("--env", default="cheetah-run")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--nu-values", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    ap.add_argument("--config", type=str,
                    default=str(Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml"))  # noqa: E501
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_ablation"))
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic data (no dm_control required).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[synthetic] Generating ablation data without dm_control ...")
        rng = np.random.default_rng(args.seed)
        all_results = []
        for method in _ABLATION_METHODS:
            for nu in args.nu_values:
                # Synthetic: cpb_full is best, others progressively worse
                base_return = 100.0
                if method == "cpb_full":
                    mean_r = base_return * (1 - nu * 0.5) + rng.normal(0, 3)
                elif method == "cpb_match_only":
                    mean_r = base_return * (1 - nu * 0.8) + rng.normal(0, 3)
                elif method == "cpb_cap_only":
                    mean_r = base_return * (1 - nu * 1.0) + rng.normal(0, 4)
                else:
                    mean_r = base_return * (1 - nu * 1.5) + rng.normal(0, 5)
                all_results.append({
                    "method": method,
                    "nu": nu,
                    "mean_return": float(mean_r),
                    "std_return": float(rng.uniform(3, 8)),
                    "synthetic": True,
                })
    else:
        suite, task = args.env.split("-", 1)
        from experiments.exp2_dreamer.run_main import _build_policy, _run
        from nms.envs.stochastic_dm_control import make_stochastic_dmc_env

        all_results = []
        for method in _ABLATION_METHODS:
            for nu in args.nu_values:
                print(f"\n--- method={method}  ν={nu:.3f} ---")
                env = make_stochastic_dmc_env(suite, task, sigma_obs=nu, seed=args.seed)
                policy = _build_policy(method, env, seed=args.seed,
                                       config_path=args.config, device=args.device)
                summary = _run(policy, env, total_steps=args.steps,
                               eval_every=args.steps // 10, out_dir=out_dir)
                env.close()
                all_results.append({
                    "method": method,
                    "nu": nu,
                    **summary,
                })

    (out_dir / "ablation_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nAblation results saved → {out_dir / 'ablation_results.json'}")

    # Print table
    print(f"\n{'Method':<20}  {'ν':>6}  {'Return':>10}")
    print("-" * 42)
    for r in all_results:
        print(f"{r['method']:<20}  {r['nu']:>6.3f}  {r['mean_return']:>10.2f}")


if __name__ == "__main__":
    main()
