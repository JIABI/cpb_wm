"""Exp 2 Phase D: OOD evaluation under physics parameter shifts.

Usage::

    python -m experiments.exp2_dreamer.run_ood \\
        --env cheetah-run \\
        --method cpb_full \\
        --mass-mults 0.5 1.0 2.0 3.0 \\
        --friction-mults 0.5 1.0 2.0 \\
        --n-episodes 10 \\
        --seed 0 \\
        --out results/runs/exp2_ood/cpb_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 2 OOD evaluation.")
    ap.add_argument("--env", default="cheetah-run")
    ap.add_argument("--method", default="cpb_full")
    ap.add_argument("--mass-mults", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--friction-mults", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    ap.add_argument("--config", type=str,
                    default=str(Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml"))  # noqa: E501
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_ood/run"))
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic data (no dm_control required).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[synthetic] Generating OOD data without dm_control ...")
        rng = np.random.default_rng(args.seed)
        results = []
        nominal_return = 100.0
        for m in args.mass_mults:
            # Performance degrades as mass deviates from 1.0
            deviation = abs(m - 1.0)
            mean_ret = nominal_return * np.exp(-deviation * 1.5) + rng.normal(0, 2)
            results.append({
                "shift": f"mass_{m:.2f}",
                "mass_mult": m,
                "friction_mult": 1.0,
                "mean_return": float(mean_ret),
                "std_return": float(rng.uniform(2, 8)),
            })
        output = {
            "method": args.method,
            "env": args.env,
            "nominal_return": nominal_return,
            "results": results,
            "synthetic": True,
        }
    else:
        suite, task = args.env.split("-", 1)

        from nms.envs.stochastic_dm_control import make_stochastic_dmc_env
        from nms.eval.ood_evaluator import OODEvaluator
        from nms.wm.dreamer_wrapper import DreamerWrapper

        env = make_stochastic_dmc_env(suite, task, seed=args.seed)
        dreamer = DreamerWrapper(config_path=args.config, env=env, device=args.device, seed=args.seed)  # noqa: E501

        evaluator = OODEvaluator(
            policy=dreamer,
            base_env=env,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        nominal = evaluator.evaluate_nominal()
        results = evaluator.sweep_mass(args.mass_mults)
        output = {
            "method": args.method,
            "env": args.env,
            "nominal": nominal,
            "results": results,
        }
        env.close()

    (out_dir / "ood_results.json").write_text(json.dumps(output, indent=2))
    print(f"OOD evaluation saved → {out_dir / 'ood_results.json'}")


if __name__ == "__main__":
    main()
