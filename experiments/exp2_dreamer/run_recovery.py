"""Exp 2 Phase E: Recovery evaluation after abrupt noise increase.

Usage::

    python -m experiments.exp2_dreamer.run_recovery \\
        --env cheetah-run \\
        --method cpb_full \\
        --sigma-ood 0.3 \\
        --switch-step 100 \\
        --horizon 500 \\
        --n-episodes 10 \\
        --seed 0 \\
        --out results/runs/exp2_recovery/cpb_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 2 recovery evaluation.")
    ap.add_argument("--env", default="cheetah-run")
    ap.add_argument("--method", default="cpb_full")
    ap.add_argument("--sigma-ood", type=float, default=0.3)
    ap.add_argument("--switch-step", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    ap.add_argument("--config", type=str,
                    default=str(Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml"))  # noqa: E501
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_recovery/run"))
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic data (no dm_control required).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[synthetic] Generating recovery data without dm_control ...")
        rng = np.random.default_rng(args.seed)
        switch = args.switch_step
        n = args.horizon
        # Simulate: returns decay after switch, recover slowly
        mean_rewards = np.concatenate([
            rng.normal(1.0, 0.1, switch),
            rng.normal(0.3, 0.2, n - switch) + np.linspace(0, 0.5, n - switch),
        ])
        result = {
            "method": args.method,
            "sigma_ood": args.sigma_ood,
            "switch_step": switch,
            "mean_pre_switch_return": float(mean_rewards[:switch].mean()),
            "mean_post_switch_return": float(mean_rewards[switch:].mean()),
            "recovery_step": int(np.random.default_rng(args.seed).integers(50, 200)),
            "area_under_recovery": float(np.maximum(1.0 - mean_rewards[switch:], 0).sum()),
            "mean_rewards": mean_rewards.tolist(),
            "synthetic": True,
        }
    else:
        suite, task = args.env.split("-", 1)

        from nms.envs.stochastic_dm_control import make_stochastic_dmc_env
        from nms.eval.recovery_evaluator import RecoveryEvaluator
        from nms.wm.dreamer_wrapper import DreamerWrapper

        def env_factory(sigma: float) -> object:
            return make_stochastic_dmc_env(suite, task, sigma_obs=sigma, seed=args.seed)

        env = env_factory(0.0)
        dreamer = DreamerWrapper(config_path=args.config, env=env, device=args.device, seed=args.seed)  # noqa: E501

        evaluator = RecoveryEvaluator(
            policy=dreamer,
            env_factory=env_factory,
            horizon=args.horizon,
            switch_step=args.switch_step,
            sigma_ood=args.sigma_ood,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        raw = evaluator.evaluate()
        result = {"method": args.method, **raw}
        env.close()

    (out_dir / "recovery_results.json").write_text(json.dumps(result, indent=2))
    print("\nRecovery evaluation complete:")
    print(f"  pre-switch return : {result['mean_pre_switch_return']:.3f}")
    print(f"  post-switch return: {result['mean_post_switch_return']:.3f}")
    print(f"  recovery_step     : {result.get('recovery_step', 'N/A')}")
    print(f"  Saved → {out_dir / 'recovery_results.json'}")


if __name__ == "__main__":
    main()
