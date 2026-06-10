"""Exp 2 Phase B: Train an EnsembleRSSM of K world models.

Trains K=3 (Phase A/B) or K=5 (Phase C) independent DreamerWrapper members
on the same environment.  Each member is saved to its own subdirectory.

Usage::

    python -m experiments.exp2_dreamer.train_ensemble \\
        --env cheetah-run \\
        --k 3 \\
        --steps 500000 \\
        --seed 0 \\
        --out results/runs/exp2_cheetah_run/seed0/ensemble
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from nms.wm.ensemble_rssm import EnsembleRSSM


def main() -> None:
    ap = argparse.ArgumentParser(description="Train EnsembleRSSM for Exp 2.")
    ap.add_argument("--env", default="cheetah-run",
                    help="Environment as 'suite-task'.")
    ap.add_argument("--k", type=int, default=3, help="Ensemble size K.")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-obs", type=float, default=0.0)
    ap.add_argument("--sigma-reward", type=float, default=0.0)
    ap.add_argument("--sigma-dynamics", type=float, default=0.0)
    ap.add_argument("--config", type=str,
                    default="configs/exp2_rl/config.yaml")
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_ensemble/ensemble"))
    ap.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env

    def env_factory() -> object:
        return make_stochastic_dmc_env(
            suite, task,
            sigma_obs=args.sigma_obs,
            sigma_reward=args.sigma_reward,
            sigma_dynamics=args.sigma_dynamics,
            seed=args.seed,
        )

    # Infer observation/action spaces from one env instance
    tmp_env = env_factory()
    obs_space = tmp_env.observation_space  # type: ignore[attr-defined]
    act_space = tmp_env.action_space       # type: ignore[attr-defined]
    tmp_env.close()                        # type: ignore[attr-defined]

    print("\n=== EnsembleRSSM Training ===")
    print(f"  env={suite}/{task}  K={args.k}  steps={args.steps}  seed={args.seed}")
    print(f"  out={out_dir}\n")

    ensemble = EnsembleRSSM(
        obs_space=obs_space,
        act_space=act_space,
        k=args.k,
        device=args.device,
        seed=args.seed,
        config_path=args.config,
    )
    ensemble.train_all(env_factory, total_steps=args.steps, log_dir=str(out_dir))
    ensemble.save_all(out_dir)

    # Save run info
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    run_info = {
        "env": args.env,
        "k": args.k,
        "total_steps": args.steps,
        "seed": args.seed,
        "git_sha": git_sha,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"\nEnsemble saved to {out_dir}")


if __name__ == "__main__":
    main()
