"""Exp 2 Phase B: Fit KMeansActionDiscretizer on environment actions.

Collects random exploration actions from the environment and fits k-means.
Saves the resulting discretizer to disk for use in train_ensemble.py and
train_baseline.py.

Usage::

    python -m experiments.exp2_dreamer.fit_interface \\
        --env cheetah-run \\
        --n-clusters 20 \\
        --fit-steps 10000 \\
        --seed 0 \\
        --out results/runs/exp2_cheetah_run/seed0/kmeans.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nms.interfaces.kmeans_interface import fit_and_save


def _collect_actions(env: object, fit_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions: list[np.ndarray] = []
    obs, _ = env.reset()  # type: ignore[attr-defined]
    for _ in range(fit_steps):
        action = rng.uniform(
            env.action_space.low,  # type: ignore[attr-defined]
            env.action_space.high,  # type: ignore[attr-defined]
        )
        obs, _, terminated, truncated, _ = env.step(action)  # type: ignore[attr-defined]
        actions.append(action.flatten().astype(np.float64))
        if terminated or truncated:
            obs, _ = env.reset()  # type: ignore[attr-defined]
    return np.stack(actions)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit k-means action interface.")
    ap.add_argument("--env", default="cheetah-run",
                    help="Environment as 'suite-task' (e.g. cheetah-run).")
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--fit-steps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-obs", type=float, default=0.0)
    ap.add_argument("--sigma-reward", type=float, default=0.0)
    ap.add_argument("--sigma-dynamics", type=float, default=0.0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_interface/kmeans.npz"))
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)

    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env

    env = make_stochastic_dmc_env(
        suite, task,
        sigma_obs=args.sigma_obs,
        sigma_reward=args.sigma_reward,
        sigma_dynamics=args.sigma_dynamics,
        seed=args.seed,
    )

    print(f"Collecting {args.fit_steps} actions from {suite}/{task} ...")
    actions = _collect_actions(env, args.fit_steps, args.seed)
    env.close()

    print(f"Fitting k-means (k={args.n_clusters}) ...")
    km = fit_and_save(actions, args.out, n_clusters=args.n_clusters, random_state=args.seed)
    print(f"Saved {args.n_clusters} centroids to {args.out}")
    print(f"Action dim: {km._centroids.shape[1]}")


if __name__ == "__main__":
    main()
