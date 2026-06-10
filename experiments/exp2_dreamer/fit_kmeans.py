"""Exp 2 Phase 1+2: Fit KMeansActionDiscretizer on TRAINED POLICY actions.

Unlike fit_interface.py (which uses random actions), this script loads a
trained DreamerWrapper checkpoint and collects actions from N full episodes
using the agent's deterministic policy.  The resulting k-means centroids
better reflect the action subspace explored during deployment.

Usage::

    python -m experiments.exp2_dreamer.fit_kmeans \\
        --dreamer-dir results/runs/exp2/_smoke/hopper_dv3_K1/member_0 \\
        --env hopper-hop \\
        --k 8 \\
        --n-rollouts 50 \\
        --out results/interfaces/hopper_hop_k8.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _collect_policy_actions(
    dreamer: Any,
    env: Any,
    n_rollouts: int,
    seed: int,
    max_steps_per_ep: int = 1_000,
) -> np.ndarray:
    """Run the trained policy for *n_rollouts* episodes; return stacked actions.

    Parameters
    ----------
    dreamer       : Loaded DreamerWrapper (load_from_dir already called).
    env           : gymnasium-compatible environment.
    n_rollouts    : Number of episodes to collect.
    seed          : Base random seed; episode i uses seed + i.
    max_steps_per_ep : Safety cap on episode length.

    Returns
    -------
    actions : np.ndarray of shape (total_steps, act_dim).
    """
    all_actions: list[np.ndarray] = []
    for ep in range(n_rollouts):
        obs, _ = env.reset(seed=seed + ep)
        dreamer.reset_state()  # fresh RSSM state per episode
        done = False
        step = 0
        while not done and step < max_steps_per_ep:
            action = dreamer.sample_action(obs)
            obs, _r, term, trunc, _ = env.step(action)
            all_actions.append(np.asarray(action, dtype=np.float64).flatten())
            done = term or trunc
            step += 1
        if (ep + 1) % 10 == 0:
            print(f"  episode {ep+1}/{n_rollouts}  ({len(all_actions)} actions total)")
    return np.stack(all_actions)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit k-means on trained-policy actions."
    )
    ap.add_argument("--dreamer-dir", type=Path, required=True,
                    help="Directory containing latest.pt checkpoint.")
    ap.add_argument("--env", default="cheetah-run",
                    help="Environment as 'suite-task'.")
    ap.add_argument("--k", type=int, default=8,
                    help="Number of k-means clusters.")
    ap.add_argument("--n-rollouts", type=int, default=50,
                    help="Number of full episodes to collect for fitting.")
    ap.add_argument("--sigma-obs", type=float, default=0.0)
    ap.add_argument("--sigma-reward", type=float, default=0.0)
    ap.add_argument("--sigma-dynamics", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/interfaces/env_k8.npz"))
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env
    from nms.interfaces.kmeans_interface import fit_and_save
    from nms.wm.dreamer_wrapper import DreamerWrapper

    config_path = str(
        Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml"
    )

    print(f"\n=== fit_kmeans: {args.env} k={args.k} n_rollouts={args.n_rollouts} ===")

    # Build env for DreamerWrapper construction
    env = make_stochastic_dmc_env(
        suite, task,
        sigma_obs=args.sigma_obs,
        sigma_reward=args.sigma_reward,
        sigma_dynamics=args.sigma_dynamics,
        seed=args.seed,
    )

    dreamer = DreamerWrapper(
        config_path=config_path, env=env,
        device=args.device, seed=args.seed,
    )

    # Locate and load checkpoint
    dreamer_dir = Path(args.dreamer_dir)
    _candidates = [dreamer_dir, dreamer_dir / "member_0"]
    ckpt_dir = next((p for p in _candidates if (p / "latest.pt").exists()), None)
    if ckpt_dir is None:
        raise FileNotFoundError(
            f"No latest.pt in: {[str(p) for p in _candidates]}"
        )
    print(f"Loading checkpoint from {ckpt_dir}/latest.pt …")
    dreamer.load_from_dir(str(ckpt_dir))
    print(f"Agent backend: {dreamer._backend}")

    # Collect policy actions
    print(f"\nCollecting {args.n_rollouts} policy episodes …")
    actions = _collect_policy_actions(
        dreamer, env, n_rollouts=args.n_rollouts, seed=args.seed
    )
    print(f"Collected {len(actions)} actions with dim {actions.shape[1]}")

    # Fit k-means
    print(f"\nFitting k-means (k={args.k}) …")
    km = fit_and_save(actions, out_path, n_clusters=args.k, random_state=args.seed)
    print(f"Saved {args.k} centroids → {out_path}")
    print(f"  centroid shape : {km._centroids.shape}")
    print(f"  action dim     : {km._centroids.shape[1]}")

    env.close()


if __name__ == "__main__":
    main()
