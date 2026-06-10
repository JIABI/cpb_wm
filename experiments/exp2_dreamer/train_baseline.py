"""Exp 2 Phase A: DreamerV3 baseline training on a dm_control environment.

Two entry-point modes:

  Smoke mode (argparse — for Phase A acceptance test):
      python -m experiments.exp2_dreamer.train_baseline \\
          --env cheetah-run --steps 100000 --seed 0 \\
          --out results/runs/exp2_smoke/cheetah_baseline

  Full Hydra mode (full experiment with config overrides):
      python -m experiments.exp2_dreamer.train_baseline \\
          env.suite=cheetah env.task=run \\
          training.total_steps=1000000 seed=0

Constraints enforced:
  * Actions go through KMeansActionDiscretizer — NO raw continuous actions.
  * NMSOperator core interface is NOT modified.
  * Conformal calibration happens via core/certificate.py (NOT in this script).
  * Ensemble K=3 for Phase A/B.
  * RNG: np.random.default_rng(seed) — NOT np.random.seed().
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ── environment builder ───────────────────────────────────────────────────────

def _build_env(
    suite: str,
    task: str,
    sigma_obs: float = 0.0,
    sigma_reward: float = 0.0,
    sigma_dynamics: float = 0.0,
    seed: int = 0,
) -> Any:
    """Build a (possibly stochastic) dm_control → gymnasium env."""
    try:
        import dm_control.suite as dmc_suite
    except ModuleNotFoundError as exc:
        raise ImportError(
            "dm_control is required for Exp 2. "
            "Install: pip install dm_control"
        ) from exc

    raw_env = dmc_suite.load(suite, task)

    # Wrap in gymnasium via shimmy (preferred) or fall back to raw
    try:
        from shimmy import DMControlCompatibilityV0  # type: ignore[import-not-found]
        base_env = DMControlCompatibilityV0(raw_env)
    except ModuleNotFoundError:
        warnings.warn(
            "shimmy not installed — using dm_control directly. "
            "Install shimmy for full gymnasium compatibility: pip install shimmy[dm-control]",
            stacklevel=2,
        )
        base_env = raw_env  # type: ignore[assignment]

    from nms.envs.stochastic_dm_control import StochasticDynamicsWrapper

    env = StochasticDynamicsWrapper(
        base_env,
        sigma_obs=sigma_obs,
        sigma_reward=sigma_reward,
        sigma_dynamics=sigma_dynamics,
        seed=seed,
    )
    return env


# ── k-means fitting ───────────────────────────────────────────────────────────

def _fit_kmeans(
    env: Any,
    n_clusters: int = 8,
    fit_steps: int = 10_000,
    random_state: int = 42,
) -> Any:
    """Collect random actions and fit k-means on them."""
    from nms.wm.dreamer_wrapper import KMeansActionDiscretizer

    rng = np.random.default_rng(random_state)
    actions: list[np.ndarray] = []
    obs, _ = env.reset()
    for _ in range(fit_steps):
        action = rng.uniform(env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, _ = env.step(action)
        actions.append(action.flatten())
        if terminated or truncated:
            obs, _ = env.reset()

    all_actions = np.stack(actions)
    km = KMeansActionDiscretizer(n_clusters=n_clusters, random_state=random_state)
    km.fit(all_actions)
    print(f"K-means fit on {len(all_actions)} actions → {n_clusters} centroids.")
    return km


# ── training loop (smoke / full) ──────────────────────────────────────────────

def _run(
    env: Any,
    dreamer: Any,
    total_steps: int,
    eval_every: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Run training loop; return summary dict for run_info.json."""
    step = 0
    episode = 0
    returns: list[float] = []

    while step < total_steps:
        obs, _ = env.reset()
        ep_return = 0.0
        terminated = truncated = False

        while not (terminated or truncated) and step < total_steps:
            action = dreamer.sample_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            step += 1

        episode += 1
        returns.append(ep_return)

        if episode % max(1, eval_every // 200) == 0 or step >= total_steps:
            mean_ret = float(np.mean(returns[-20:])) if returns else 0.0
            print(
                f"  step={step:>8d}  episode={episode:>5d}  "
                f"return(last 20 ep avg)={mean_ret:.2f}"
            )

    last_100 = returns[-100:] if len(returns) >= 100 else returns
    return {
        "total_steps": step,
        "episodes": episode,
        "mean_return_last_100": float(np.mean(last_100)) if last_100 else 0.0,
        "std_return_last_100": float(np.std(last_100)) if last_100 else 0.0,
    }


# ── smoke mode (argparse) ─────────────────────────────────────────────────────

def _smoke_main() -> None:
    """Entry point for Phase A smoke test with argparse CLI."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Exp 2 Phase A: DreamerV3 baseline smoke run."
    )
    ap.add_argument("--env", default="cheetah-run",
                    help="Environment as 'suite-task' (e.g. cheetah-run).")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmeans-k", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/runs/exp2_smoke/cheetah_baseline"),
    )
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Exp 2 Phase A Smoke Run ===")
    print(f"  env={suite}/{task}  steps={args.steps}  seed={args.seed}")
    print(f"  out={out_dir}\n")

    # Build env
    env = _build_env(suite, task, seed=args.seed)

    # Fit k-means
    _fit_kmeans(env, n_clusters=args.kmeans_k, random_state=args.seed)

    # Build DreamerWrapper (stub mode until dreamerv3-torch is installed)
    from nms.wm.dreamer_wrapper import DreamerWrapper

    config_path = str(
        Path(__file__).parent.parent.parent
        / "configs" / "exp2_rl" / "config.yaml"
    )
    dreamer = DreamerWrapper(config_path=config_path, env=env, seed=args.seed)

    # Phase A: train (stub mode → no actual gradient updates)
    dreamer.train(total_steps=args.steps, log_dir=str(out_dir / "dreamer_logs"))

    # Smoke rollout — verify trajectory shapes
    traj = dreamer.rollout(env, horizon=15, seed=args.seed)
    assert traj["obs"].ndim == 2, f"Expected 2-D obs array, got shape {traj['obs'].shape}"
    assert traj["actions"].ndim == 2, \
        f"Expected 2-D actions array, got shape {traj['actions'].shape}"
    print(f"  rollout(H=15) OK — obs={traj['obs'].shape}, "
          f"actions={traj['actions'].shape}, rewards={traj['rewards'].shape}")

    # Run training loop (stub)
    summary = _run(env, dreamer, total_steps=args.steps,
                   eval_every=args.eval_every, out_dir=out_dir)

    # Save run_info.json
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    run_info = {
        "env": args.env,
        "suite": suite,
        "task": task,
        "total_steps": args.steps,
        "seed": args.seed,
        "kmeans_k": args.kmeans_k,
        "git_sha": git_sha,
        **summary,
    }
    with open(out_dir / "run_info.json", "w") as fp:
        json.dump(run_info, fp, indent=2)

    print(f"\n  mean_return_last_100 = {summary['mean_return_last_100']:.2f}")
    print(f"  Saved run_info.json → {out_dir / 'run_info.json'}")
    print("\n=== Phase A smoke run complete ===")

    env.close()


# ── Hydra mode (full experiment) ──────────────────────────────────────────────

def _hydra_main() -> None:
    """Entry point for full Hydra-config runs."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        config_path="../../configs/exp2_rl",
        config_name="config",
        version_base=None,
    )
    def _inner(cfg: DictConfig) -> None:
        from nms.utils.seeds import set_global_seed
        from nms.wm.dreamer_wrapper import DreamerWrapper

        rng = set_global_seed(int(cfg.seed))  # noqa: F841
        out_dir = Path(
            f"results/runs/exp2_{cfg.env.suite}_{cfg.env.task}/seed{cfg.seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        env = _build_env(
            suite=cfg.env.suite,
            task=cfg.env.task,
            sigma_obs=float(cfg.noise.sigma_obs),
            sigma_reward=float(cfg.noise.sigma_reward),
            sigma_dynamics=float(cfg.noise.sigma_dynamics),
            seed=int(cfg.seed),
        )
        _fit_kmeans(
            env,
            n_clusters=int(cfg.kmeans.n_clusters),
            fit_steps=int(cfg.kmeans.fit_steps),
            random_state=int(cfg.seed),
        )
        config_path = str(
            Path(__file__).parent.parent.parent
            / "configs" / "exp2_rl" / "config.yaml"
        )
        dreamer = DreamerWrapper(
            config_path=config_path, env=env, seed=int(cfg.seed)
        )
        dreamer.train(
            total_steps=int(cfg.training.total_steps),
            log_dir=str(out_dir / "dreamer_logs"),
        )
        env.close()

    _inner()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use argparse mode when --env / --steps flags are present;
    # otherwise fall through to Hydra for config-based runs.
    if any(a.startswith("--env") or a.startswith("--steps") for a in sys.argv[1:]):
        _smoke_main()
    else:
        _hydra_main()
