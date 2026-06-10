"""Exp 2 Phase C: Main training driver for all 12 methods.

Supports both argparse (--dry-run for CI) and Hydra (full experiment) modes.

12 methods:
  cpb_full          : DreamerV3 + CPB (match + cap)
  cpb_match_only    : DreamerV3 + CPB match only (no cap)
  cpb_cap_only      : DreamerV3 + CPB cap only (no match)
  dreamer_default   : DreamerV3 τ=1.0
  dreamer_fixed_001 : DreamerV3 τ=0.01
  dreamer_fixed_01  : DreamerV3 τ=0.1
  dreamer_fixed_05  : DreamerV3 τ=0.5
  dreamer_autotuned : DreamerV3 SAC entropy autotuning
  ppo_entropy       : PPO with entropy bonus
  noisynet          : DreamerV3 + NoisyNet exploration
  ensemble_disag    : DreamerV3 + ensemble disagreement bonus
  rnd               : DreamerV3 + RND intrinsic reward

Usage::

    # Dry run (CI, no dm_control required)
    python -m experiments.exp2_dreamer.run_main --dry-run --method cpb_full

    # Full Hydra run
    python -m experiments.exp2_dreamer.run_main \\
        env=cheetah_run noise=clean method=cpb_full seed=0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ── method registry ───────────────────────────────────────────────────────────

_METHODS = [
    "cpb_full", "cpb_match_only", "cpb_cap_only",
    "dreamer_default", "dreamer_fixed_001", "dreamer_fixed_01", "dreamer_fixed_05",
    "dreamer_autotuned",
    "ppo_entropy", "noisynet", "ensemble_disag", "rnd",
]


# ── environment builder ───────────────────────────────────────────────────────

def _build_env(suite: str, task: str, sigma_obs: float, sigma_reward: float,
               sigma_dynamics: float, seed: int) -> Any:
    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env
    return make_stochastic_dmc_env(
        suite, task,
        sigma_obs=sigma_obs, sigma_reward=sigma_reward,
        sigma_dynamics=sigma_dynamics, seed=seed,
    )


# ── policy builder ────────────────────────────────────────────────────────────

def _build_policy(method: str, env: Any, seed: int, config_path: str,
                  device: str, kmeans_path: str | None = None,
                  c_cal: float | None = None) -> Any:
    from nms.wm.dreamer_wrapper import DreamerWrapper

    dreamer = DreamerWrapper(config_path=config_path, env=env, device=device, seed=seed)

    if method in ("cpb_full", "cpb_match_only", "cpb_cap_only"):
        from nms.controllers.cpb_dreamer import CPBDreamerController
        from nms.interfaces.kmeans_interface import load_interface

        km = load_interface(kmeans_path) if kmeans_path else None
        if km is None:
            warnings.warn("kmeans not provided for CPB — using default.", stacklevel=2)
            from nms.wm.dreamer_wrapper import KMeansActionDiscretizer
            km = KMeansActionDiscretizer(n_clusters=20)
            rng = np.random.default_rng(seed)
            dummy_actions = rng.uniform(-1, 1, size=(500, env.action_space.shape[0]))
            km.fit(dummy_actions)

        ctrl = CPBDreamerController(dreamer, km, alpha=0.1, horizon=15, c_cal=c_cal)

        if method == "cpb_cap_only":
            # Cap only: use c_cal directly, no match calibration
            ctrl._bandwidth = {}
        return ctrl

    if method == "dreamer_default":
        from nms.baselines.dreamer_variants import make_dreamer_default
        return make_dreamer_default(dreamer)
    if method == "dreamer_fixed_001":
        from nms.baselines.dreamer_variants import make_dreamer_fixed
        return make_dreamer_fixed(dreamer, 0.01)
    if method == "dreamer_fixed_01":
        from nms.baselines.dreamer_variants import make_dreamer_fixed
        return make_dreamer_fixed(dreamer, 0.1)
    if method == "dreamer_fixed_05":
        from nms.baselines.dreamer_variants import make_dreamer_fixed
        return make_dreamer_fixed(dreamer, 0.5)
    if method == "dreamer_autotuned":
        from nms.baselines.dreamer_variants import AutotunedDreamer
        act_dim = env.action_space.shape[0]
        return AutotunedDreamer(dreamer, act_dim=act_dim)

    if method == "ppo_entropy":
        from nms.baselines.ppo_entropy import PPOEntropyBaseline
        return PPOEntropyBaseline(
            env.observation_space, env.action_space,
            entropy_coef=0.01, device=device, seed=seed,
        )

    if method == "noisynet":
        from nms.baselines.dreamer_variants import make_dreamer_default
        from nms.baselines.noisy_net_wrapper import NoisyNetWrapper
        base = make_dreamer_default(dreamer)
        obs_dim = int(np.prod(env.observation_space.shape))
        return NoisyNetWrapper(base, obs_dim=obs_dim, device=device, seed=seed)

    if method == "ensemble_disag":
        from nms.baselines.dreamer_variants import make_dreamer_default
        from nms.baselines.ensemble_disagreement import EnsembleDisagreementBaseline
        from nms.wm.ensemble_rssm import EnsembleRSSM

        class _FakeEnvForEnsemble:
            def __init__(self) -> None:
                self.observation_space = env.observation_space
                self.action_space = env.action_space

        ensemble = EnsembleRSSM(
            env.observation_space, env.action_space,
            k=3, device=device, seed=seed, config_path=config_path,
        )
        base = make_dreamer_default(dreamer)
        return EnsembleDisagreementBaseline(base, ensemble, bonus_scale=0.1)

    if method == "rnd":
        from nms.baselines.dreamer_variants import make_dreamer_default
        from nms.baselines.rnd_wrapper import RNDWrapper
        obs_dim = int(np.prod(env.observation_space.shape))
        base = make_dreamer_default(dreamer)
        return RNDWrapper(base, obs_dim=obs_dim, device=device, seed=seed)

    raise ValueError(f"Unknown method: {method!r}. Valid: {_METHODS}")


# ── training loop ─────────────────────────────────────────────────────────────

def _run(policy: Any, env: Any, total_steps: int, eval_every: int,
         out_dir: Path) -> dict[str, Any]:
    step = 0
    episode = 0
    returns: list[float] = []

    while step < total_steps:
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        while not done and step < total_steps:
            action = policy.sample_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
            step += 1
        episode += 1
        returns.append(ep_return)
        if episode % max(1, eval_every // 200) == 0 or step >= total_steps:
            mean_ret = float(np.mean(returns[-20:])) if returns else 0.0
            print(f"  step={step:>8d}  ep={episode:>5d}  return={mean_ret:.2f}")

    last_100 = returns[-100:] if returns else []
    return {
        "total_steps": step,
        "episodes": episode,
        "mean_return_last_100": float(np.mean(last_100)) if last_100 else 0.0,
        "std_return_last_100": float(np.std(last_100)) if last_100 else 0.0,
    }


# ── argparse (dry-run / smoke) mode ──────────────────────────────────────────

def _argparse_main() -> None:
    ap = argparse.ArgumentParser(description="Exp 2 run_main — all 12 methods.")
    ap.add_argument("--env", default="cheetah-run")
    ap.add_argument("--method", default="cpb_full", choices=_METHODS)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-obs", type=float, default=0.0)
    ap.add_argument("--sigma-reward", type=float, default=0.0)
    ap.add_argument("--sigma-dynamics", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--device", type=str, default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    ap.add_argument("--kmeans-path", type=str, default=None)
    ap.add_argument("--config", type=str,
                    default=str(Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml"))  # noqa: E501
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_main/run"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify imports and policy construction without training.")
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Exp 2 run_main ===")
    print(f"  method={args.method}  env={suite}/{task}  steps={args.steps}  seed={args.seed}")

    if args.dry_run:
        print("  [dry-run] Importing and constructing policy ...")

        class _MockEnv:
            class _Space:
                shape = (6,)
                low = np.array([-1.0] * 6)
                high = np.array([1.0] * 6)
            observation_space = _Space()
            action_space = _Space()

        dummy = _MockEnv()
        policy = _build_policy(
            args.method, dummy, seed=args.seed,
            config_path=args.config, device=args.device,
            kmeans_path=args.kmeans_path,
        )
        print(f"  [dry-run] Policy type: {type(policy).__name__}")
        print("  [dry-run] OK — no training performed.")
        return

    env = _build_env(suite, task,
                     sigma_obs=args.sigma_obs, sigma_reward=args.sigma_reward,
                     sigma_dynamics=args.sigma_dynamics, seed=args.seed)
    policy = _build_policy(args.method, env, seed=args.seed, config_path=args.config,
                           device=args.device, kmeans_path=args.kmeans_path)

    summary = _run(policy, env, total_steps=args.steps,
                   eval_every=args.eval_every, out_dir=out_dir)
    env.close()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    run_info = {
        "method": args.method,
        "env": args.env,
        "steps": args.steps,
        "seed": args.seed,
        "git_sha": git_sha,
        **summary,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"\n  mean_return_last_100 = {summary['mean_return_last_100']:.2f}")
    print(f"  Saved → {out_dir / 'run_info.json'}")


# ── Hydra mode ────────────────────────────────────────────────────────────────

def _hydra_main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        config_path="../../configs/exp2_rl",
        config_name="main",
        version_base=None,
    )
    def _inner(cfg: DictConfig) -> None:
        from nms.utils.seeds import set_global_seed
        set_global_seed(int(cfg.seed))

        out_dir = Path(f"results/runs/exp2_{cfg.env.suite}_{cfg.env.task}"
                       f"/{cfg.method.name}/seed{cfg.seed}")
        out_dir.mkdir(parents=True, exist_ok=True)

        env = _build_env(
            suite=cfg.env.suite, task=cfg.env.task,
            sigma_obs=float(cfg.noise.sigma_obs),
            sigma_reward=float(cfg.noise.sigma_reward),
            sigma_dynamics=float(cfg.noise.sigma_dynamics),
            seed=int(cfg.seed),
        )

        config_path = str(
            Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml"
        )
        policy = _build_policy(
            method=str(cfg.method.name),
            env=env,
            seed=int(cfg.seed),
            config_path=config_path,
            device=str(cfg.get("device", "cpu")),
        )
        run_summary = _run(
            policy, env,
            total_steps=int(cfg.training.total_steps),
            eval_every=int(cfg.training.eval_every),
            out_dir=out_dir,
        )
        env.close()

        import yaml

        from nms.utils.logging import write_run_info
        write_run_info(out_dir, yaml.dump(dict(cfg)))
        (out_dir / "summary.json").write_text(json.dumps(run_summary, indent=2))

    _inner()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flag_args = {"--dry-run", "--env", "--method", "--steps", "--seed", "--out", "--help"}
    if any(a.split("=")[0] in flag_args for a in sys.argv[1:]) or "--dry-run" in sys.argv:
        _argparse_main()
    else:
        _hydra_main()
