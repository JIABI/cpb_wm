"""Exp 2 Phase 1: Smoke-gate fast training (Phase 1.1 / 1.2).

Trains a SINGLE DreamerV3 world model with fast CPU-friendly config for use
in Phase 1 smoke gates.  The fast config (batch_size=4, batch_length=8,
train_ratio=8, prefill=500) mirrors the CPU integration test and completes
in ~20-30 minutes per run.

This is NOT the production training script (use train_ensemble.py for that).
The output checkpoint is suitable for smoke-gate calibration only.

Usage::

    python -m experiments.exp2_dreamer.train_smoke \\
        --env hopper-hop --sigma-obs 0.1 --steps 10000 --seed 0 \\
        --out results/runs/exp2/_smoke/hopper_dv3_K1/member_0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-gate fast DreamerV3 training.")
    ap.add_argument("--env", default="hopper-hop",
                    help="Environment as 'suite-task'.")
    ap.add_argument("--steps", type=int, default=10_000,
                    help="Total env steps (fast config, not production).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-obs", type=float, default=0.1)
    ap.add_argument("--sigma-reward", type=float, default=0.0)
    ap.add_argument("--sigma-dynamics", type=float, default=0.0)
    ap.add_argument("--config", type=str,
                    default="configs/exp2_rl/config.yaml")
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2/_smoke/member_0"))
    # Fast config overrides (for CPU smoke runs)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batch-length", type=int, default=8)
    ap.add_argument("--train-ratio", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=500)
    ap.add_argument("--pretrain", type=int, default=1)
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device: cpu (default for smoke), mps, cuda.")
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env
    from nms.wm.dreamer_wrapper import DreamerWrapper

    env = make_stochastic_dmc_env(
        suite, task,
        sigma_obs=args.sigma_obs,
        sigma_reward=args.sigma_reward,
        sigma_dynamics=args.sigma_dynamics,
        seed=args.seed,
    )

    print(f"\n=== Smoke-gate training: {args.env} ===")
    print(f"  steps      = {args.steps:,}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  batch_len  = {args.batch_length}")
    print(f"  prefill    = {args.prefill}")
    print(f"  sigma_obs  = {args.sigma_obs}")
    print(f"  device     = {args.device}")
    print(f"  out        = {out_dir}\n")

    agent = DreamerWrapper(
        config_path=args.config,
        env=env,
        device=args.device,
        seed=args.seed,
    )

    # Override config for fast CPU run
    agent._ensure_agent()
    cfg = agent._nm_config
    cfg.batch_size   = args.batch_size
    cfg.batch_length = args.batch_length
    cfg.train_ratio  = args.train_ratio
    cfg.prefill      = args.prefill
    cfg.pretrain     = args.pretrain

    t0 = time.time()
    agent.train(total_steps=args.steps, log_dir=str(out_dir))
    elapsed = time.time() - t0

    env.close()

    ckpt = out_dir / "latest.pt"
    if ckpt.exists():
        print(f"\nCheckpoint saved: {ckpt} ({ckpt.stat().st_size//1024} KB)")
    else:
        raise RuntimeError(f"Training completed but no checkpoint at {ckpt}!")

    # Save run info
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"

    run_info = {
        "env": args.env,
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "batch_length": args.batch_length,
        "train_ratio": args.train_ratio,
        "prefill": args.prefill,
        "sigma_obs": args.sigma_obs,
        "device": args.device,
        "elapsed_s": round(elapsed, 1),
        "smoke_gate": True,
        "git_sha": git_sha,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    print(f"\nTraining complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Steps/s   : {args.steps / elapsed:.1f}")
    print(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
