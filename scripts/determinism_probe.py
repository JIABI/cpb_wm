"""Amendment A: Determinism probe for Phase 1 smoke-gate checkpoints.

Verifies two properties of the trained DreamerV3 world model BEFORE
declaring any Phase 1 gate RED at 100k steps:

  1. **Determinism at σ=0**: Given the same initial observation, the
     imagined-rollout first action must exactly match the real-rollout
     first action (both use actor.mode(), no noise).

  2. **Noise visibility at σ=0.3**: When actor temperature = 0.3, the
     imagined first action must diverge from the real first action by
     more than 0.1 in L2 norm. This proves the σ-noise-injection patch
     is active and not silently bypassed.

  3. **WM drift accumulation at σ=0**: Over H=5 imagination steps, the
     per-step action distance must grow (world-model errors compound).

Acceptance criteria (all must pass before declaring Phase 1 RED):
  - byte_equal_count_5seeds == 5
  - mean_t0_dist_at_sigma0    < 1e-5
  - mean_t0_dist_at_sigma_0_3 > 0.10
  - per_step_dist_sigma0[0]   < 1e-5
  - per_step_dist_sigma0[H-1] > per_step_dist_sigma0[0]

STOP-condition: if any criterion fails, report failing criterion and
dump raw a_imag[0], a_real[0] for inspection. Do NOT proceed to Phase 2.

Usage::

    python scripts/determinism_probe.py \\
        --env hopper-hop \\
        --dreamer-dir results/runs/exp2/_smoke/hopper_dv3_K1/ \\
        --kmeans-path results/interfaces/hopper_hop_k8.npz \\
        --out results/runs/exp2/_smoke/determinism_probe_hopper_100k.json

    python scripts/determinism_probe.py \\
        --env cheetah-run \\
        --dreamer-dir results/runs/exp2/_smoke/cheetah_dv3_100K/ \\
        --kmeans-path results/interfaces/cheetah_run_k8.npz \\
        --out results/runs/exp2/_smoke/determinism_probe_cheetah_100k.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


_SEEDS = [42, 43, 44, 45, 46]
_HORIZON = 5
_SIGMA_PROBE = 0.3
_CRITERION_T0_SIGMA0_MAX = 1e-5
_CRITERION_T0_SIGMA_0_3_MIN = 0.10


def _load_dreamer(dreamer_dir: Path, env: Any, device: str, seed: int) -> Any:
    """Load DreamerWrapper from checkpoint directory."""
    from nms.wm.dreamer_wrapper import DreamerWrapper

    config_path = str(Path(__file__).parent.parent / "configs" / "exp2_rl" / "config.yaml")
    dreamer = DreamerWrapper(config_path=config_path, env=env, device=device, seed=seed)

    # Resolve checkpoint: try dir directly, then member_0 subdirectory
    candidates = [dreamer_dir, dreamer_dir / "member_0"]
    ckpt_dir = next((p for p in candidates if (p / "latest.pt").exists()), None)
    if ckpt_dir is None:
        raise FileNotFoundError(
            f"No latest.pt found in {[str(p) for p in candidates]}"
        )
    print(f"Loading checkpoint from {ckpt_dir}/latest.pt ...")
    dreamer.load_from_dir(str(ckpt_dir))
    return dreamer


def run_probe(
    dreamer: Any,
    env: Any,
    horizon: int = _HORIZON,
    seeds: list[int] = _SEEDS,
    sigma_probe: float = _SIGMA_PROBE,
) -> dict:
    """Execute the full determinism probe and return result dict."""
    # Accumulators
    t0_dists_sigma0: list[float] = []
    t0_dists_sigma_p: list[float] = []
    per_step_sigma0_accum: list[list[float]] = []  # [seed_idx][step]
    per_step_sigma_p_accum: list[list[float]] = []
    byte_equal_list: list[bool] = []

    print(f"\n=== Determinism probe (H={horizon}, σ_probe={sigma_probe}) ===")

    for seed in seeds:
        print(f"\n  Seed {seed}:")

        # ── 1. Real rollout at σ=0 ───────────────────────────────────────
        dreamer.set_actor_temperature(0.0)
        real_traj = dreamer.rollout(env, horizon=horizon, seed=seed)
        obs_real0 = real_traj["obs"][0]  # shape (obs_dim,)
        a_real = real_traj["actions"]    # shape (H, act_dim)

        print(f"    real  a[0] = {a_real[0]}")

        # ── 2. Imagined rollout at σ=0 ───────────────────────────────────
        imag_traj_0 = dreamer.rollout_imagined(obs_real0, horizon=horizon, sigma=0.0, seed=seed)
        a_imag_0 = imag_traj_0["actions"]  # shape (H, act_dim)

        print(f"    imag0 a[0] = {a_imag_0[0]}")

        # t=0 distance at σ=0
        t0_dist_0 = float(np.linalg.norm(a_imag_0[0] - a_real[0]))
        t0_dists_sigma0.append(t0_dist_0)

        # Byte equality at t=0
        be = np.array_equal(a_imag_0[0], a_real[0])
        byte_equal_list.append(bool(be))
        print(f"    ||a_imag[0] - a_real[0]|| @ σ=0 : {t0_dist_0:.2e}  (byte_equal={be})")

        # Per-step distances at σ=0
        per_step_0 = []
        H_actual = min(len(a_real), len(a_imag_0))
        for t in range(H_actual):
            d = float(np.linalg.norm(a_imag_0[t] - a_real[t]))
            per_step_0.append(d)
        per_step_sigma0_accum.append(per_step_0)

        # ── 3. Imagined rollout at σ=σ_probe ────────────────────────────
        imag_traj_p = dreamer.rollout_imagined(obs_real0, horizon=horizon, sigma=sigma_probe, seed=seed)
        a_imag_p = imag_traj_p["actions"]

        t0_dist_p = float(np.linalg.norm(a_imag_p[0] - a_real[0]))
        t0_dists_sigma_p.append(t0_dist_p)
        print(f"    ||a_imag[0] - a_real[0]|| @ σ={sigma_probe}: {t0_dist_p:.4f}")

        per_step_p = []
        H_p_actual = min(len(a_real), len(a_imag_p))
        for t in range(H_p_actual):
            d = float(np.linalg.norm(a_imag_p[t] - a_real[t]))
            per_step_p.append(d)
        per_step_sigma_p_accum.append(per_step_p)

    # ── Aggregate ────────────────────────────────────────────────────────
    mean_t0_dist_sigma0 = float(np.mean(t0_dists_sigma0))
    mean_t0_dist_sigma_p = float(np.mean(t0_dists_sigma_p))
    byte_equal_count = int(sum(byte_equal_list))

    # Average per-step distances across seeds (pad shorter trajs)
    def _mean_per_step(accum: list[list[float]]) -> list[float]:
        max_len = max(len(s) for s in accum)
        result = []
        for t in range(max_len):
            vals = [s[t] for s in accum if t < len(s)]
            result.append(float(np.mean(vals)))
        return result

    per_step_sigma0_mean = _mean_per_step(per_step_sigma0_accum)
    per_step_sigma_p_mean = _mean_per_step(per_step_sigma_p_accum)

    print(f"\n  Summary:")
    print(f"    byte_equal_count_5seeds       = {byte_equal_count}/5")
    print(f"    mean_t0_dist_at_sigma0        = {mean_t0_dist_sigma0:.2e}  (need < {_CRITERION_T0_SIGMA0_MAX:.0e})")
    print(f"    mean_t0_dist_at_sigma_{sigma_probe} = {mean_t0_dist_sigma_p:.4f}  (need > {_CRITERION_T0_SIGMA_0_3_MIN})")
    print(f"    per_step_dist_sigma0          = {[f'{d:.2e}' for d in per_step_sigma0_mean]}")
    print(f"    per_step_dist_sigma_{sigma_probe}  = {[f'{d:.4f}' for d in per_step_sigma_p_mean]}")

    result: dict = {
        "byte_equal_at_sigma0_seed_42": byte_equal_list[0] if byte_equal_list else None,
        "byte_equal_count_5seeds": byte_equal_count,
        "mean_t0_dist_at_sigma0": mean_t0_dist_sigma0,
        f"mean_t0_dist_at_sigma_{sigma_probe}".replace(".", "_"): mean_t0_dist_sigma_p,
        "per_step_dist_sigma0": per_step_sigma0_mean,
        f"per_step_dist_sigma_{sigma_probe}".replace(".", "_"): per_step_sigma_p_mean,
        # store raw first-action pairs for debugging
        "_raw_a_real_0_seed42": a_real[0].tolist() if len(_SEEDS) > 0 else None,
        "_raw_a_imag0_0_seed42": None,  # filled below if seeds included 42
        "_raw_a_imagp_0_seed42": None,
    }
    return result


def check_acceptance(result: dict, horizon: int = _HORIZON) -> tuple[bool, list[str]]:
    """Return (all_pass, list_of_failure_messages)."""
    failures = []

    # Criterion 1: byte equality
    bec = result.get("byte_equal_count_5seeds", 0)
    if bec < 5:
        failures.append(
            f"FAIL byte_equal_count_5seeds={bec} (need 5). "
            f"Noise-injection or seeding is broken at σ=0."
        )

    # Criterion 2: t0 dist at sigma=0
    d0 = result.get("mean_t0_dist_at_sigma0", 999.0)
    if d0 >= _CRITERION_T0_SIGMA0_MAX:
        failures.append(
            f"FAIL mean_t0_dist_at_sigma0={d0:.2e} ≥ {_CRITERION_T0_SIGMA0_MAX:.0e}. "
            f"First action not reproducible from same obs at σ=0."
        )

    # Criterion 3: t0 dist at sigma=probe
    sigma_key = f"mean_t0_dist_at_sigma_{_SIGMA_PROBE}".replace(".", "_")
    dp = result.get(sigma_key, 0.0)
    if dp < _CRITERION_T0_SIGMA_0_3_MIN:
        failures.append(
            f"FAIL {sigma_key}={dp:.4f} < {_CRITERION_T0_SIGMA_0_3_MIN}. "
            f"σ={_SIGMA_PROBE} noise is not producing visible action divergence."
        )

    # Criterion 4: per-step drift at sigma=0
    psd = result.get("per_step_dist_sigma0", [])
    if len(psd) >= 2:
        if psd[0] >= _CRITERION_T0_SIGMA0_MAX:
            failures.append(
                f"FAIL per_step_dist_sigma0[0]={psd[0]:.2e} ≥ {_CRITERION_T0_SIGMA0_MAX:.0e}. "
                f"Step-0 is not deterministic."
            )
        if psd[-1] <= psd[0]:
            failures.append(
                f"FAIL per_step_dist_sigma0[-1]={psd[-1]:.4f} ≤ per_step_dist_sigma0[0]={psd[0]:.2e}. "
                f"WM drift is not accumulating over horizon."
            )
    else:
        failures.append("FAIL per_step_dist_sigma0 has fewer than 2 entries.")

    return len(failures) == 0, failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Amendment A determinism probe.")
    ap.add_argument("--env", required=True, help="Environment as 'suite-task'.")
    ap.add_argument("--dreamer-dir", type=Path, required=True)
    ap.add_argument("--kmeans-path", type=Path, required=True,
                    help="Path to k-means .npz (used only to check file exists).")
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2/_smoke/determinism_probe.json"))
    ap.add_argument("--horizon", type=int, default=_HORIZON)
    ap.add_argument("--sigma-probe", type=float, default=_SIGMA_PROBE)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for DreamerWrapper construction (not probe seeds).")
    args = ap.parse_args()

    suite, task = args.env.split("-", 1)

    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env

    env = make_stochastic_dmc_env(suite, task, seed=args.seed)
    dreamer = _load_dreamer(Path(args.dreamer_dir), env, device=args.device, seed=args.seed)

    # Verify k-means file exists (not used for probe, but required as dep)
    if not Path(args.kmeans_path).exists():
        print(f"WARNING: k-means not found at {args.kmeans_path} — probe continues without it.")

    result = run_probe(dreamer, env, horizon=args.horizon, sigma_probe=args.sigma_probe)

    # Add metadata
    result["env"] = args.env
    result["dreamer_dir"] = str(args.dreamer_dir)
    result["horizon"] = args.horizon
    result["sigma_probe"] = args.sigma_probe

    # Check acceptance
    all_pass, failures = check_acceptance(result, horizon=args.horizon)

    print(f"\n{'='*60}")
    if all_pass:
        print("✅ DETERMINISM PROBE PASSED — all 4 criteria satisfied.")
        print("   Proceed to Phase 1 calibration.")
        result["verdict"] = "PASS"
    else:
        print("❌ DETERMINISM PROBE FAILED — STOP-condition triggered.")
        print("   Do NOT retrain further. Do NOT advance to Phase 2.")
        print("\n   Failing criteria:")
        for f in failures:
            print(f"     • {f}")
        print("\n   Raw action dump (seed=42):")
        print(f"     a_real[0]  = {result.get('_raw_a_real_0_seed42')}")
        print(f"     a_imag0[0] = {result.get('_raw_a_imag0_0_seed42')}")
        result["verdict"] = "FAIL"
        result["failures"] = failures
    print(f"{'='*60}")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nOutput: {out_path}")

    env.close()

    # Exit code signals pass/fail for shell pipelines
    import sys
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
