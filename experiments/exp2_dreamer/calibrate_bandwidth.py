"""Exp 2 Phase B: Calibrate NMS bandwidth B_{H,α}(ν) for DreamerV3.

Uses CPBDreamerController to sweep σ values and estimate the empirical
violation rate R_H(σ; ν) for multiple ν values.  Saves bandwidth curve
to a CSV and certified bandwidth per ν to a JSON.

Usage (standard)::

    python -m experiments.exp2_dreamer.calibrate_bandwidth \\
        --env cheetah-run \\
        --dreamer-dir results/runs/exp2_cheetah_run/seed0/dreamer_logs \\
        --kmeans-path results/runs/exp2_cheetah_run/seed0/kmeans.npz \\
        --nu-values 0.05 0.10 0.15 0.20 0.30 \\
        --sigma-grid-max 0.5 \\
        --n-pairs 200 \\
        --horizon 15 \\
        --alpha 0.1 \\
        --seed 0 \\
        --out results/runs/exp2_cheetah_run/seed0/bandwidth

Usage (OOD pilot, Phase 1.2)::

    python -m experiments.exp2_dreamer.calibrate_bandwidth \\
        --env cheetah-run \\
        --dreamer-dir results/runs/exp2_smoke/cheetah_dv3_100K/member_0 \\
        --kmeans-path results/interfaces/cheetah_run_k8.npz \\
        --nu-grid 0.2 \\
        --ood-mass-friction 0.4 \\
        --n-pairs 30 --horizon 5 --alpha 0.3 \\
        --sigma-grid-max 1.5 --sigma-grid-n 10 \\
        --out results/runs/exp2/_smoke/cheetah_ood_pilot_delta0.4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Exp 2 NMS bandwidth.")
    ap.add_argument("--env", default="cheetah-run")
    ap.add_argument("--dreamer-dir", type=Path, required=True,
                    help="Directory containing trained DreamerWrapper checkpoint "
                         "(must contain latest.pt or a member_0/ subdirectory).")
    ap.add_argument("--kmeans-path", type=Path, required=True)
    # Accept either --nu-values or --nu-grid (aliases)
    _nu_group = ap.add_mutually_exclusive_group()
    _nu_group.add_argument("--nu-values", type=float, nargs="+",
                           default=None,
                           help="List of ν̂ levels to calibrate.")
    _nu_group.add_argument("--nu-grid", type=float, nargs="+",
                           default=None,
                           help="Alias for --nu-values (smoke gate: pass single value).")
    ap.add_argument("--sigma-grid-max", type=float, default=0.5)
    ap.add_argument("--sigma-grid-n", type=int, default=21)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    ap.add_argument("--ood-mass-friction", type=float, default=None,
                    metavar="DELTA",
                    help="If set, wrap the calibration environment with "
                         "MassFrictionPerturbWrapper(delta) for OOD testing "
                         "(Phase 1.2 / Phase 9).  Default: None (no OOD wrapping).")
    ap.add_argument("--nu-estimator", choices=["fixed", "ensemble"], default="fixed",
                    help="How to compute ν̂ for each calibration pair. "
                         "'fixed' (default): use --nu-values/--nu-grid as fixed env-noise "
                         "levels. 'ensemble' (Phase 2.5): use EnsembleNuEstimator to "
                         "compute ν̂ from the K-member ensemble at each calibration s_0. "
                         "Requires --dreamer-dir to point at an ensemble (member_0/, "
                         "member_1/, …) rather than a single member.")
    ap.add_argument("--nu-estimator-horizon", type=int, default=5,
                    help="Horizon H for EnsembleNuEstimator (only used with "
                         "--nu-estimator ensemble).")
    ap.add_argument("--nu-estimator-samples", type=int, default=8,
                    help="n_samples for EnsembleNuEstimator (only used with "
                         "--nu-estimator ensemble).")
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2_calibrate/bandwidth"))
    args = ap.parse_args()

    # Resolve nu_values from either --nu-values or --nu-grid
    nu_values: list[float]
    if args.nu_grid is not None:
        nu_values = args.nu_grid
    elif args.nu_values is not None:
        nu_values = args.nu_values
    else:
        nu_values = [0.05, 0.10, 0.15, 0.20, 0.30]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite, task = args.env.split("-", 1)
    sigma_grid = np.linspace(0.0, args.sigma_grid_max, args.sigma_grid_n)

    from nms.controllers.cpb_dreamer import CPBDreamerController
    from nms.envs.stochastic_dm_control import make_stochastic_dmc_env
    from nms.interfaces.kmeans_interface import load_interface
    from nms.wm.dreamer_wrapper import DreamerWrapper

    # Load k-means
    kmeans = load_interface(args.kmeans_path)

    # Build a base env to give DreamerWrapper its obs/act spaces
    env_base = make_stochastic_dmc_env(suite, task, seed=args.seed)

    config_path = str(Path(__file__).parent.parent.parent / "configs" / "exp2_rl" / "config.yaml")
    dreamer = DreamerWrapper(
        config_path=config_path, env=env_base,
        device=args.device, seed=args.seed,
    )

    # ── Locate checkpoint directory ───────────────────────────────────────────
    dreamer_dir = Path(args.dreamer_dir)
    _candidates = [
        dreamer_dir,
        dreamer_dir / "member_0",
    ]
    ckpt_dir = next((p for p in _candidates if (p / "latest.pt").exists()), None)
    if ckpt_dir is None:
        raise FileNotFoundError(
            f"No latest.pt found in any of: {[str(p) for p in _candidates]}\n"
            f"Train the model first (train_ensemble.py or train_dreamer.py)."
        )
    print(f"Loading trained agent from {ckpt_dir}/latest.pt …")
    dreamer.load_from_dir(str(ckpt_dir))
    print(f"Agent backend: {dreamer._backend}")

    # ── Ensemble ν̂ estimator (Phase 2.5) ──────────────────────────────────────
    nu_estimator = None
    if args.nu_estimator == "ensemble":
        from nms.wm.dreamer_wrapper import DreamerWrapper as _DW
        from nms.wm.ensemble_nu import EnsembleNuEstimator

        # Load all available member checkpoints from dreamer_dir
        members: list[Any] = [dreamer]  # member_0 already loaded
        member_idx = 1
        while True:
            m_dir = dreamer_dir / f"member_{member_idx}"
            if not (m_dir / "latest.pt").exists():
                break
            print(f"Loading ensemble member {member_idx} from {m_dir} …")
            m = _DW(config_path=config_path, env=env_base, device=args.device, seed=args.seed + member_idx)
            m.load_from_dir(str(m_dir))
            members.append(m)
            member_idx += 1

        nu_estimator = EnsembleNuEstimator(
            members=members,
            horizon=args.nu_estimator_horizon,
        )
        print(f"EnsembleNuEstimator: {nu_estimator.k} members, H={nu_estimator.H}")

    ctrl = CPBDreamerController(dreamer, kmeans, alpha=args.alpha, horizon=args.horizon)

    # ── Calibrate for each ν ──────────────────────────────────────────────────
    all_rows: list[dict] = []
    bandwidth_dict: dict[str, float] = {}  # string keys for JSON serialisation

    ood_delta = args.ood_mass_friction
    ood_label = f"  [OOD δ={ood_delta:.2f}]" if ood_delta is not None else ""
    nu_est_label = f"  [ν̂-estimator: {args.nu_estimator}]" if args.nu_estimator != "fixed" else ""
    print(f"\n=== Bandwidth calibration: {args.env}{ood_label}{nu_est_label} ===")
    print(f"  ν̂ values : {nu_values}")
    print(f"  σ grid   : 0 … {args.sigma_grid_max} ({args.sigma_grid_n} pts)")
    print(f"  n_pairs  : {args.n_pairs}")
    print(f"  horizon  : {args.horizon}")
    print(f"  α        : {args.alpha}")

    for nu in nu_values:
        print(f"\nCalibrating ν={nu:.3f} …")

        # Build the calibration env (with optional OOD perturbation)
        env_nu = make_stochastic_dmc_env(suite, task, sigma_obs=nu, seed=args.seed)
        if ood_delta is not None and ood_delta > 0.0:
            from nms.envs.ood_wrappers import MassFrictionPerturbWrapper
            env_nu = MassFrictionPerturbWrapper(env_nu, delta=ood_delta)

        if nu_estimator is not None:
            # Phase 2.5: compute ν̂ from ensemble at a batch of starting states
            print(f"  Using EnsembleNuEstimator to compute ν̂ …")
            _obs_sample, _ = env_nu.reset()
            nu_hat_computed = nu_estimator.estimate(
                _obs_sample, n_samples=args.nu_estimator_samples
            )
            print(f"  ν̂_ensemble = {nu_hat_computed:.4f}  (fixed ν = {nu:.3f})")
            ctrl.calibrate(env_nu, nu_hat=nu_hat_computed,
                           sigma_grid=sigma_grid, n_pairs=args.n_pairs)
            env_nu.close()
            bw = ctrl.get_bandwidth(nu_hat_computed)
            bandwidth_dict[str(nu_hat_computed)] = float(bw)
            for sigma, r_h in ctrl.violation_rates.items():
                all_rows.append({
                    "nu": nu_hat_computed, "sigma": float(sigma),
                    "r_h": float(r_h), "bandwidth": float(bw),
                    "nu_fixed": nu,
                })
        else:
            # Fixed ν̂ mode (Phase 1, Phase 4 default)
            ctrl.calibrate(env_nu, nu_hat=nu, sigma_grid=sigma_grid, n_pairs=args.n_pairs)
            env_nu.close()
            bw = ctrl.get_bandwidth(nu)
            bandwidth_dict[str(nu)] = float(bw)
            for sigma, r_h in ctrl.violation_rates.items():
                all_rows.append({
                    "nu": nu, "sigma": float(sigma),
                    "r_h": float(r_h), "bandwidth": float(bw),
                })

    # ── Save outputs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "bandwidth_curve.csv"
    df.to_csv(csv_path, index=False)

    meta: dict = {
        "alpha": args.alpha,
        "horizon": args.horizon,
        "env": args.env,
        "bandwidth": bandwidth_dict,
    }
    if ood_delta is not None:
        meta["ood_mass_friction_delta"] = ood_delta

    json_path = out_dir / "bandwidth.json"
    json_path.write_text(json.dumps(meta, indent=2))

    env_base.close()
    print("\nBandwidth calibration complete.")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    for nu_str, bw in bandwidth_dict.items():
        print(f"  ν={float(nu_str):.3f}  B_{{H,α}}={bw:.4f}")


if __name__ == "__main__":
    main()
