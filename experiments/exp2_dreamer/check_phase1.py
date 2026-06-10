"""Phase 1 smoke-gate acceptance checker.

Reads calibration output from calibrate_bandwidth.py and reports
GREEN / YELLOW / RED for both Phase 1.1 (hopper-hop U-shape) and
Phase 1.2 (cheetah-run OOD bandwidth collapse).

Usage::

    # Phase 1.1 check (U-shape)
    python -m experiments.exp2_dreamer.check_phase1 \
        --gate 1.1 \
        --csv results/runs/exp2/_smoke/hopper_calibration_smoke.json/bandwidth_curve.csv

    # Phase 1.2 check (OOD collapse)
    python -m experiments.exp2_dreamer.check_phase1 \
        --gate 1.2 \
        --delta-dirs results/runs/exp2/_smoke/cheetah_ood_pilot_delta0.0.json \\
                     results/runs/exp2/_smoke/cheetah_ood_pilot_delta0.2.json \\
                     ...

    # Or auto-discover delta dirs:
    python -m experiments.exp2_dreamer.check_phase1 \\
        --gate 1.2 \\
        --ood-dir results/runs/exp2/_smoke \\
        --ood-prefix cheetah_ood_pilot_delta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check_gate_1_1(csv_path: Path) -> str:
    """Check Phase 1.1: hopper-hop U-shape gate.

    Returns 'GREEN', 'YELLOW', or 'RED'.
    """
    import csv

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERROR: empty CSV")
        return "RED"

    # Filter to nu=0.2 (the smoke gate nu)
    nu_rows = [r for r in rows if abs(float(r["nu"]) - 0.2) < 0.01]
    if not nu_rows:
        # Use all rows if no nu=0.2 filter applies
        nu_rows = rows
        print("Warning: no rows with nu=0.2, using all rows")

    r0 = float(nu_rows[0]["r_h"])  # R_H at sigma=0

    mid_rows = [r for r in nu_rows if 0.2 <= float(r["sigma"]) <= 0.8]
    r_min_mid = min(float(r["r_h"]) for r in mid_rows) if mid_rows else float("inf")
    r_max = max(float(r["r_h"]) for r in nu_rows)

    print(f"\n=== Phase 1.1: Hopper-hop U-shape gate ===")
    print(f"  R_H(σ=0)       = {r0:.4f}  (need > 0.20 for GREEN)")
    print(f"  min R_H(mid)   = {r_min_mid:.4f}  (need < 0.20 for GREEN)")
    print(f"  max R_H        = {r_max:.4f}")

    # Print full curve
    print(f"\n  σ-curve (ν=0.2):")
    for r in nu_rows:
        sigma = float(r["sigma"])
        r_h = float(r["r_h"])
        bar = "█" * int(r_h * 40)
        print(f"    σ={sigma:.2f}  R_H={r_h:.4f}  {bar}")

    # R_H at σ=1.5 (last entry in sigma grid)
    r_last = float(nu_rows[-1]["r_h"])
    spread = abs(r0 - r_last)

    if r0 > 0.20 and (not mid_rows or r_min_mid < 0.20):
        verdict = "GREEN"
        print(f"\n✅ GREEN — U-shape confirmed (R_H(0)={r0:.3f} > 0.20, "
              f"min_mid={r_min_mid:.3f} < 0.20)")
        print("   → Proceed to Phase 2.")
    elif r0 < 0.10:
        verdict = "YELLOW"
        print(f"\n⚠️  YELLOW — Monotone rising (R_H(0)={r0:.3f} < 0.10)")
        print("   → Try finger-spin as alternative env.")
    elif spread < 0.05:
        verdict = "RED"
        print(f"\n❌ RED — Flat (|R_H(0)-R_H(σ_max)|={spread:.3f} < 0.05)")
        print("   → Metric may be broken; investigate before proceeding.")
    else:
        # R_H(0) high, mid values don't dip below 0.20 — needs more training
        verdict = "NEEDS_TRAINING"
        print(f"\n⏳ NEEDS_TRAINING — R_H(0)={r0:.3f}>0.20 but min_mid={r_min_mid:.3f}")
        print(f"   spread={spread:.3f} ≥ 0.05 so metric is working,")
        print("   but WM needs more training to show U-shape.")
        print("   → Re-run with 100K steps (train_ensemble.py default).")
    return verdict


def check_gate_1_2(delta_dirs: list[Path]) -> str:
    """Check Phase 1.2: cheetah-run OOD bandwidth collapse gate.

    Returns one of: 'GREEN', 'YELLOW', 'SIGNAL_PRESENT', 'NEEDS_TRAINING', 'RED'.

    Amendment D.2 decision table:
      GREEN          — bandwidth > 0 AND collapses with δ
      YELLOW         — bandwidth monotone decrease but not fully collapsed
      SIGNAL_PRESENT — bandwidth = 0 for all δ BUT R_H(σ=0) grows monotonically
                       with δ → acceptable for Phase 2 (Fig 3a reformulated)
      NEEDS_TRAINING — bandwidth = 0 AND R_H at σ=0 is flat in δ AND
                       in-distribution R_H > α (WM simply undertrained)
      RED            — bandwidth = 0 AND R_H is flat in δ (OOD wrapper
                       not affecting WM even with a trained model)
    """
    import csv as _csv

    widths: list[tuple[float, float]] = []
    # R_H at sigma=0 for each delta — used to detect SIGNAL_PRESENT vs RED
    rh_at_sigma0_per_delta: list[tuple[float, float]] = []  # (delta, rh)
    min_r_h_delta0: float | None = None
    alpha: float = 0.3

    for ddir in sorted(delta_dirs):
        json_path = Path(ddir) / "bandwidth.json"
        if not json_path.exists():
            print(f"Warning: missing {json_path}")
            continue
        meta = json.loads(json_path.read_text())
        bw = meta.get("bandwidth", {})
        delta = meta.get("ood_mass_friction_delta", None)
        alpha = float(meta.get("alpha", 0.3))
        # Try to get bandwidth at nu=0.2
        nu_bw = bw.get("0.2", None) or bw.get(0.2, None)
        if nu_bw is None and bw:
            nu_bw = list(bw.values())[0]
        if nu_bw is not None and delta is not None:
            widths.append((float(delta), float(nu_bw)))

            # Read CSV for this delta — get R_H at sigma=0 (first row)
            csv_path = Path(ddir) / "bandwidth_curve.csv"
            if csv_path.exists():
                rows: list[dict] = []
                with open(csv_path, newline="") as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
                if rows:
                    rh_sigma0 = float(rows[0]["r_h"])
                    rh_at_sigma0_per_delta.append((float(delta), rh_sigma0))
                    if abs(float(delta)) < 0.01:
                        min_r_h_delta0 = min(float(r["r_h"]) for r in rows)

    if not widths:
        print("ERROR: no valid calibration files found")
        return "RED"

    widths.sort(key=lambda x: x[0])
    rh_at_sigma0_per_delta.sort(key=lambda x: x[0])

    print(f"\n=== Phase 1.2: Cheetah-run OOD bandwidth collapse gate ===")
    for delta, bw in widths:
        bar = "█" * int(bw * 40)
        print(f"  δ={delta:.1f}  B_{{H,α}}={bw:.4f}  {bar}")

    bw_delta0 = widths[0][1]
    bw_delta_max = widths[-1][1]

    print(f"\n  B(δ=0)   = {bw_delta0:.4f}")
    print(f"  B(δ_max) = {bw_delta_max:.4f}")

    # R_H at sigma=0 across deltas
    if rh_at_sigma0_per_delta:
        print(f"\n  R_H(σ=0) by δ (OOD signal):")
        for d, rh in rh_at_sigma0_per_delta:
            bar = "█" * int(rh * 20)
            print(f"    δ={d:.1f}  R_H(σ=0)={rh:.4f}  {bar}")

    if min_r_h_delta0 is not None:
        print(f"\n  min R_H(δ=0, all σ) = {min_r_h_delta0:.4f}  (need ≤ α={alpha:.2f} for bandwidth > 0)")

    is_decreasing_bw = all(
        widths[i][1] >= widths[i + 1][1] for i in range(len(widths) - 1)
    )

    # Is R_H(sigma=0) monotone-increasing with delta?
    rh_vals = [rh for _, rh in rh_at_sigma0_per_delta]
    rh_monotone_increasing = (
        len(rh_vals) >= 2
        and all(rh_vals[i] <= rh_vals[i + 1] for i in range(len(rh_vals) - 1))
    )
    rh_range = max(rh_vals) - min(rh_vals) if rh_vals else 0.0

    # ── Verdict logic (Amendment D.2) ────────────────────────────────────
    if bw_delta0 > 0.2 and bw_delta_max < 0.1:
        verdict = "GREEN"
        print(f"\n✅ GREEN — Bandwidth collapses with OOD "
              f"(B(0)={bw_delta0:.3f} > 0.20, B(max)={bw_delta_max:.3f} < 0.10)")
        print("   → Fig 3a achievable directly; proceed to Phase 9.")

    elif is_decreasing_bw and bw_delta0 > bw_delta_max:
        verdict = "YELLOW"
        print(f"\n⚠️  YELLOW — Monotone bandwidth decrease but not collapsed "
              f"(B(0)={bw_delta0:.3f}, B(max)={bw_delta_max:.3f})")
        print("   → Extend δ-grid to [0, 1.5] in Phase 9.")

    elif bw_delta0 == 0.0 and bw_delta_max == 0.0 and rh_monotone_increasing and rh_range >= 0.03:
        # All bandwidths zero BUT R_H at sigma=0 grows monotonically with delta.
        # OOD wrapper IS affecting WM; bandwidth metric just can't produce a finite
        # width because R_H never drops below alpha. Paper-relevant signal present.
        verdict = "SIGNAL_PRESENT"
        rh0 = rh_at_sigma0_per_delta[0][1] if rh_at_sigma0_per_delta else float("nan")
        rh_end = rh_at_sigma0_per_delta[-1][1] if rh_at_sigma0_per_delta else float("nan")
        print(f"\n🟡 SIGNAL_PRESENT — All B=0.0, but R_H(σ=0) grows with δ "
              f"({rh0:.3f} → {rh_end:.3f}, range={rh_range:.3f})")
        print("   OOD perturbation affects WM risk monotonically.")
        print("   → Fig 3a reformulated: y-axis = R_H(σ=0) vs δ (not bandwidth).")
        print("   → SIGNAL_PRESENT is acceptable for proceeding to Phase 2.")
        print("   → Full bandwidth collapse deferred to ensemble Phase 9.")

    elif bw_delta0 == 0.0 and bw_delta_max == 0.0 and min_r_h_delta0 is not None and min_r_h_delta0 > alpha:
        # All bandwidths zero AND even in-distribution R_H never drops below alpha.
        # Model is too undertrained for any signal to emerge.
        verdict = "NEEDS_TRAINING"
        print(f"\n⏳ NEEDS_TRAINING — All B=0.0 because in-distribution "
              f"min R_H={min_r_h_delta0:.3f} > α={alpha:.2f}")
        print(f"   WM hasn't learned calibrated dynamics yet (R_H flat across δ and σ).")
        print(f"   → Re-run with 100K training steps (production config).")

    else:
        verdict = "RED"
        print(f"\n❌ RED — No bandwidth shrinkage and R_H flat in δ "
              f"(B(0)={bw_delta0:.3f}, B(max)={bw_delta_max:.3f}, "
              f"R_H range={rh_range:.3f})")
        print("   → Mass/friction perturbation not affecting WM risk. "
              "Consider obs_corruption or action_delay.")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Phase 1 smoke-gate acceptance.")
    ap.add_argument("--gate", choices=["1.1", "1.2"], required=True)
    # Phase 1.1
    ap.add_argument("--csv", type=Path, help="Path to bandwidth_curve.csv (Phase 1.1).")
    # Phase 1.2
    ap.add_argument("--delta-dirs", type=Path, nargs="+",
                    help="Calibration output dirs for each delta (Phase 1.2).")
    ap.add_argument("--ood-dir", type=Path,
                    help="Directory to auto-discover delta calibration dirs.")
    ap.add_argument("--ood-prefix", type=str, default="cheetah_ood_pilot_delta",
                    help="Prefix for auto-discovered calibration dirs.")
    args = ap.parse_args()

    if args.gate == "1.1":
        if args.csv is None:
            ap.error("--csv required for gate 1.1")
        verdict = check_gate_1_1(args.csv)
    else:  # 1.2
        if args.delta_dirs:
            dirs = args.delta_dirs
        elif args.ood_dir:
            dirs = sorted(args.ood_dir.glob(f"{args.ood_prefix}*"))
        else:
            ap.error("Either --delta-dirs or --ood-dir required for gate 1.2")
        verdict = check_gate_1_2(dirs)

    print(f"\nGate {args.gate} verdict: {verdict}")


if __name__ == "__main__":
    main()
