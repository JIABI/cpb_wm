"""Amendment D.1: Aggregate OOD signal CSV writer.

Reads all cheetah OOD calibration output directories and writes a
single CSV summarising per-delta R_H values and bandwidth widths.
This CSV is written unconditionally after every Phase 1.2 calibration
sweep, regardless of bandwidth status.

The resulting ``cheetah_ood_signal.csv`` is paper-relevant evidence for
Fig 3a panel design — even when bandwidth = 0.0 for all deltas.

Usage::

    python scripts/aggregate_ood_signal.py \\
        --ood-dir results/runs/exp2/_smoke \\
        --out results/runs/exp2/_smoke/cheetah_ood_signal.csv

    # Or with explicit delta list:
    python scripts/aggregate_ood_signal.py \\
        --ood-dir results/runs/exp2/_smoke \\
        --ood-prefix cheetah_ood_pilot_delta \\
        --deltas 0.0 0.2 0.4 0.6 0.8 \\
        --out results/runs/exp2/_smoke/cheetah_ood_signal.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def aggregate(
    ood_dir: Path,
    ood_prefix: str = "cheetah_ood_pilot_delta",
    deltas: list[float] | None = None,
) -> list[dict]:
    """Collect per-delta R_H + bandwidth stats from calibration dirs.

    Searches ``ood_dir`` for subdirectories matching ``ood_prefix``.
    Returns rows sorted by delta.
    """
    rows: list[dict] = []

    if deltas is not None:
        dirs = [ood_dir / f"{ood_prefix}{d:.1f}.json" for d in sorted(deltas)]
    else:
        dirs = sorted(ood_dir.glob(f"{ood_prefix}*.json"))

    for ddir in dirs:
        ddir = Path(ddir)
        bw_path = ddir / "bandwidth.json"
        csv_path = ddir / "bandwidth_curve.csv"
        if not bw_path.exists() or not csv_path.exists():
            continue

        bw_meta = json.loads(bw_path.read_text())
        delta = bw_meta.get("ood_mass_friction_delta", None)
        if delta is None:
            # Try to parse from dir name
            stem = ddir.name
            try:
                delta = float(stem.replace(ood_prefix, "").replace(".json", ""))
            except ValueError:
                continue

        curve: list[dict] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                curve.append(row)

        if not curve:
            continue

        rh_at_sigma0 = float(curve[0]["r_h"])
        rh_at_sigma_max = float(curve[-1]["r_h"])
        rh_min = min(float(r["r_h"]) for r in curve)
        rh_max = max(float(r["r_h"]) for r in curve)

        bw_dict = bw_meta.get("bandwidth", {})
        bw_width = list(bw_dict.values())[0] if bw_dict else 0.0

        rows.append({
            "delta": float(delta),
            "rh_at_sigma_0": rh_at_sigma0,
            "rh_at_sigma_max": rh_at_sigma_max,
            "rh_min_over_sigma": rh_min,
            "rh_max_over_sigma": rh_max,
            "bandwidth_width": float(bw_width),
        })

    rows.sort(key=lambda r: r["delta"])
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "delta", "rh_at_sigma_0", "rh_at_sigma_max",
        "rh_min_over_sigma", "rh_max_over_sigma", "bandwidth_width",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print(f"{'δ':>5}  {'R_H(σ=0)':>10}  {'R_H(σ_max)':>10}  {'R_H_min':>8}  {'B_width':>8}")
    print("-" * 55)
    for r in rows:
        print(
            f"  {r['delta']:.1f}  "
            f"{r['rh_at_sigma_0']:>10.4f}  "
            f"{r['rh_at_sigma_max']:>10.4f}  "
            f"{r['rh_min_over_sigma']:>8.4f}  "
            f"{r['bandwidth_width']:>8.4f}"
        )

    if rows:
        rh0_vals = [r["rh_at_sigma_0"] for r in rows]
        is_monotone = all(rh0_vals[i] <= rh0_vals[i + 1] for i in range(len(rh0_vals) - 1))
        rh_range = max(rh0_vals) - min(rh0_vals)
        print(f"\n  R_H(σ=0) range:    {min(rh0_vals):.4f} → {max(rh0_vals):.4f}  (Δ={rh_range:.4f})")
        print(f"  R_H(σ=0) monotone: {is_monotone}")
        print(f"  Any bandwidth > 0: {any(r['bandwidth_width'] > 0 for r in rows)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Phase 1.2 OOD signal.")
    ap.add_argument("--ood-dir", type=Path,
                    default=Path("results/runs/exp2/_smoke"))
    ap.add_argument("--ood-prefix", type=str, default="cheetah_ood_pilot_delta")
    ap.add_argument("--deltas", type=float, nargs="+", default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("results/runs/exp2/_smoke/cheetah_ood_signal.csv"))
    args = ap.parse_args()

    rows = aggregate(args.ood_dir, args.ood_prefix, args.deltas)

    if not rows:
        print("ERROR: no calibration data found in", args.ood_dir)
        return

    print(f"\n=== Phase 1.2 OOD signal aggregate ({len(rows)} delta levels) ===\n")
    print_summary(rows)

    write_csv(rows, args.out)
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
