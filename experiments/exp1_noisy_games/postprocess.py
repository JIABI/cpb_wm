"""Exp 1 post-processing: add setting_name column and generate all figures.

Three settings are supported:

  exp1a  — self-play baseline (spiral_only, right-censored).
           Adds setting_name, plots bandwidth figure (skips projection panel).

  exp1b  — mixture hero (spiral_or_exploit).
           Adds setting_name, plots bandwidth hero figures (all 4 panels).

  exp1c  — Phase C: horizon sweep → 2D CIR heatmap.
           1. Runs the CIR bandwidth sweep for H=[50,100,400] if not present.
           2. Merges with exp1b (H=200) data.
           3. Plots CIR heatmap via plot_cir.py for each ν in --cir-nus.

Usage:
    # Archive self-play baseline
    python -m experiments.exp1_noisy_games.postprocess --setting exp1a

    # Generate mixture hero figures
    python -m experiments.exp1_noisy_games.postprocess --setting exp1b

    # All three in one pass
    python -m experiments.exp1_noisy_games.postprocess
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Setting definitions ──────────────────────────────────────────────────────

SETTINGS: dict[str, dict[str, object]] = {
    "exp1a": {
        "csv_in": Path("results/runs/exp1a_clean_lower_bound/bandwidth.csv"),
        "setting_name": "self_play_spiral_only",
        "fig1": Path("results/figures/exp1a_bandwidth.pdf"),
        "fig2": Path("results/figures/exp1a_lemma1.pdf"),
        "plot_extra_args": ["--skip-projection-panel"],
        "train_nus": ["0.05", "0.15", "0.30"],
        "test_nus": ["0.10", "0.20"],
    },
    "exp1b": {
        "csv_in": Path("results/runs/exp1b_full_bandwidth/bandwidth.csv"),
        "setting_name": "mixture_spiral_or_exploit",
        "fig1": Path("results/figures/exp1b_bandwidth_hero.pdf"),
        "fig2": Path("results/figures/exp1b_lemma1_proj.pdf"),
        "plot_extra_args": [],
        "train_nus": ["0.05", "0.15", "0.30"],
        "test_nus": ["0.10", "0.20"],
    },
}

# Phase C configuration ─────────────────────────────────────────────────────
EXP1C_CIR_SWEEP_OUT = Path("results/runs/exp1c_cir/bandwidth_cir_extra.csv")
EXP1C_CIR_MERGED = Path("results/runs/exp1c_cir/bandwidth_cir.csv")
EXP1C_CIR_FIG_DIR = Path("results/figures/")
EXP1C_CIR_NUS = [0.05, 0.10, 0.20]  # per-ν CIR heatmaps


def add_setting_name(csv_path: Path, setting_name: str) -> None:
    """Inject a 'setting_name' column into the CSV (idempotent)."""
    df = pd.read_csv(csv_path)
    df["setting_name"] = setting_name
    df.to_csv(csv_path, index=False)
    print(f"  Added setting_name={setting_name!r} → {csv_path}")


def run_plot(setting_name: str, cfg: dict[str, object]) -> int:
    """Call plot_bandwidth.py for one setting; return subprocess returncode."""
    csv_in: Path = cfg["csv_in"]  # type: ignore[assignment]
    fig1: Path = cfg["fig1"]  # type: ignore[assignment]
    fig2: Path = cfg["fig2"]  # type: ignore[assignment]
    extra: list[str] = cfg["plot_extra_args"]  # type: ignore[assignment]
    train_nus: list[str] = cfg["train_nus"]  # type: ignore[assignment]
    test_nus: list[str] = cfg["test_nus"]  # type: ignore[assignment]

    cmd = [
        sys.executable, "-m",
        "experiments.exp1_noisy_games.plot_bandwidth",
        "--csv", str(csv_in),
        "--out-fig1", str(fig1),
        "--out-fig2", str(fig2),
        "--horizon", "200",
        "--alpha", "0.1",
        "--train-nus", *train_nus,
        "--test-nus", *test_nus,
        "--response-form", "best",
        *extra,
    ]
    print(f"\nRunning plot_bandwidth for {setting_name}:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_phase_c(cir_nus: list[float]) -> int:
    """Phase C: run CIR extra sweep, merge, and plot heatmaps.

    Returns 0 on success, non-zero on error.
    """
    # ── Step 1: run the extra horizon sweep (H = 50, 100, 400) ──────────────
    if not EXP1C_CIR_SWEEP_OUT.exists():
        print("\n=== Phase C: running CIR extra sweep (H=50,100,400) ===")
        cmd_sweep = [
            sys.executable, "-m",
            "experiments.exp1_noisy_games.run_bandwidth",
            "--noise-levels", "0.0", "0.05", "0.10", "0.15", "0.20", "0.30",
            "--sigma-min", "0.0", "--sigma-max", "1.0", "--sigma-n", "51",
            "--horizons", "50", "100", "400",
            "--episodes", "1000",
            "--seeds", "5",
            "--opponent-mixture", "reciprocal:0.8,always_defect:0.2",
            "--violation-mode", "spiral_or_exploit",
            "--spiral-len", "15",
            "--exploit-len", "15",
            "--out", str(EXP1C_CIR_SWEEP_OUT),
        ]
        print("  " + " ".join(cmd_sweep))
        rc = subprocess.run(cmd_sweep, check=False).returncode
        if rc != 0:
            print(f"  CIR sweep failed (rc={rc})")
            return rc
    else:
        print(f"\nPhase C: CIR extra sweep already exists → {EXP1C_CIR_SWEEP_OUT}")

    # ── Step 2: merge with exp1b (H=200) ────────────────────────────────────
    exp1b_csv = SETTINGS["exp1b"]["csv_in"]  # type: ignore[index]
    if not exp1b_csv.exists():  # type: ignore[union-attr]
        print(f"  exp1b CSV not found: {exp1b_csv}  [cannot merge]")
        return 1

    print(f"\nPhase C: merging CIR sweep with {exp1b_csv} ...")
    df_extra = pd.read_csv(EXP1C_CIR_SWEEP_OUT)
    df_exp1b = pd.read_csv(exp1b_csv)  # type: ignore[arg-type]
    # Keep only H=200 rows from exp1b (the extra sweep has H=50,100,400)
    if "horizon" in df_exp1b.columns:
        df_h200 = df_exp1b[df_exp1b["horizon"] == 200]
    else:
        df_h200 = df_exp1b.assign(horizon=200)
    merged = pd.concat([df_extra, df_h200], ignore_index=True)
    EXP1C_CIR_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(EXP1C_CIR_MERGED, index=False)
    print(f"  Merged {len(df_extra)} (extra) + {len(df_h200)} (H=200) rows → {EXP1C_CIR_MERGED}")

    # ── Step 3: plot CIR heatmaps ────────────────────────────────────────────
    nu_args = [str(n) for n in cir_nus]
    out_cir = EXP1C_CIR_FIG_DIR / "exp1c_cir.pdf"
    cmd_cir = [
        sys.executable, "-m",
        "experiments.exp1_noisy_games.plot_cir",
        "--csv", str(EXP1C_CIR_MERGED),
        "--nu", *nu_args,
        "--alpha", "0.1",
        "--out", str(out_cir),
    ]
    print("\nRunning plot_cir:")
    print("  " + " ".join(cmd_cir))
    return subprocess.run(cmd_cir, check=False).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 1 post-processing pipeline.")
    ap.add_argument(
        "--setting",
        nargs="+",
        choices=[*list(SETTINGS), "exp1c"],
        default=[*list(SETTINGS), "exp1c"],
        help="Which setting(s) to process (default: all).",
    )
    ap.add_argument(
        "--cir-nus",
        nargs="+",
        type=float,
        default=EXP1C_CIR_NUS,
        help="ν values for CIR heatmap (exp1c only).",
    )
    args = ap.parse_args()

    failed = []
    for name in args.setting:
        print(f"\n=== Processing {name} ===")

        if name == "exp1c":
            rc = run_phase_c(args.cir_nus)
            if rc != 0:
                failed.append(name)
            continue

        cfg = SETTINGS[name]
        csv_path: Path = cfg["csv_in"]  # type: ignore[assignment]

        if not csv_path.exists():
            print(f"  CSV not found: {csv_path}  [SKIP — run run_bandwidth.py first]")
            failed.append(name)
            continue

        add_setting_name(csv_path, str(cfg["setting_name"]))
        rc = run_plot(name, cfg)
        if rc != 0:
            print(f"  plot_bandwidth exited with code {rc}")
            failed.append(name)

        # For exp1b: also run ν̂ sensitivity analysis and plot
        if name == "exp1b":
            nuhat_csv = csv_path.parent / "nu_hat_sensitivity.csv"
            nuhat_fig = Path("results/figures/exp1b_nuhat_sensitivity.pdf")
            if not nuhat_csv.exists():
                print(f"\n  Running ν̂ sensitivity analysis for {name} ...")
                cmd_sens = [
                    sys.executable, "-m",
                    "experiments.exp1_noisy_games.run_nuhat_sensitivity",
                    "--csv", str(csv_path),
                    "--out", str(nuhat_csv),
                    "--horizon", "200", "--alpha", "0.1",
                    "--nu-true", "0.05", "0.10", "0.15", "0.20", "0.30",
                    "--biases", "-0.5", "-0.2", "0.0", "0.2", "0.5",
                    "--episodes", "500", "--seeds", "3",
                    "--opponent-mixture", "reciprocal:0.8,always_defect:0.2",
                    "--violation-mode", "spiral_or_exploit",
                    "--spiral-len", "15", "--exploit-len", "15",
                ]
                print("  " + " ".join(cmd_sens))
                rc_sens = subprocess.run(cmd_sens, check=False).returncode
                if rc_sens != 0:
                    print(f"  run_nuhat_sensitivity exited with code {rc_sens}")
                    failed.append(f"{name}_nuhat")
            else:
                print(f"\n  ν̂ sensitivity CSV already exists → {nuhat_csv}")

            if nuhat_csv.exists():
                cmd_plot = [
                    sys.executable, "-m",
                    "experiments.exp1_noisy_games.plot_nuhat_sensitivity",
                    "--csv", str(nuhat_csv),
                    "--out", str(nuhat_fig),
                    "--alpha", "0.1",
                ]
                print("  " + " ".join(cmd_plot))
                rc_plot = subprocess.run(cmd_plot, check=False).returncode
                if rc_plot != 0:
                    print(f"  plot_nuhat_sensitivity exited with code {rc_plot}")
                    failed.append(f"{name}_nuhat_plot")

    if failed:
        print(f"\nFailed settings: {failed}")
        sys.exit(1)
    print("\nAll settings processed successfully.")


if __name__ == "__main__":
    main()
