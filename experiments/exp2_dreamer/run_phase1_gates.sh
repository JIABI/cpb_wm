#!/bin/bash
# Phase 1 smoke-gate calibration pipeline (Amendment v1).
#
# Order per Amendment A/C/D/E:
#   1. Determinism probe on both 100k checkpoints (Amendment A)
#      — STOP if any criterion fails.
#   2. Phase 1.1: fit hopper k-means → calibrate → plot → check (Amendment C.1)
#   3. Phase 1.2: fit cheetah k-means → OOD calibration → signal CSV → check
#      (Amendment D.1 + D.2)
#   4. Write status JSONs for both gates (Amendment E)
#
# Usage:
#   cd /Users/bella/Downloads/nms
#   bash experiments/exp2_dreamer/run_phase1_gates.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "Phase 1 Smoke Gates — Calibration Pipeline (Amendment v1)"
echo "$(date)"
echo "================================================================"

# ── Determine checkpoint step counts ─────────────────────────────────────────
HOPPER_CKPT="results/runs/exp2/_smoke/hopper_dv3_K1/member_0/latest.pt"
CHEETAH_CKPT="results/runs/exp2/_smoke/cheetah_dv3_100K/member_0/latest.pt"

if [ ! -f "$HOPPER_CKPT" ]; then
    echo "ERROR: Hopper checkpoint not found: $HOPPER_CKPT"
    echo "Run train_smoke.py for hopper-hop first."
    exit 1
fi
if [ ! -f "$CHEETAH_CKPT" ]; then
    echo "ERROR: Cheetah checkpoint not found: $CHEETAH_CKPT"
    echo "Run train_smoke.py for cheetah-run first."
    exit 1
fi
echo "Checkpoints found: ✓"

# Read checkpoint step counts from run_info.json if available
HOPPER_STEPS=$(python3 -c "import json; d=json.load(open('results/runs/exp2/_smoke/hopper_dv3_K1/member_0/run_info.json')); print(d.get('total_steps', 100000))" 2>/dev/null || echo "100000")
CHEETAH_STEPS=$(python3 -c "import json; d=json.load(open('results/runs/exp2/_smoke/cheetah_dv3_100K/member_0/run_info.json')); print(d.get('total_steps', 100000))" 2>/dev/null || echo "100000")
echo "Hopper steps: $HOPPER_STEPS   Cheetah steps: $CHEETAH_STEPS"

# ── Amendment A: Determinism probe ───────────────────────────────────────────
echo ""
echo "================================================================"
echo "Amendment A: Determinism probe (mandatory before any RED verdict)"
echo "================================================================"

echo "[A.1] Running determinism probe for hopper-hop..."
python scripts/determinism_probe.py \
    --env hopper-hop \
    --dreamer-dir results/runs/exp2/_smoke/hopper_dv3_K1/ \
    --kmeans-path results/interfaces/hopper_hop_k8.npz \
    --out "results/runs/exp2/_smoke/determinism_probe_hopper_${HOPPER_STEPS}.json"

HOPPER_PROBE_EXIT=$?
if [ $HOPPER_PROBE_EXIT -ne 0 ]; then
    echo ""
    echo "❌ STOP: Hopper determinism probe FAILED."
    echo "   Do NOT retrain or advance to Phase 2."
    echo "   Investigate σ-noise-injection patch before proceeding."
    python scripts/write_phase_status.py \
        --phase "determinism_probe" \
        --verdict "FAIL" \
        --checkpoint-steps "$HOPPER_STEPS" \
        --next-action "stop_for_review" \
        --notes "Hopper determinism probe failed. σ-noise-injection or encoder init bug." \
        --out results/runs/exp2/_status/determinism_probe_100k.json
    exit 1
fi

echo "[A.2] Running determinism probe for cheetah-run..."
python scripts/determinism_probe.py \
    --env cheetah-run \
    --dreamer-dir results/runs/exp2/_smoke/cheetah_dv3_100K/ \
    --kmeans-path results/interfaces/cheetah_run_k8.npz \
    --out "results/runs/exp2/_smoke/determinism_probe_cheetah_${CHEETAH_STEPS}.json"

CHEETAH_PROBE_EXIT=$?
if [ $CHEETAH_PROBE_EXIT -ne 0 ]; then
    echo ""
    echo "❌ STOP: Cheetah determinism probe FAILED."
    python scripts/write_phase_status.py \
        --phase "determinism_probe" \
        --verdict "FAIL" \
        --checkpoint-steps "$CHEETAH_STEPS" \
        --next-action "stop_for_review" \
        --notes "Cheetah determinism probe failed." \
        --out results/runs/exp2/_status/determinism_probe_100k.json
    exit 1
fi

echo "✅ Both determinism probes PASSED."
python scripts/write_phase_status.py \
    --phase "determinism_probe" \
    --verdict "PASS" \
    --checkpoint-steps "$HOPPER_STEPS" \
    --next-action "advance" \
    --notes "Hopper + cheetah determinism probes both passed all 4 criteria." \
    --evidence \
        "results/runs/exp2/_smoke/determinism_probe_hopper_${HOPPER_STEPS}.json" \
        "results/runs/exp2/_smoke/determinism_probe_cheetah_${CHEETAH_STEPS}.json" \
    --out results/runs/exp2/_status/determinism_probe_100k.json

# ── Phase 1.1: Hopper-hop — Fit k-means → calibrate → plot → check ───────────
echo ""
echo "================================================================"
echo "Phase 1.1: Hopper-hop U-shape gate (Amendment C.1)"
echo "================================================================"

echo "[1/4] Fitting k-means on hopper-hop trained policy..."
python -m experiments.exp2_dreamer.fit_kmeans \
    --dreamer-dir results/runs/exp2/_smoke/hopper_dv3_K1/member_0 \
    --env hopper-hop \
    --k 8 \
    --n-rollouts 20 \
    --seed 0 \
    --device cpu \
    --out results/interfaces/hopper_hop_k8.npz

echo "[2/4] Calibrating bandwidth for hopper-hop (100k checkpoint)..."
python -m experiments.exp2_dreamer.calibrate_bandwidth \
    --dreamer-dir results/runs/exp2/_smoke/hopper_dv3_K1 \
    --kmeans-path results/interfaces/hopper_hop_k8.npz \
    --env hopper-hop \
    --n-pairs 30 \
    --horizon 5 \
    --alpha 0.3 \
    --sigma-grid-max 1.5 \
    --sigma-grid-n 15 \
    --nu-grid 0.2 \
    --device cpu \
    --out results/runs/exp2/_smoke/hopper_calibration_100k.json

echo "[3/4] Plotting R_H(σ) curve unconditionally (Amendment C.1)..."
python3 -c "
import csv, sys
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

rows = list(csv.DictReader(open('results/runs/exp2/_smoke/hopper_calibration_100k.json/bandwidth_curve.csv')))
sigmas = [float(r['sigma']) for r in rows]
rh = [float(r['r_h']) for r in rows]

print('R_H range:', min(rh), 'to', max(rh))
print('R_H at sigma=0:', rh[0])
mid = [r for s,r in zip(sigmas,rh) if 0.2 <= s <= 0.8]
print('R_H min over sigma in [0.2, 0.8]:', min(mid) if mid else 'N/A')

if HAS_MPL:
    plt.figure(figsize=(6,4))
    plt.plot(sigmas, rh, 'o-', label='ν̂=0.2')
    plt.axhline(0.3, ls='--', color='red', label='α=0.3')
    plt.xlabel('σ'); plt.ylabel('R_H'); plt.title('Hopper-hop 100k smoke')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig('results/runs/exp2/_smoke/hopper_100k_curve.png', dpi=120, bbox_inches='tight')
    print('Saved hopper_100k_curve.png')
else:
    print('matplotlib not available — skipping PNG plot')
"

echo "[4/4] Checking Phase 1.1 acceptance..."
PHASE11_OUTPUT=$(python -m experiments.exp2_dreamer.check_phase1 \
    --gate 1.1 \
    --csv results/runs/exp2/_smoke/hopper_calibration_100k.json/bandwidth_curve.csv 2>&1)
echo "$PHASE11_OUTPUT"
PHASE11_VERDICT=$(echo "$PHASE11_OUTPUT" | grep "^Gate 1.1 verdict:" | awk '{print $NF}')
echo ""
echo "Gate 1.1 verdict: $PHASE11_VERDICT"

# Determine next action for Phase 1.1
case "$PHASE11_VERDICT" in
    GREEN)
        PHASE11_NEXT="advance"
        PHASE11_NOTES="U-shape confirmed at ${HOPPER_STEPS} steps. Proceed to Phase 2."
        ;;
    YELLOW)
        PHASE11_NEXT="env_swap"
        PHASE11_NOTES="Monotone rising (cheetah-like). Try finger-spin as alternative env per Amendment C.3."
        ;;
    RED)
        PHASE11_NEXT="stop_for_review"
        PHASE11_NOTES="Uniformly high R_H persists at ${HOPPER_STEPS} steps. Architecture or metric problem. STOP."
        ;;
    UNCLEAR)
        PHASE11_NEXT="rerun"
        PHASE11_NOTES="No clear shape at ${HOPPER_STEPS} steps. Rerun with --n-pairs 100."
        ;;
    NEEDS_TRAINING)
        PHASE11_NEXT="env_swap"
        PHASE11_NOTES="Still NEEDS_TRAINING at ${HOPPER_STEPS} steps. Try env swap (finger-spin) per Amendment C.3."
        ;;
    *)
        PHASE11_NEXT="rerun"
        PHASE11_NOTES="Unrecognised verdict: $PHASE11_VERDICT"
        ;;
esac

python scripts/write_phase_status.py \
    --phase "1.1" \
    --verdict "$PHASE11_VERDICT" \
    --checkpoint-steps "$HOPPER_STEPS" \
    --next-action "$PHASE11_NEXT" \
    --notes "$PHASE11_NOTES" \
    --evidence \
        "results/runs/exp2/_smoke/hopper_calibration_100k.json/bandwidth_curve.csv" \
        "results/runs/exp2/_smoke/hopper_100k_curve.png" \
    --out results/runs/exp2/_status/phase1_1_100k.json

# ── Phase 1.2: Cheetah-run — Fit k-means → OOD calibration → signal CSV ──────
echo ""
echo "================================================================"
echo "Phase 1.2: Cheetah-run OOD bandwidth collapse gate (Amendments D.1 + D.2)"
echo "================================================================"

echo "[0/3] Fitting k-means on cheetah-run trained policy..."
python -m experiments.exp2_dreamer.fit_kmeans \
    --dreamer-dir results/runs/exp2/_smoke/cheetah_dv3_100K/member_0 \
    --env cheetah-run \
    --k 8 \
    --n-rollouts 20 \
    --seed 0 \
    --device cpu \
    --out results/interfaces/cheetah_run_k8.npz

echo "[1/3] Running OOD calibration for delta in {0.0, 0.2, 0.4, 0.6, 0.8}..."
for delta in 0.0 0.2 0.4 0.6 0.8; do
    echo "  δ=${delta} ..."
    python -m experiments.exp2_dreamer.calibrate_bandwidth \
        --dreamer-dir results/runs/exp2/_smoke/cheetah_dv3_100K \
        --kmeans-path results/interfaces/cheetah_run_k8.npz \
        --env cheetah-run \
        --ood-mass-friction "$delta" \
        --n-pairs 30 \
        --horizon 5 \
        --alpha 0.3 \
        --sigma-grid-max 1.5 \
        --sigma-grid-n 10 \
        --nu-grid 0.2 \
        --device cpu \
        --out "results/runs/exp2/_smoke/cheetah_ood_pilot_delta${delta}.json"
done

echo "[2/3] Writing aggregate OOD signal CSV (Amendment D.1)..."
python scripts/aggregate_ood_signal.py \
    --ood-dir results/runs/exp2/_smoke \
    --ood-prefix cheetah_ood_pilot_delta \
    --out results/runs/exp2/_smoke/cheetah_ood_signal.csv

echo "[3/3] Checking Phase 1.2 acceptance (Amendment D.2)..."
PHASE12_OUTPUT=$(python -m experiments.exp2_dreamer.check_phase1 \
    --gate 1.2 \
    --ood-dir results/runs/exp2/_smoke \
    --ood-prefix cheetah_ood_pilot_delta 2>&1)
echo "$PHASE12_OUTPUT"
PHASE12_VERDICT=$(echo "$PHASE12_OUTPUT" | grep "^Gate 1.2 verdict:" | awk '{print $NF}')
echo ""
echo "Gate 1.2 verdict: $PHASE12_VERDICT"

# Determine next action for Phase 1.2
case "$PHASE12_VERDICT" in
    GREEN)
        PHASE12_NEXT="advance"
        PHASE12_NOTES="Bandwidth collapses with OOD at ${CHEETAH_STEPS} steps. Fig 3a achievable directly."
        ;;
    YELLOW)
        PHASE12_NEXT="advance"
        PHASE12_NOTES="Monotone bandwidth decrease at ${CHEETAH_STEPS} steps. Extend delta-grid to [0,1.5] in Phase 9."
        ;;
    SIGNAL_PRESENT)
        PHASE12_NEXT="advance"
        PHASE12_NOTES="SIGNAL_PRESENT at ${CHEETAH_STEPS} steps. R_H(sigma=0) grows with delta. Fig 3a reformulated as R_H vs delta. Proceed to Phase 2."
        ;;
    NEEDS_TRAINING)
        PHASE12_NEXT="rerun"
        PHASE12_NOTES="NEEDS_TRAINING at ${CHEETAH_STEPS} steps. R_H flat in delta. Extend training."
        ;;
    RED)
        PHASE12_NEXT="stop_for_review"
        PHASE12_NOTES="RED at ${CHEETAH_STEPS} steps. OOD wrapper not affecting WM risk. Consider obs_corruption or action_delay."
        ;;
    *)
        PHASE12_NEXT="rerun"
        PHASE12_NOTES="Unrecognised verdict: $PHASE12_VERDICT"
        ;;
esac

python scripts/write_phase_status.py \
    --phase "1.2" \
    --verdict "$PHASE12_VERDICT" \
    --checkpoint-steps "$CHEETAH_STEPS" \
    --next-action "$PHASE12_NEXT" \
    --notes "$PHASE12_NOTES" \
    --evidence \
        "results/runs/exp2/_smoke/cheetah_ood_signal.csv" \
        "results/runs/exp2/_smoke/cheetah_ood_pilot_delta0.0.json/bandwidth_curve.csv" \
    --out results/runs/exp2/_status/phase1_2_100k.json

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 1 pipeline complete."
echo "================================================================"
echo ""
echo "  Determinism probe : PASS"
echo "  Phase 1.1         : $PHASE11_VERDICT  → $PHASE11_NEXT"
echo "  Phase 1.2         : $PHASE12_VERDICT  → $PHASE12_NEXT"
echo ""
echo "Status files:"
echo "  results/runs/exp2/_status/determinism_probe_100k.json"
echo "  results/runs/exp2/_status/phase1_1_100k.json"
echo "  results/runs/exp2/_status/phase1_2_100k.json"
echo ""

# STOP-conditions
if [ "$PHASE11_NEXT" = "stop_for_review" ] || [ "$PHASE12_NEXT" = "stop_for_review" ]; then
    echo "⚠️  STOP-condition triggered on one or more gates."
    echo "   Do NOT advance to Phase 2 without human review."
    exit 2
fi

# Acceptable verdicts for Phase 2 advance: GREEN, YELLOW, SIGNAL_PRESENT on 1.2
PHASE12_OK=$(echo "GREEN YELLOW SIGNAL_PRESENT" | grep -w "$PHASE12_VERDICT" || true)
PHASE11_OK=$(echo "GREEN" | grep -w "$PHASE11_VERDICT" || true)

if [ -n "$PHASE11_OK" ] && [ -n "$PHASE12_OK" ]; then
    echo "✅ Both gates acceptable for Phase 2 advance."
    echo "   Report numerical verdicts to user before starting Phase 2."
else
    echo "⚠️  One or both gates are not yet GREEN."
    echo "   Review verdicts above. See Amendment C.2 and D.2 for decision trees."
fi
