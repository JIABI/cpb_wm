"""Amendment E: Phase-boundary status JSON writer.

Writes auditable status records to ``results/runs/exp2/_status/`` at
each phase gate. These files are later composed into
``paper_evidence_audit.md``.

Usage::

    # From Python:
    from scripts.write_phase_status import write_status
    write_status(
        phase="1.1",
        verdict="GREEN",
        evidence_files=["results/runs/exp2/_smoke/hopper_calibration_100k.json/bandwidth_curve.csv"],
        checkpoint_steps=100000,
        next_action="advance",
        notes="U-shape confirmed at 100K steps.",
    )

    # From CLI:
    python scripts/write_phase_status.py \\
        --phase 1.1 \\
        --verdict GREEN \\
        --checkpoint-steps 100000 \\
        --next-action advance \\
        --notes "U-shape confirmed at 100K steps." \\
        --evidence results/runs/exp2/_smoke/hopper_calibration_100k.json/bandwidth_curve.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STATUS_DIR = Path("results/runs/exp2/_status")

_VALID_VERDICTS = {
    "GREEN", "YELLOW", "RED",
    "NEEDS_TRAINING", "SIGNAL_PRESENT", "UNCLEAR",
    "PASS", "FAIL",
    "DEFERRED",
}
_VALID_NEXT_ACTIONS = {
    "advance", "env_swap", "rerun", "stop_for_review", "deferred",
}


def write_status(
    phase: str,
    verdict: str,
    evidence_files: list[str] | None = None,
    checkpoint_steps: int | None = None,
    next_action: str = "advance",
    notes: str = "",
    extra: dict[str, Any] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write a phase-boundary status record and return its path.

    Parameters
    ----------
    phase:
        Phase identifier, e.g. ``"1.1"``, ``"1.2"``, ``"determinism_probe"``.
    verdict:
        One of the VALID_VERDICTS strings.
    evidence_files:
        List of paths to evidence files (CSV, JSON, PNG).
    checkpoint_steps:
        Training steps at which the verdict was recorded.
    next_action:
        One of the VALID_NEXT_ACTIONS strings.
    notes:
        Free-form annotation.
    extra:
        Additional key-value pairs merged into the record.
    out_path:
        Override output path; defaults to
        ``_STATUS_DIR / f"phase{phase.replace('.', '_')}_{checkpoint_steps//1000}k.json"``.
    """
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"verdict {verdict!r} not in {_VALID_VERDICTS}")
    if next_action not in _VALID_NEXT_ACTIONS:
        raise ValueError(f"next_action {next_action!r} not in {_VALID_NEXT_ACTIONS}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record: dict[str, Any] = {
        "phase": phase,
        "verdict": verdict,
        "evidence_files": evidence_files or [],
        "timestamp": ts,
        "next_action": next_action,
        "notes": notes,
    }
    if checkpoint_steps is not None:
        record["checkpoint_steps"] = checkpoint_steps
    if extra:
        record.update(extra)

    if out_path is None:
        step_tag = f"{checkpoint_steps // 1000}k" if checkpoint_steps else "latest"
        safe_phase = phase.replace(".", "_")
        out_path = _STATUS_DIR / f"phase{safe_phase}_{step_tag}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a phase-boundary status JSON.")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--verdict", required=True, choices=sorted(_VALID_VERDICTS))
    ap.add_argument("--checkpoint-steps", type=int, default=None)
    ap.add_argument("--next-action", default="advance", choices=sorted(_VALID_NEXT_ACTIONS))
    ap.add_argument("--notes", default="")
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    path = write_status(
        phase=args.phase,
        verdict=args.verdict,
        evidence_files=args.evidence,
        checkpoint_steps=args.checkpoint_steps,
        next_action=args.next_action,
        notes=args.notes,
        out_path=args.out,
    )
    print(f"Status written → {path}")
    print(json.dumps(json.loads(path.read_text()), indent=2))


if __name__ == "__main__":
    main()
