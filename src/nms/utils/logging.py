from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import wandb


def _to_plain_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "items"):
        return dict(config)
    return {"config": str(config)}


def init_wandb(project: str, mode: str, config: Any) -> Any:
    """Initialize wandb; in disabled mode returns None."""
    if mode == "disabled":
        return None
    return wandb.init(project=project, mode=mode, config=_to_plain_dict(config))


def log_scalar_local(csv_path: Path, step: int, key: str, value: float) -> None:
    """Append one scalar record to a local CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["step", "key", "value"])
        if header_needed:
            writer.writeheader()
        writer.writerow({"step": step, "key": key, "value": value})


def write_run_info(output_dir: Path, config_text: str) -> Path:
    """Persist git SHA, full config text, and pip freeze into run_info.json.

    Gracefully degrades when git or pip are unavailable (e.g., in a zip-extracted
    directory that has no .git, or in a restricted environment).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        sha = "unknown"

    try:
        pip_freeze = subprocess.check_output(
            ["pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        try:
            pip_freeze = subprocess.check_output(
                ["python", "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
            ).splitlines()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pip_freeze = []

    payload = {"git_sha": sha, "config": config_text, "pip_freeze": pip_freeze}
    out_path = output_dir / "run_info.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
