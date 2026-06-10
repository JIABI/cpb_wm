"""Discover and register the dreamerv3-torch repo on sys.path.

Search order:
  1. NMS_DREAMERV3_PATH env var (explicit override)
  2. Sibling of the nms repo root (common monorepo layout)
  3. ~/PyCharmMiscProject/dreamerv3-torch (typical macOS dev layout)
  4. ~/dreamerv3-torch (home-directory fallback)

If none found, DREAMERV3_AVAILABLE is False and callers must either raise or
check NMS_ALLOW_MINIMAL_FALLBACK.

Validated against NM512/dreamerv3-torch commit 6ef8646
(https://github.com/NM512/dreamerv3-torch, 2024).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_and_register_dreamerv3() -> Path | None:
    candidates: list[Path] = []

    if env_path := os.environ.get("NMS_DREAMERV3_PATH"):
        candidates.append(Path(env_path).expanduser())

    # This file lives at:  <nms_repo>/src/nms/wm/_dreamerv3_path.py
    # repo root = four parents up
    _here = Path(__file__).resolve()
    repo_root = _here.parent.parent.parent.parent
    candidates.append(repo_root.parent / "dreamerv3-torch")

    # Typical macOS PyCharm project layout
    candidates.append(Path.home() / "PyCharmMiscProject" / "dreamerv3-torch")

    # Plain home-dir fallback
    candidates.append(Path.home() / "dreamerv3-torch")

    for path in candidates:
        if (path / "dreamer.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    return None


DREAMERV3_PATH: Path | None = find_and_register_dreamerv3()
DREAMERV3_AVAILABLE: bool = DREAMERV3_PATH is not None

# Pinned commit SHA of NM512/dreamerv3-torch used during Phase C integration.
# Any checkout at this SHA (or later) should be compatible.
DREAMERV3_COMMIT_SHA: str = "6ef8646"
