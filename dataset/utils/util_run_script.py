from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "configs").exists() and (path / "tokenlight_dataset").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root()


def run(cmd: list[str]) -> int:
    print("[RunScript] " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return 0
