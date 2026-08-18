#!/usr/bin/env python3
"""Launch the fixed 10x10x5, power-0.6 renderer on multiple GPUs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_objaverse_fixed_multi_gpu as base  # noqa: E402


WORKER = SCRIPT_DIR / "render_objaverse_fixed_10x10x5_dataset.py"


def read_scene_ids(path: Path, manifest_count: int) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Object scene list does not exist: {path}")
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise SystemExit(f"JSON scene list must be an array: {path}")
        tokens = values
    else:
        tokens = [
            line.strip().split()[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    scene_ids = []
    seen = set()
    for token in tokens:
        text = str(token)
        try:
            index = int(text.split("_", 1)[1]) if text.startswith("scene_") else int(text)
        except (ValueError, IndexError) as exc:
            raise SystemExit(f"Invalid scene id in {path}: {token}") from exc
        if index < 0 or index >= manifest_count:
            raise SystemExit(f"Scene id is outside the {manifest_count}-row manifest: {token}")
        scene_id = f"scene_{index:06d}"
        if scene_id not in seen:
            seen.add(scene_id)
            scene_ids.append(scene_id)
    if not scene_ids:
        raise SystemExit(f"Object scene list is empty: {path}")
    return scene_ids


original_worker_command = base.worker_command


def worker_command(args, item, manifest: Path, output_root: Path) -> list[str]:
    args.power_values = [0.6, 0.6, 0.6, 0.6]
    command = original_worker_command(args, item, manifest, output_root)
    base_worker = str(SCRIPT_DIR / "render_objaverse_fixed_dataset.py")
    return [str(WORKER) if token == base_worker else token for token in command]


base.read_scene_ids = read_scene_ids
base.worker_command = worker_command


if __name__ == "__main__":
    raise SystemExit(base.main())
