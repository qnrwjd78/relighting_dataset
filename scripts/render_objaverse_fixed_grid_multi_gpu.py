#!/usr/bin/env python3
"""Launch the configurable fixed-grid renderer on multiple GPUs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_objaverse_fixed_multi_gpu as base  # noqa: E402


def pop_option(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


GRID_SIZE = pop_option("--grid-size", "7x7x5")
GRID_POWER_SCALE = pop_option("--grid-power-scale", "0.6")
SCENE_ID_MIN = int(pop_option("--scene-id-min", "0"))
SCENE_ID_MAX = int(pop_option("--scene-id-max", str(2**31 - 1)))
if SCENE_ID_MIN < 0 or SCENE_ID_MAX < SCENE_ID_MIN:
    raise SystemExit("scene range must satisfy 0 <= min <= max")


def read_scene_ids(path: Path, manifest_count: int) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Object scene list does not exist: {path}")
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise SystemExit(f"JSON scene list must be an array: {path}")
        tokens = values
    else:
        tokens = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    result = []
    for token in tokens:
        text = str(token)
        index = int(text.split("_", 1)[1]) if text.startswith("scene_") else int(text)
        if not 0 <= index < manifest_count:
            raise SystemExit(f"Scene id is outside the {manifest_count}-row manifest: {token}")
        scene_id = f"scene_{index:06d}"
        if SCENE_ID_MIN <= index <= SCENE_ID_MAX and scene_id not in result:
            result.append(scene_id)
    if not result:
        raise SystemExit(f"No scene ids in inclusive range [{SCENE_ID_MIN}, {SCENE_ID_MAX}]")
    print(f"[scene-filter] selected={len(result)} first={result[0]} last={result[-1]}", flush=True)
    return result


original_worker_command = base.worker_command


def worker_command(args, item, manifest: Path, output_root: Path) -> list[str]:
    args.power_values = [float(GRID_POWER_SCALE)] * 4
    command = original_worker_command(args, item, manifest, output_root)
    base_worker = str(SCRIPT_DIR / "render_objaverse_fixed_dataset.py")
    worker = str(SCRIPT_DIR / "render_objaverse_fixed_grid_dataset.py")
    command = [worker if token == base_worker else token for token in command]
    command.extend(["--grid-size", GRID_SIZE, "--grid-power-scale", GRID_POWER_SCALE])
    return command


base.read_scene_ids = read_scene_ids
base.worker_command = worker_command


if __name__ == "__main__":
    raise SystemExit(base.main())
