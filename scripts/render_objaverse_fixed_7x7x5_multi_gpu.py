#!/usr/bin/env python3
"""Launch the fixed 7x7x5, power-0.6 renderer on multiple GPUs."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_objaverse_fixed_10x10x5_multi_gpu as launcher  # noqa: E402


launcher.WORKER = SCRIPT_DIR / "render_objaverse_fixed_7x7x5_dataset.py"
original_read_scene_ids = launcher.read_scene_ids


def pop_int_option(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires an integer value")
    try:
        value = int(sys.argv[index + 1])
    except ValueError as exc:
        raise SystemExit(f"{name} requires an integer value") from exc
    del sys.argv[index : index + 2]
    return value


SCENE_ID_MIN = pop_int_option("--scene-id-min", 0)
SCENE_ID_MAX = pop_int_option("--scene-id-max", 2**31 - 1)
if SCENE_ID_MIN < 0 or SCENE_ID_MAX < SCENE_ID_MIN:
    raise SystemExit("scene-id range must satisfy 0 <= --scene-id-min <= --scene-id-max")


def read_scene_ids(path: Path, manifest_count: int) -> list[str]:
    scene_ids = original_read_scene_ids(path, manifest_count)
    selected = [
        scene_id
        for scene_id in scene_ids
        if SCENE_ID_MIN <= int(scene_id.rsplit("_", 1)[1]) <= SCENE_ID_MAX
    ]
    if not selected:
        raise SystemExit(
            f"No scene ids from {path} are within inclusive range "
            f"[{SCENE_ID_MIN}, {SCENE_ID_MAX}]"
        )
    print(
        f"[scene-filter] inclusive_range=[{SCENE_ID_MIN}, {SCENE_ID_MAX}] "
        f"selected={len(selected)}/{len(scene_ids)} first={selected[0]} last={selected[-1]}",
        flush=True,
    )
    return selected


launcher.base.read_scene_ids = read_scene_ids


if __name__ == "__main__":
    raise SystemExit(launcher.base.main())
