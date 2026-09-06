#!/usr/bin/env python3
"""Render a configurable fixed upper-half light grid using the fixed renderer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_objaverse_fixed_dataset as base  # noqa: E402


def pop_option(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def parse_grid(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise SystemExit("--grid-size must look like 7x7x5") from exc
    if len(result) != 3 or any(part < 1 for part in result):
        raise SystemExit("--grid-size must contain three positive integers")
    return result  # type: ignore[return-value]


GRID_X, GRID_Y, GRID_Z = parse_grid(pop_option("--grid-size", "7x7x5"))
POWER_SCALE = float(pop_option("--grid-power-scale", "0.6"))
GRID_COUNT = GRID_X * GRID_Y * GRID_Z
GRID_NAME = f"{GRID_X}x{GRID_Y}x{GRID_Z}"


def cell_centers(count: int, lower: float, upper: float) -> list[float]:
    step = (upper - lower) / count
    return [lower + (index + 0.5) * step for index in range(count)]


X_VALUES = cell_centers(GRID_X, -1.0, 1.0)
Y_VALUES = cell_centers(GRID_Y, -1.0, 1.0)
Z_VALUES = cell_centers(GRID_Z, 0.0, 1.0)


def fixed_upper_half_grid_candidates(power_values: list[float], seed: int) -> list[dict]:
    del power_values, seed
    candidates = []
    for ix, x in enumerate(X_VALUES):
        for iy, y in enumerate(Y_VALUES):
            for iz, z in enumerate(Z_VALUES):
                candidates.append({
                    "position_id": len(candidates),
                    "canonical_position": [x, y, z],
                    "candidate_source": f"fixed_upper_half_grid_{GRID_NAME}",
                    "grid_cell": [ix, iy, iz],
                    "grid_resolution": [GRID_X, GRID_Y, GRID_Z],
                    "power_scale": POWER_SCALE,
                })
    return candidates


def patch_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [patch_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    patched = {key: patch_metadata(item) for key, item in value.items()}
    sampling = patched.get("sampling")
    if isinstance(sampling, dict) and sampling.get("position_policy", "").startswith("fixed canonical upper-half"):
        count = int(sampling.get("position_count", 0))
        sampling.update({
            "position_candidate_count": GRID_COUNT,
            "rejected_position_count": GRID_COUNT - count,
            "position_policy": f"fixed canonical upper-half {GRID_NAME} grid centers; reject invalid candidates; no refill",
            "position_axes": {"x": X_VALUES, "y": Y_VALUES, "z": Z_VALUES},
            "power_values": [POWER_SCALE],
            "power_policy": f"fixed white-light power scale {POWER_SCALE} for every candidate",
        })
    settings = patched.get("settings")
    if isinstance(settings, dict) and settings.get("fixed_upper_half_white_grid"):
        settings.update({
            "position_candidate_count": GRID_COUNT,
            "fixed_grid_resolution": [GRID_X, GRID_Y, GRID_Z],
            "fixed_grid_region": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 1.0]},
        })
    return patched


original_write_json = base.write_json
original_selected_scene_specs = base.selected_scene_specs


def write_json(path: Path, data: dict) -> None:
    original_write_json(path, patch_metadata(data))


def selected_scene_specs(args, root: Path):
    source_root, specs = original_selected_scene_specs(args, root)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?").split(",")[0]
    return source_root, tqdm(specs, total=len(specs), desc=f"GPU {gpu} scenes", unit="scene", dynamic_ncols=True, mininterval=1.0)


def force_fixed_cli() -> None:
    if "--fixed-upper-half-white-grid" not in sys.argv:
        sys.argv.append("--fixed-upper-half-white-grid")
    if "--power-values" in sys.argv:
        start = sys.argv.index("--power-values")
        end = start + 1
        while end < len(sys.argv) and not sys.argv[end].startswith("--"):
            end += 1
        del sys.argv[start:end]
    sys.argv.extend(["--power-values", *([str(POWER_SCALE)] * 4)])


base.fixed_upper_half_grid_candidates = fixed_upper_half_grid_candidates
base.write_json = write_json
base.selected_scene_specs = selected_scene_specs


if __name__ == "__main__":
    force_fixed_cli()
    raise SystemExit(base.main())
