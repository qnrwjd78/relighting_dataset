#!/usr/bin/env python3
"""Render Objaverse data with a fixed 10x10x5 upper-half light grid.

This is a thin variant of ``render_objaverse_fixed_dataset.py``.  It keeps the
same scene setup, validation, rendering, masks, and CLI, while replacing the
fixed 4x4x2 candidate grid with 500 cell centers in the canonical cube's upper
half (x/y in [-1, 1], z in [0, 1]).
"""

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


GRID_X = 10
GRID_Y = 10
GRID_Z = 5
GRID_COUNT = GRID_X * GRID_Y * GRID_Z
GRID_NAME = f"{GRID_X}x{GRID_Y}x{GRID_Z}"
FIXED_POWER_SCALE = 0.6


def cell_centers(count: int, lower: float, upper: float) -> list[float]:
    step = (float(upper) - float(lower)) / int(count)
    return [float(lower) + (index + 0.5) * step for index in range(int(count))]


X_VALUES = cell_centers(GRID_X, -1.0, 1.0)
Y_VALUES = cell_centers(GRID_Y, -1.0, 1.0)
Z_VALUES = cell_centers(GRID_Z, 0.0, 1.0)


def fixed_upper_half_grid_candidates(power_values: list[float], seed: int) -> list[dict]:
    del power_values, seed
    powers = [FIXED_POWER_SCALE] * GRID_COUNT

    candidates = []
    position_id = 0
    for ix, x in enumerate(X_VALUES):
        for iy, y in enumerate(Y_VALUES):
            for iz, z in enumerate(Z_VALUES):
                candidates.append(
                    {
                        "position_id": position_id,
                        "canonical_position": [x, y, z],
                        "candidate_source": f"fixed_upper_half_grid_{GRID_NAME}",
                        "grid_cell": [ix, iy, iz],
                        "grid_resolution": [GRID_X, GRID_Y, GRID_Z],
                        "power_scale": powers[position_id],
                    }
                )
                position_id += 1
    return candidates


def patch_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [patch_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value

    patched = {key: patch_metadata(item) for key, item in value.items()}

    sampling = patched.get("sampling")
    if isinstance(sampling, dict) and sampling.get("position_policy", "").startswith("fixed canonical upper-half"):
        position_count = int(sampling.get("position_count", 0))
        sampling.update(
            {
                "position_candidate_count": GRID_COUNT,
                "rejected_position_count": GRID_COUNT - position_count,
                "position_policy": f"fixed canonical upper-half {GRID_NAME} grid centers; reject invalid candidates; no refill",
                "position_axes": {"x": X_VALUES, "y": Y_VALUES, "z": Z_VALUES},
                "power_values": [FIXED_POWER_SCALE],
                "power_policy": f"fixed white-light power scale {FIXED_POWER_SCALE} for every candidate",
            }
        )

    settings = patched.get("settings")
    if isinstance(settings, dict) and settings.get("fixed_upper_half_white_grid"):
        settings.update(
            {
                "position_candidate_count": GRID_COUNT,
                "fixed_grid_resolution": [GRID_X, GRID_Y, GRID_Z],
                "fixed_grid_region": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 1.0]},
            }
        )
    return patched


original_write_json = base.write_json
original_selected_scene_specs = base.selected_scene_specs


def write_json(path: Path, data: dict) -> None:
    original_write_json(path, patch_metadata(data))


def selected_scene_specs(args, root: Path):
    source_root, specs = original_selected_scene_specs(args, root)
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?").split(",")[0]
    progress = tqdm(
        specs,
        total=len(specs),
        desc=f"GPU {visible_gpu} scenes",
        unit="scene",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    return source_root, progress


base.fixed_upper_half_grid_candidates = fixed_upper_half_grid_candidates
base.write_json = write_json
base.selected_scene_specs = selected_scene_specs


def force_fixed_cli() -> None:
    """Keep the inherited four-value parser contract while forcing 0.6."""
    if "--fixed-upper-half-white-grid" not in sys.argv:
        sys.argv.append("--fixed-upper-half-white-grid")
    if "--power-values" in sys.argv:
        start = sys.argv.index("--power-values")
        end = start + 1
        while end < len(sys.argv) and not sys.argv[end].startswith("--"):
            end += 1
        del sys.argv[start:end]
    sys.argv.extend(["--power-values", "0.6", "0.6", "0.6", "0.6"])


if __name__ == "__main__":
    force_fixed_cli()
    raise SystemExit(base.main())
