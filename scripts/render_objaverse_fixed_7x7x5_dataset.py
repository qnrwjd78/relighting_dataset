#!/usr/bin/env python3
"""Render Objaverse data with a fixed 7x7x5 upper-half light grid."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_objaverse_fixed_10x10x5_dataset as variant  # noqa: E402


variant.GRID_X = 7
variant.GRID_Y = 7
variant.GRID_Z = 5
variant.GRID_COUNT = variant.GRID_X * variant.GRID_Y * variant.GRID_Z
variant.GRID_NAME = f"{variant.GRID_X}x{variant.GRID_Y}x{variant.GRID_Z}"
variant.X_VALUES = variant.cell_centers(variant.GRID_X, -1.0, 1.0)
variant.Y_VALUES = variant.cell_centers(variant.GRID_Y, -1.0, 1.0)
variant.Z_VALUES = variant.cell_centers(variant.GRID_Z, 0.0, 1.0)

# The imported functions resolve these grid constants at call time.
variant.base.fixed_upper_half_grid_candidates = variant.fixed_upper_half_grid_candidates
variant.base.write_json = variant.write_json
variant.base.selected_scene_specs = variant.selected_scene_specs


if __name__ == "__main__":
    variant.force_fixed_cli()
    raise SystemExit(variant.base.main())
