#!/usr/bin/env python3
"""Render two fixed32 white-light power variants at all 4x4x2 positions.

Each scene gets one HDRI shared by all samples.  Two scene-specific point-light
power configurations are sampled from [0.3, 0.6, 0.9, 1.2] and repeated at every
fixed position, giving exactly 64 position samples per scene.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_objaverse_fixed_dataset as base  # noqa: E402
import render_objaverse_random_dataset as relight  # noqa: E402


POSITION_COUNT = 32
VARIANT_COUNT = 2
SAMPLE_COUNT = POSITION_COUNT * VARIANT_COUNT
FIXED32_POWER_VALUES = [0.3, 0.6, 0.9, 1.2]
current_scene_number = 0
selected_light_configs: list[dict] = []


def fixed_candidates(_power_values: list[float], seed: int) -> list[dict]:
    del seed
    xy_values = [-0.75, -0.25, 0.25, 0.75]
    z_values = [0.25, 0.75]
    if len(selected_light_configs) != VARIANT_COUNT:
        raise RuntimeError("Two source light configurations were not selected for the current scene")

    candidates = []
    for ix, x in enumerate(xy_values):
        for iy, y in enumerate(xy_values):
            for iz, z in enumerate(z_values):
                grid_position_id = (ix * 4 + iy) * 2 + iz
                for variant in selected_light_configs:
                    sample_id = grid_position_id * VARIANT_COUNT + variant["light_variant_id"]
                    candidates.append(
                        {
                            "position_id": sample_id,
                            "canonical_position": [x, y, z],
                            "candidate_source": "fixed_upper_half_grid_4x4x2_random2",
                            "grid_cell": [ix, iy, iz],
                            "grid_resolution": [4, 4, 2],
                            **variant,
                        }
                    )
    return candidates


original_render_scene = base.render_scene
original_write_json = base.write_json


def render_scene(args, source_root, scene_id, output_root, root, asset_override=None):
    global current_scene_number, selected_light_configs
    current_scene_number = int(scene_id.rsplit("_", 1)[1])
    if source_root is None:
        raise RuntimeError("fixed32 random2 requires --scene-set and a replay --source-root")
    selection_rng = random.Random(int(args.seed) + current_scene_number * 1009 + 64002)
    selected_powers = selection_rng.sample(FIXED32_POWER_VALUES, VARIANT_COUNT)
    selected_light_configs = [
        {
            "light_variant_id": variant_id,
            "light_config_source_task": "fixed32_power_set",
            "light_config_source_name": f"power_{power_scale:.1f}",
            "render_color": [1.0, 1.0, 1.0],
            "power_scale": power_scale,
        }
        for variant_id, power_scale in enumerate(selected_powers)
    ]

    # This exact-64 variant intentionally renders every canonical grid cell.
    # It does not drop cells that overlap the object bbox/receiver bounds or
    # fail the brightness pre-check.
    base.object_bbox_reject = lambda *_args, **_kwargs: (True, None)
    relight.point_inside_receiver_bounds = lambda *_args, **_kwargs: (True, None)
    relight.validate_point_light_component = lambda *_args, **_kwargs: (
        True,
        {"valid": True, "skip_reason": None, "policy": "force_all_fixed32_random2"},
    )
    return original_render_scene(args, source_root, scene_id, output_root, root, asset_override)


def patch_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [patch_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: patch_metadata(item) for key, item in value.items()}
    sampling = result.get("sampling")
    if isinstance(sampling, dict) and sampling.get("position_policy", "").startswith("fixed canonical upper-half"):
        lights = [row for row in result.get("samples", []) if row.get("task") == "position"]
        configs = {}
        for row in lights:
            light = row.get("light", {})
            variant_id = int(light.get("id", 0)) % VARIANT_COUNT
            configs.setdefault(
                variant_id,
                {
                    "light_variant_id": variant_id,
                    "source_task": light.get("light_config_source_task"),
                    "source_name": light.get("light_config_source_name"),
                    "render_color": light.get("render_color"),
                    "power_scale": light.get("power_scale"),
                },
            )
        sampling.update(
            {
                "position_candidate_count": SAMPLE_COUNT,
                "position_count": len(lights),
                "rejected_position_count": SAMPLE_COUNT - len(lights),
                "grid_position_count": POSITION_COUNT,
                "light_variants_per_position": VARIANT_COUNT,
                "random_light_configs": [configs[key] for key in sorted(configs)],
                "position_policy": "all fixed canonical upper-half 4x4x2 cells x two scene-random light configurations; no rejection",
                "power_policy": "select two values per scene from fixed32 power set [0.3, 0.6, 0.9, 1.2]",
                "color_policy": "fixed white RGB [1,1,1] for both variants and all 32 positions",
            }
        )
    settings = result.get("settings")
    if isinstance(settings, dict) and settings.get("fixed_upper_half_white_grid"):
        settings.update({"position_candidate_count": SAMPLE_COUNT, "grid_position_count": POSITION_COUNT, "light_variants_per_position": VARIANT_COUNT})
    return result


def write_json(path: Path, data: dict) -> None:
    original_write_json(path, patch_metadata(data))


base.fixed_upper_half_grid_candidates = fixed_candidates
base.render_scene = render_scene
base.write_json = write_json


if __name__ == "__main__":
    raise SystemExit(base.main())
