#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final composed PNG samples from completed TokenLight EXR scenes."
    )
    parser.add_argument("--source", default="outputs/objaverse_dataset_exr")
    parser.add_argument("--dest", default="outputs/objaverse_dataset_png")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--two-light-samples", type=int, default=64)
    parser.add_argument("--two-light-min-distance", type=float, default=0.75)
    parser.add_argument("--two-light-min-distance-count", type=int, default=32)
    parser.add_argument("--global-ambient-samples", type=int, default=32)
    parser.add_argument("--ambient-scale-min", type=float, default=0.1)
    parser.add_argument("--ambient-scale-max", type=float, default=1.25)
    parser.add_argument("--intensity-min", type=float, default=0.25)
    parser.add_argument("--intensity-max", type=float, default=4.0)
    parser.add_argument("--intensity-sampling", choices=["linear", "log"], default="log")
    parser.add_argument("--color-saturation-min", type=float, default=0.0)
    parser.add_argument("--color-saturation-max", type=float, default=1.0)
    parser.add_argument("--color-value-min", type=float, default=1.0)
    parser.add_argument("--color-value-max", type=float, default=1.0)
    parser.add_argument("--global-diffuse-samples", type=int, default=0)
    parser.add_argument("--include-global-diffuse", action="store_true")
    parser.add_argument("--pbr-png", dest="pbr_png", action="store_true", default=True)
    parser.add_argument("--no-pbr-png", dest="pbr_png", action="store_false")
    parser.add_argument("--single-lights", choices=["all", "none"], default="all")
    parser.add_argument("--max-single-lights", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-existing", action="store_true", help="Copy metadata/masks/previews instead of hardlinking.")
    parser.add_argument(
        "--delete-source-exr",
        action="store_true",
        help="After each scene is converted successfully, delete EXR files from the source scene directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_source_roots(source: Path) -> list[Path]:
    if (source / "scenes").is_dir():
        return [source]
    roots = sorted(path.parent for path in source.glob("*/scenes") if path.is_dir())
    if not roots:
        raise SystemExit(f"SOURCE does not contain scenes/ directly or one level below it: {source}")
    return roots


def completed_scene_dirs(source_roots: list[Path], scene_offset: int, scene_limit: int) -> list[Path]:
    scene_dirs: list[Path] = []
    for source_root in source_roots:
        scene_dirs.extend(meta.parent for meta in sorted((source_root / "scenes").glob("scene_*/meta.json")))
    if scene_offset > 0:
        scene_dirs = scene_dirs[scene_offset:]
    if scene_limit > 0:
        scene_dirs = scene_dirs[:scene_limit]
    if not scene_dirs:
        raise SystemExit("No completed scenes found. A completed scene must contain scenes/<scene_id>/meta.json.")
    return scene_dirs


def dest_root_for(source: Path, dest: Path, source_root: Path) -> Path:
    if source_root == source:
        return dest
    return dest / source_root.relative_to(source)


def link_or_copy(src: Path, dst: Path, copy_existing: bool, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if copy_existing:
        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_path(scene_dir: Path, entry: dict[str, Any] | str) -> Path:
    if isinstance(entry, dict):
        value = entry.get("render_exr") or entry.get("exr") or entry.get("render")
    else:
        value = entry
    path = scene_dir / str(value)
    if path.suffix.lower() != ".exr":
        path = path.with_suffix(".exr")
    return path


def read_component(scene_dir: Path, entry: dict[str, Any] | str):
    from tokenlight_dataset.exr_io import read_exr

    return read_exr(entry_path(scene_dir, entry))


def read_pbr_exr(path: Path):
    import numpy as np

    try:
        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        available = set(header["channels"].keys())
        channel_sets = [("R", "G", "B"), ("X", "Y", "Z")]
        selected = None
        for names in channel_sets:
            if all(name in available for name in names):
                selected = names
                break
        if selected is None:
            if not available:
                raise RuntimeError("EXR has no channels")
            first = sorted(available)[0]
            selected = (first, first, first)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [
            np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width)
            for name in selected
        ]
        return np.stack(channels, axis=-1), list(selected)
    except Exception:
        from tokenlight_dataset.exr_io import read_exr

        return read_exr(path), ["R", "G", "B"]


def valid_lights(scene_dir: Path, spatial: dict[str, Any]) -> list[dict[str, Any]]:
    lights = []
    for light in spatial.get("point_lights", []):
        if not light.get("valid", True) or not light.get("render"):
            continue
        if entry_path(scene_dir, light).exists():
            lights.append(light)
    if not lights:
        raise RuntimeError(f"No valid spatial point lights in {scene_dir}")
    return lights


def sample_color(
    rng: random.Random,
    hue_slot: tuple[int, int] | None = None,
    saturation_range: tuple[float, float] = (0.0, 1.0),
    value_range: tuple[float, float] = (1.0, 1.0),
):
    import numpy as np

    if hue_slot is None:
        hue = rng.random()
    else:
        slot_index, slot_count = hue_slot
        slot_count = max(1, int(slot_count))
        hue = (float(slot_index % slot_count) + rng.random()) / float(slot_count)
    saturation = rng.uniform(float(saturation_range[0]), float(saturation_range[1]))
    value = rng.uniform(float(value_range[0]), float(value_range[1]))
    rgb = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return np.array(rgb, dtype=np.float32), [float(hue % 1.0), float(saturation), float(value)]


def sample_intensity(
    rng: random.Random,
    intensity_range: tuple[float, float],
    sampling: str,
) -> float:
    lo = float(intensity_range[0])
    hi = float(intensity_range[1])
    if sampling == "log" and lo > 0.0 and hi > 0.0:
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return rng.uniform(lo, hi)


def canonical_position(light: dict[str, Any]) -> list[float] | None:
    position = light.get("canonical_position")
    if not isinstance(position, list) or len(position) < 3:
        return None
    return [float(position[0]), float(position[1]), float(position[2])]


def canonical_distance(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    pa = canonical_position(a)
    pb = canonical_position(b)
    if pa is None or pb is None:
        return None
    return sum((av - bv) ** 2 for av, bv in zip(pa, pb)) ** 0.5


def sample_two_light_pairs(
    lights: list[dict[str, Any]],
    total_count: int,
    min_distance: float,
    min_distance_count: int,
    rng: random.Random,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str, float | None]], dict[str, Any]]:
    if len(lights) < 2 or total_count <= 0:
        return [], {
            "total_requested": max(0, total_count),
            "min_distance": min_distance,
            "min_distance_requested": max(0, min_distance_count),
            "min_distance_selected": 0,
            "random_selected": 0,
            "available_pairs": 0,
            "available_min_distance_pairs": 0,
        }

    all_pairs: list[tuple[dict[str, Any], dict[str, Any], float | None]] = []
    far_pairs: list[tuple[dict[str, Any], dict[str, Any], float | None]] = []
    for left_index, left in enumerate(lights[:-1]):
        for right in lights[left_index + 1 :]:
            distance = canonical_distance(left, right)
            all_pairs.append((left, right, distance))
            if distance is not None and distance >= min_distance:
                far_pairs.append((left, right, distance))

    rng.shuffle(all_pairs)
    rng.shuffle(far_pairs)

    selected: list[tuple[dict[str, Any], dict[str, Any], str, float | None]] = []
    selected_keys: set[tuple[int, int]] = set()

    far_target = min(max(0, min_distance_count), max(0, total_count))
    for left, right, distance in far_pairs[:far_target]:
        key = tuple(sorted((int(left["id"]), int(right["id"]))))
        selected.append((left, right, "min_distance", distance))
        selected_keys.add(key)

    remaining = total_count - len(selected)
    random_candidates = []
    for left, right, distance in all_pairs:
        key = tuple(sorted((int(left["id"]), int(right["id"]))))
        if key not in selected_keys:
            random_candidates.append((left, right, distance))

    for left, right, distance in random_candidates[:remaining]:
        key = tuple(sorted((int(left["id"]), int(right["id"]))))
        selected.append((left, right, "random", distance))
        selected_keys.add(key)

    while len(selected) < total_count and all_pairs:
        left, right, distance = rng.choice(all_pairs)
        selected.append((left, right, "random_repeat", distance))

    stats = {
        "total_requested": max(0, total_count),
        "min_distance": min_distance,
        "min_distance_requested": far_target,
        "min_distance_selected": sum(1 for row in selected if row[2] == "min_distance"),
        "random_selected": sum(1 for row in selected if row[2] in {"random", "random_repeat"}),
        "available_pairs": len(all_pairs),
        "available_min_distance_pairs": len(far_pairs),
    }
    return selected[:total_count], stats


def spatial_light_component(scene_dir: Path, spatial: dict[str, Any], light: dict[str, Any], ambient):
    import numpy as np

    component = read_component(scene_dir, light)
    if spatial.get("point_light_output_semantics") == "ambient_plus_point_light_target":
        component = component - ambient
    source_color_value = light.get("render_color")
    if source_color_value is None and light.get("component_color_semantics") != "random_color_for_later_composition":
        source_color_value = light.get("component_color")
    source_color = np.asarray(source_color_value or [1.0, 1.0, 1.0], dtype=np.float32)
    component = component / np.maximum(source_color.reshape(1, 1, 3), 1e-4)
    return np.maximum(component, 0.0)


def global_diffuse_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    return meta.get("global_diffuse") or meta.get("spatial", {}).get("global_diffuse")


def sample_global_background(
    scene_dir: Path,
    meta: dict[str, Any],
    rng: random.Random,
    include_global_diffuse: bool,
    ambient_scale_range: tuple[float, float],
):
    import numpy as np

    spatial = meta["spatial"]
    if not include_global_diffuse:
        ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        scale = rng.uniform(float(ambient_scale_range[0]), float(ambient_scale_range[1]))
        return scale * ambient, {
            "ambient_render": spatial.get("ambient_render"),
            "ambient_scale": scale,
            "global_diffuse": None,
        }

    diffuse = global_diffuse_meta(meta)
    if not diffuse or not diffuse.get("variants"):
        ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        scale = rng.uniform(float(ambient_scale_range[0]), float(ambient_scale_range[1]))
        return scale * ambient, {"ambient_scale": scale, "global_diffuse": None}

    variants = [row for row in diffuse.get("variants", []) if row.get("render")]
    if not variants:
        raise RuntimeError(f"No global_diffuse variants in {scene_dir}")
    variant = rng.choice(variants)
    dg = float(variant.get("dg", variant.get("normalized_diffuse", 0.0)))
    complete_targets = bool(diffuse.get("complete_target_variants", True))
    if complete_targets:
        return read_component(scene_dir, variant), {
            "complete_target_variants": True,
            "variant_id": variant.get("id"),
            "dg": dg,
        }

    ambient_entry = diffuse.get("ambient_output", diffuse.get("ambient_render"))
    if ambient_entry is None:
        raise RuntimeError(f"Component global_diffuse metadata needs ambient_output or ambient_render in {scene_dir}")
    ambient = read_component(scene_dir, ambient_entry)
    component = read_component(scene_dir, variant)
    ambient_range = diffuse.get("ambient_scale_range", [0.85, 1.15])
    intensity_range = diffuse.get("intensity_range", [0.85, 1.15])
    ambient_scale = rng.uniform(float(ambient_range[0]), float(ambient_range[1]))
    intensity = rng.uniform(float(intensity_range[0]), float(intensity_range[1]))
    color = np.asarray(diffuse.get("light", {}).get("color", [1.0, 1.0, 1.0]), dtype=np.float32)
    linear = ambient_scale * ambient + intensity * component * color.reshape(1, 1, 3)
    return linear, {
        "complete_target_variants": False,
        "variant_id": variant.get("id"),
        "dg": dg,
        "ambient_scale": ambient_scale,
        "intensity": intensity,
        "color": color.tolist(),
    }


def save_png(linear, path: Path) -> None:
    from PIL import Image

    from tokenlight_dataset.tonemap import reinhard, to_uint8

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(reinhard(linear)), mode="RGB").save(path, compress_level=1)


def save_unit_png(img, path: Path) -> None:
    from PIL import Image

    from tokenlight_dataset.tonemap import to_uint8

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img), mode="RGB").save(path, compress_level=1)


def normalize_depth_png(depth_img):
    import numpy as np

    depth = np.asarray(depth_img[:, :, 0], dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 0.0) & (depth < 1.0e6)
    if not np.any(finite):
        encoded = np.zeros_like(depth, dtype=np.float32)
        return np.repeat(encoded[:, :, None], 3, axis=2), {
            "encoding": "depth_percentile_1_99_near_white",
            "min_meters": 0.0,
            "max_meters": 1.0,
            "valid_pixel_ratio": 0.0,
        }
    valid = depth[finite]
    depth_min = float(np.percentile(valid, 1.0))
    depth_max = float(np.percentile(valid, 99.0))
    if depth_max <= depth_min + 1.0e-6:
        depth_max = depth_min + 1.0
    normalized = 1.0 - (depth - depth_min) / (depth_max - depth_min)
    normalized = np.where(finite, normalized, 0.0)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    return np.repeat(normalized[:, :, None], 3, axis=2), {
        "encoding": "depth_percentile_1_99_near_white",
        "min_meters": depth_min,
        "max_meters": depth_max,
        "valid_pixel_ratio": float(np.mean(finite)),
    }


def normalize_pbr_png(aux_type: str, img):
    import numpy as np

    img = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if aux_type == "depth":
        return normalize_depth_png(img)
    if aux_type == "normal":
        encoded = img
        if float(np.min(encoded)) < -1.0e-4 or float(np.max(encoded)) > 1.0 + 1.0e-4:
            encoded = encoded * 0.5 + 0.5
            encoding = "normal_xyz_minus1_1_to_0_1"
        else:
            encoding = "normal_rgb_0_1"
        return np.clip(encoded, 0.0, 1.0), {"encoding": encoding}
    if aux_type == "roughness":
        gray = np.mean(img[:, :, :3], axis=2)
        encoded = np.repeat(np.clip(gray, 0.0, 1.0)[:, :, None], 3, axis=2)
        return encoded.astype(np.float32), {"encoding": "roughness_grayscale_0_1"}
    return np.clip(img[:, :, :3], 0.0, 1.0), {"encoding": f"{aux_type}_rgb_0_1"}


def pbr_source_path(scene_dir: Path, pbr_maps: dict[str, Any], aux_type: str) -> Path | None:
    value = pbr_maps.get(aux_type)
    if isinstance(value, dict):
        value = value.get("render_exr") or value.get("exr") or value.get("render")
    if value:
        path = scene_dir / str(value)
        if path.exists():
            return path
    fallback = scene_dir / "pbr" / f"{aux_type}.exr"
    return fallback if fallback.exists() else None


def render_pbr_aux_samples(scene_dir: Path, scene_dest: Path, meta: dict[str, Any], overwrite: bool) -> list[dict[str, Any]]:
    pbr_maps = meta.get("pbr_maps") or meta.get("spatial", {}).get("pbr_maps") or {}
    samples = []
    for aux_type in ("albedo", "normal", "roughness", "depth"):
        source_path = pbr_source_path(scene_dir, pbr_maps, aux_type)
        if source_path is None:
            continue
        img, channels = read_pbr_exr(source_path)
        encoded, encoding_meta = normalize_pbr_png(aux_type, img)
        rel_path = Path("samples") / f"pbr_{aux_type}.png"
        out_path = scene_dest / rel_path
        if overwrite or not out_path.exists():
            save_unit_png(encoded, out_path)
        samples.append(
            {
                "image": rel_path.as_posix(),
                "scene_id": meta.get("scene_id"),
                "task": "pbr_aux",
                "aux_type": aux_type,
                "source": source_path.relative_to(scene_dir).as_posix(),
                "source_channels": channels,
                **encoding_meta,
            }
        )
    return samples


def render_global_ambient_samples(
    scene_dest: Path,
    meta: dict[str, Any],
    ambient,
    count: int,
    scale_range: tuple[float, float],
    rng: random.Random,
    overwrite: bool,
) -> list[dict[str, Any]]:
    samples = []
    for idx in range(max(0, count)):
        scale = rng.uniform(float(scale_range[0]), float(scale_range[1]))
        rel_path = Path("samples") / f"global_ambient_{idx:03d}.png"
        out_path = scene_dest / rel_path
        if overwrite or not out_path.exists():
            save_png(float(scale) * ambient, out_path)
        samples.append(
            {
                "image": rel_path.as_posix(),
                "scene_id": meta.get("scene_id"),
                "task": "global_ambient",
                "ambient_scale": scale,
            }
        )
    return samples


def render_global_diffuse_samples(
    scene_dir: Path,
    scene_dest: Path,
    meta: dict[str, Any],
    count: int,
    rng: random.Random,
    overwrite: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    diffuse = global_diffuse_meta(meta)
    if not diffuse:
        return []
    variants = sorted(
        [row for row in diffuse.get("variants", []) if row.get("render")],
        key=lambda row: float(row.get("dg", row.get("normalized_diffuse", 0.0))),
    )
    selected = [rng.choice(variants) for _ in range(max(0, count))] if variants else []
    if not selected:
        return []

    complete_targets = bool(diffuse.get("complete_target_variants", True))
    ambient = None
    color = None
    if not complete_targets:
        ambient_entry = diffuse.get("ambient_output", diffuse.get("ambient_render"))
        if ambient_entry is None:
            raise RuntimeError(f"Component global_diffuse metadata needs ambient_output or ambient_render in {scene_dir}")
        ambient = read_component(scene_dir, ambient_entry)
        color = np.asarray(diffuse.get("light", {}).get("color", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)

    samples = []
    for idx, variant in enumerate(selected):
        rel_path = Path("samples") / f"global_diffuse_{idx:03d}.png"
        out_path = scene_dest / rel_path
        dg = float(variant.get("dg", variant.get("normalized_diffuse", 0.0)))
        if complete_targets:
            linear = read_component(scene_dir, variant)
            condition = {"complete_target_variants": True, "variant_id": variant.get("id"), "dg": dg}
        else:
            component = read_component(scene_dir, variant)
            linear = ambient + component * color
            condition = {
                "complete_target_variants": False,
                "variant_id": variant.get("id"),
                "dg": dg,
                "ambient_scale": 1.0,
                "intensity": 1.0,
                "color": color.reshape(3).tolist(),
            }
        if overwrite or not out_path.exists():
            save_png(linear, out_path)
        samples.append(
            {
                "image": rel_path.as_posix(),
                "scene_id": meta.get("scene_id"),
                "task": "global_diffuse",
                "global_control": condition,
            }
        )
    return samples


def render_sample(
    scene_dir: Path,
    scene_dest: Path,
    meta: dict[str, Any],
    ambient,
    light_components: dict[int, Any],
    lights: list[dict[str, Any]],
    selected_lights: list[dict[str, Any]],
    out_name: str,
    rng: random.Random,
    include_global_diffuse: bool,
    ambient_scale_range: tuple[float, float],
    intensity_range: tuple[float, float],
    intensity_sampling: str,
    color_slots: list[tuple[int, int]] | None,
    color_saturation_range: tuple[float, float],
    color_value_range: tuple[float, float],
    overwrite: bool,
) -> dict[str, Any]:
    linear, global_condition = sample_global_background(scene_dir, meta, rng, include_global_diffuse, ambient_scale_range)
    light_conditions = []
    for light_index, light in enumerate(selected_lights):
        hue_slot = color_slots[light_index] if color_slots and light_index < len(color_slots) else None
        color, color_hsv = sample_color(rng, hue_slot, color_saturation_range, color_value_range)
        intensity = sample_intensity(rng, intensity_range, intensity_sampling)
        linear = linear + intensity * light_components[int(light["id"])] * color.reshape(1, 1, 3)
        light_conditions.append(
            {
                "id": int(light["id"]),
                "position": light.get("canonical_position"),
                "color": color.tolist(),
                "color_hsv": color_hsv,
                "intensity": intensity,
                "radius": light.get("canonical_radius"),
                "base_energy": light.get("canonical_energy"),
            }
        )

    rel_path = Path("samples") / out_name
    out_path = scene_dest / rel_path
    if overwrite or not out_path.exists():
        save_png(linear, out_path)
    return {
        "image": rel_path.as_posix(),
        "scene_id": meta.get("scene_id", scene_dir.name),
        "task": "single_light" if len(selected_lights) == 1 else "two_light",
        "global_control": global_condition,
        "lights": light_conditions,
    }


def stage_reference_files(scene_dir: Path, scene_dest: Path, copy_existing: bool, overwrite: bool) -> int:
    count = 0
    for src in scene_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() == ".exr":
            continue
        rel = src.relative_to(scene_dir)
        if rel.parts and rel.parts[0] == "samples":
            continue
        link_or_copy(src, scene_dest / rel, copy_existing, overwrite)
        count += 1
    return count


def delete_source_exr_files(scene_dir: Path) -> int:
    deleted = 0
    for path in sorted(scene_dir.rglob("*.exr")):
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def stage_scene(
    scene_dir: Path,
    source: Path,
    dest: Path,
    seed: int,
    single_lights: str,
    max_single_lights: int,
    two_light_samples: int,
    global_ambient_samples: int,
    global_diffuse_samples: int,
    include_global_diffuse: bool,
    pbr_png: bool,
    copy_existing: bool,
    overwrite: bool,
    delete_source_exr: bool,
    ambient_scale_range: tuple[float, float],
    intensity_range: tuple[float, float],
    intensity_sampling: str,
    two_light_min_distance: float,
    two_light_min_distance_count: int,
    color_saturation_range: tuple[float, float],
    color_value_range: tuple[float, float],
) -> tuple[str, int, int, int]:
    meta = load_json(scene_dir / "meta.json")
    source_root = scene_dir.parents[1]
    scene_dest = dest_root_for(source, dest, source_root) / "scenes" / scene_dir.name
    linked = stage_reference_files(scene_dir, scene_dest, copy_existing, overwrite)

    spatial = meta["spatial"]
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    lights = valid_lights(scene_dir, spatial)
    if max_single_lights > 0:
        single_candidates = lights[:max_single_lights]
    else:
        single_candidates = lights

    light_components = {
        int(light["id"]): spatial_light_component(scene_dir, spatial, light, ambient)
        for light in lights
    }

    scene_seed = seed + sum(ord(ch) for ch in scene_dir.name)
    rng = random.Random(scene_seed)
    samples: list[dict[str, Any]] = []

    samples.extend(
        render_global_ambient_samples(
            scene_dest,
            meta,
            ambient,
            global_ambient_samples,
            ambient_scale_range,
            random.Random(scene_seed * 100_000 + 20_000),
            overwrite,
        )
    )
    if include_global_diffuse:
        samples.extend(
            render_global_diffuse_samples(
                scene_dir,
                scene_dest,
                meta,
                global_diffuse_samples,
                random.Random(scene_seed * 100_000 + 30_000),
                overwrite,
            )
        )
    if pbr_png:
        samples.extend(render_pbr_aux_samples(scene_dir, scene_dest, meta, overwrite))

    if single_lights == "all":
        single_color_count = max(1, len(single_candidates))
        for single_index, light in enumerate(single_candidates):
            light_id = int(light["id"])
            sample_rng = random.Random(scene_seed * 100_000 + light_id)
            samples.append(
                render_sample(
                    scene_dir,
                    scene_dest,
                    meta,
                    ambient,
                    light_components,
                    lights,
                    [light],
                    f"light_{light_id:03d}.png",
                    sample_rng,
                    include_global_diffuse,
                    ambient_scale_range,
                    intensity_range,
                    intensity_sampling,
                    [(single_index, single_color_count)],
                    color_saturation_range,
                    color_value_range,
                    overwrite,
                )
            )

    if len(lights) >= 2 and two_light_samples > 0:
        pair_rows, pair_sampling = sample_two_light_pairs(
            lights,
            two_light_samples,
            two_light_min_distance,
            two_light_min_distance_count,
            rng,
        )
        double_color_count = max(1, len(pair_rows))
        for idx, (left, right, pair_source, pair_distance) in enumerate(pair_rows):
            sample_rng = random.Random(scene_seed * 100_000 + 10_000 + idx)
            sample = render_sample(
                scene_dir,
                scene_dest,
                meta,
                ambient,
                light_components,
                lights,
                [left, right],
                f"two_lights_{idx:03d}.png",
                sample_rng,
                include_global_diffuse,
                ambient_scale_range,
                intensity_range,
                intensity_sampling,
                [(idx, double_color_count), (idx + max(1, double_color_count // 2), double_color_count)],
                color_saturation_range,
                color_value_range,
                overwrite,
            )
            sample["pair_sampling"] = pair_source
            sample["pair_canonical_distance"] = pair_distance
            samples.append(sample)
    else:
        pair_sampling = None

    manifest = {
        "schema": "tokenlight_composed_png_samples_v1",
        "source_scene": str(scene_dir),
        "scene_id": meta.get("scene_id", scene_dir.name),
        "single_light_count": sum(1 for row in samples if row["task"] == "single_light"),
        "two_light_count": sum(1 for row in samples if row["task"] == "two_light"),
        "global_ambient_count": sum(1 for row in samples if row["task"] == "global_ambient"),
        "global_diffuse_count": sum(1 for row in samples if row["task"] == "global_diffuse"),
        "pbr_aux_count": sum(1 for row in samples if row["task"] == "pbr_aux"),
        "ambient_scale_range": [float(ambient_scale_range[0]), float(ambient_scale_range[1])],
        "intensity_range": [float(intensity_range[0]), float(intensity_range[1])],
        "intensity_sampling": intensity_sampling,
        "color_space": "hsv",
        "color_hue_sampling": "stratified_by_sample_index",
        "color_saturation_range": [float(color_saturation_range[0]), float(color_saturation_range[1])],
        "color_value_range": [float(color_value_range[0]), float(color_value_range[1])],
        "two_light_pair_sampling": pair_sampling,
        "samples": samples,
    }
    deleted_exr_count = delete_source_exr_files(scene_dir) if delete_source_exr else 0
    manifest["deleted_source_exr_count"] = deleted_exr_count
    manifest_path = scene_dest / "samples_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return scene_dir.name, len(samples), linked, deleted_exr_count


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    if not source.is_dir():
        raise SystemExit(f"SOURCE is not a directory: {source}")
    if dest == source or source in dest.parents:
        raise SystemExit("DEST must be outside SOURCE.")

    source_roots = discover_source_roots(source)
    scenes = completed_scene_dirs(source_roots, args.scene_offset, args.scene_limit)

    print(f"[INFO] source={source}")
    print(f"[INFO] dest={dest}")
    print(f"[INFO] source_roots={len(source_roots)} scene_offset={args.scene_offset} completed_scenes={len(scenes)}")
    print(
        f"[INFO] workers={args.workers} single_lights={args.single_lights} "
        f"two_light_samples={args.two_light_samples} "
        f"global_ambient_samples={args.global_ambient_samples} "
        f"global_diffuse_samples={args.global_diffuse_samples} "
        f"include_global_diffuse={args.include_global_diffuse} seed={args.seed}"
    )

    if args.dry_run:
        for root in source_roots:
            print(f"[DRY-RUN] root {root} -> {dest_root_for(source, dest, root)}")
        print("[DRY-RUN] no files written")
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_linked = 0
    total_deleted_exr = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                stage_scene,
                scene,
                source,
                dest,
                args.seed,
                args.single_lights,
                args.max_single_lights,
                args.two_light_samples,
                args.global_ambient_samples,
                args.global_diffuse_samples,
                args.include_global_diffuse,
                args.pbr_png,
                args.copy_existing,
                args.overwrite,
                args.delete_source_exr,
                (float(args.ambient_scale_min), float(args.ambient_scale_max)),
                (float(args.intensity_min), float(args.intensity_max)),
                str(args.intensity_sampling),
                float(args.two_light_min_distance),
                int(args.two_light_min_distance_count),
                (float(args.color_saturation_min), float(args.color_saturation_max)),
                (float(args.color_value_min), float(args.color_value_max)),
            )
            for scene in scenes
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            scene_id, sample_count, linked, deleted_exr = future.result()
            total_samples += sample_count
            total_linked += linked
            total_deleted_exr += deleted_exr
            if idx == 1 or idx % 25 == 0 or idx == len(futures):
                print(
                    f"[PROGRESS] scenes={idx}/{len(futures)} last={scene_id} "
                    f"samples={total_samples} linked={total_linked} deleted_exr={total_deleted_exr}",
                    flush=True,
                )

    dataset_manifest = {
        "schema": "tokenlight_composed_png_dataset_v1",
        "source": str(source),
        "scene_count": len(scenes),
        "sample_count": total_samples,
        "single_lights": args.single_lights,
        "two_light_samples_per_scene": args.two_light_samples,
        "two_light_min_distance": args.two_light_min_distance,
        "two_light_min_distance_count": args.two_light_min_distance_count,
        "global_ambient_samples_per_scene": args.global_ambient_samples,
        "ambient_scale_range": [args.ambient_scale_min, args.ambient_scale_max],
        "intensity_range": [args.intensity_min, args.intensity_max],
        "intensity_sampling": args.intensity_sampling,
        "color_space": "hsv",
        "color_hue_sampling": "stratified_by_sample_index",
        "color_saturation_range": [args.color_saturation_min, args.color_saturation_max],
        "color_value_range": [args.color_value_min, args.color_value_max],
        "global_diffuse_samples_per_scene": args.global_diffuse_samples,
        "include_global_diffuse": args.include_global_diffuse,
        "pbr_png": args.pbr_png,
        "delete_source_exr": args.delete_source_exr,
        "deleted_source_exr_count": total_deleted_exr,
        "seed": args.seed,
    }
    (dest / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote composed PNG stage: {dest}")
    print(f"[DONE] samples={total_samples} linked={total_linked} deleted_exr={total_deleted_exr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
