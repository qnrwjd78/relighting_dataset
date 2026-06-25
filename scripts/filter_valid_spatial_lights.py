#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image


def read_exr_rgb(path: Path) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []
    for name in ("R", "G", "B"):
        raw = exr.channel(name, pixel_type)
        channels.append(np.frombuffer(raw, dtype=np.float32).reshape(height, width))
    return np.stack(channels, axis=-1)


def read_object_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.ones(shape, dtype=bool)
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def receiver_reject_reason(light: dict, bounds: dict | None, margin: float) -> str | None:
    if not bounds:
        return None
    point = np.asarray(light["world_position"], dtype=np.float64)
    radius = float(light.get("world_radius", 0.0))
    origin = np.asarray(bounds["origin"], dtype=np.float64)
    right = np.asarray(bounds["right"], dtype=np.float64)
    forward = np.asarray(bounds["forward"], dtype=np.float64)
    rel = point - origin
    x = float(np.dot(rel, right))
    y = float(np.dot(rel, forward))
    z = float(point[2])
    safety = radius + margin
    floor_z = float(bounds["floor_z"])
    wall_height = float(bounds["wall_height"])
    if z < floor_z + safety:
        return "below_floor"
    if z > floor_z + wall_height - safety:
        return "above_wall_height"
    if y > float(bounds["back_y"]) - safety:
        return "behind_back_wall"
    if y < float(bounds["front_y"]) + safety:
        return "outside_ground_front"
    if abs(x) > float(bounds["half_width"]) - safety:
        return "outside_side_bounds"
    return None


def summarize_luma(values: np.ndarray, pixel_threshold: float) -> dict:
    if values.size == 0:
        return {
            "mean": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "affected_fraction": 0.0,
        }
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "affected_fraction": 0.0,
        }
    return {
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "affected_fraction": float(np.mean(finite > pixel_threshold)),
    }


def evaluate_light(scene_dir: Path, light: dict, mask: np.ndarray | None, bounds: dict | None, args: argparse.Namespace) -> dict:
    rel = light.get("render_exr") or light.get("render")
    exr_path = scene_dir / rel if rel else None
    reasons: list[str] = []
    bounds_reason = receiver_reject_reason(light, bounds, args.bounds_margin)
    if bounds_reason:
        reasons.append(bounds_reason)
    if exr_path is None or not exr_path.exists():
        reasons.append("missing_exr")
        return {
            "id": light.get("id"),
            "valid_post": False,
            "reject_reasons": reasons,
            "render_exr": rel,
        }

    rgb = read_exr_rgb(exr_path)
    lum = luma(rgb)
    if not np.isfinite(lum).all():
        reasons.append("non_finite_pixels")
    image_stats = summarize_luma(lum, args.pixel_threshold)
    if mask is None:
        mask = np.ones(lum.shape, dtype=bool)
    object_stats = summarize_luma(lum[mask], args.pixel_threshold)

    if image_stats["affected_fraction"] < args.min_image_affected_fraction:
        reasons.append("too_few_affected_pixels")
    if image_stats["mean"] < args.min_image_mean:
        reasons.append("image_mean_too_low")
    if image_stats["max"] < args.min_image_max:
        reasons.append("image_max_too_low")
    if args.require_object_impact:
        if object_stats["affected_fraction"] < args.min_object_affected_fraction:
            reasons.append("object_too_few_affected_pixels")
        if object_stats["mean"] < args.min_object_mean:
            reasons.append("object_mean_too_low")

    return {
        "id": int(light.get("id", -1)),
        "valid_post": len(reasons) == 0,
        "reject_reasons": reasons,
        "render_exr": rel,
        "canonical_position": light.get("canonical_position"),
        "world_position": light.get("world_position"),
        "world_radius": light.get("world_radius"),
        "world_energy": light.get("world_energy"),
        "image_mean": image_stats["mean"],
        "image_max": image_stats["max"],
        "image_p95": image_stats["p95"],
        "image_p99": image_stats["p99"],
        "image_affected_fraction": image_stats["affected_fraction"],
        "object_mean": object_stats["mean"],
        "object_max": object_stats["max"],
        "object_p95": object_stats["p95"],
        "object_p99": object_stats["p99"],
        "object_affected_fraction": object_stats["affected_fraction"],
    }


def evaluate_scene(scene_dir: Path, args: argparse.Namespace) -> dict | None:
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    spatial = meta.get("spatial") or {}
    lights = spatial.get("point_lights") or []
    if not lights:
        return None

    first_rel = lights[0].get("render_exr") or lights[0].get("render")
    first_path = scene_dir / first_rel
    if not first_path.exists():
        height = width = None
        mask = None
    else:
        first_rgb = read_exr_rgb(first_path)
        height, width = first_rgb.shape[:2]
        mask_rel = meta.get("masks", {}).get("object", "masks/object_mask.png")
        mask = read_object_mask(scene_dir / mask_rel, (height, width))

    bounds = spatial.get("receiver_bounds")
    evaluated = [evaluate_light(scene_dir, light, mask, bounds, args) for light in lights]
    valid_ids = [item["id"] for item in evaluated if item.get("valid_post")]
    rejected_ids = [item["id"] for item in evaluated if not item.get("valid_post")]
    result = {
        "scene_id": meta.get("scene_id", scene_dir.name),
        "scene_dir": str(scene_dir),
        "thresholds": {
            "bounds_margin": args.bounds_margin,
            "pixel_threshold": args.pixel_threshold,
            "min_image_affected_fraction": args.min_image_affected_fraction,
            "min_image_mean": args.min_image_mean,
            "min_image_max": args.min_image_max,
            "require_object_impact": args.require_object_impact,
            "min_object_affected_fraction": args.min_object_affected_fraction,
            "min_object_mean": args.min_object_mean,
        },
        "valid_count": len(valid_ids),
        "rejected_count": len(rejected_ids),
        "valid_light_ids": valid_ids,
        "rejected_light_ids": rejected_ids,
        "lights": evaluated,
    }
    (scene_dir / "valid_lights.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_global_outputs(root: Path, results: list[dict]) -> None:
    summary_path = root / "valid_lights_summary.jsonl"
    with summary_path.open("w") as f:
        for result in results:
            f.write(json.dumps({k: v for k, v in result.items() if k != "lights"}) + "\n")

    csv_path = root / "valid_lights.csv"
    fields = [
        "scene_id",
        "light_id",
        "valid_post",
        "reject_reasons",
        "render_exr",
        "image_mean",
        "image_max",
        "image_p99",
        "image_affected_fraction",
        "object_mean",
        "object_max",
        "object_p99",
        "object_affected_fraction",
        "world_x",
        "world_y",
        "world_z",
        "world_radius",
        "world_energy",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for light in result["lights"]:
                world = light.get("world_position") or [None, None, None]
                writer.writerow({
                    "scene_id": result["scene_id"],
                    "light_id": light.get("id"),
                    "valid_post": light.get("valid_post"),
                    "reject_reasons": ";".join(light.get("reject_reasons", [])),
                    "render_exr": light.get("render_exr"),
                    "image_mean": light.get("image_mean"),
                    "image_max": light.get("image_max"),
                    "image_p99": light.get("image_p99"),
                    "image_affected_fraction": light.get("image_affected_fraction"),
                    "object_mean": light.get("object_mean"),
                    "object_max": light.get("object_max"),
                    "object_p99": light.get("object_p99"),
                    "object_affected_fraction": light.get("object_affected_fraction"),
                    "world_x": world[0],
                    "world_y": world[1],
                    "world_z": world[2],
                    "world_radius": light.get("world_radius"),
                    "world_energy": light.get("world_energy"),
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rendered spatial light components by receiver bounds and EXR impact.")
    parser.add_argument("--root", default="outputs/objaverse_sample", help="Dataset output root containing scenes/.")
    parser.add_argument("--scene-glob", default="scene_*", help="Scene directory glob under root/scenes.")
    parser.add_argument("--bounds-margin", type=float, default=0.02)
    parser.add_argument("--pixel-threshold", type=float, default=1e-4)
    parser.add_argument("--min-image-affected-fraction", type=float, default=1e-3)
    parser.add_argument("--min-image-mean", type=float, default=1e-5)
    parser.add_argument("--min-image-max", type=float, default=1e-3)
    parser.add_argument("--require-object-impact", action="store_true", default=True)
    parser.add_argument("--no-require-object-impact", dest="require_object_impact", action="store_false")
    parser.add_argument("--min-object-affected-fraction", type=float, default=5e-3)
    parser.add_argument("--min-object-mean", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    scenes_root = root / "scenes"
    scene_dirs = sorted(path for path in scenes_root.glob(args.scene_glob) if path.is_dir())
    results = []
    for scene_dir in scene_dirs:
        result = evaluate_scene(scene_dir, args)
        if result is not None:
            results.append(result)
            print(f"{result['scene_id']}: valid {result['valid_count']} / {result['valid_count'] + result['rejected_count']}")
    write_global_outputs(root, results)
    print(f"Wrote {root / 'valid_lights_summary.jsonl'}")
    print(f"Wrote {root / 'valid_lights.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
