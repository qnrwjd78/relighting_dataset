#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Tuple


Vec3 = Tuple[float, float, float]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def to_vec3(value: list[float] | tuple[float, float, float]) -> Vec3:
    return (float(value[0]), float(value[1]), float(value[2]))


def vec_to_list(value: Vec3) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def bbox_corners(bbox_min: Vec3, bbox_max: Vec3) -> list[Vec3]:
    return [
        (x, y, z)
        for x in (bbox_min[0], bbox_max[0])
        for y in (bbox_min[1], bbox_max[1])
        for z in (bbox_min[2], bbox_max[2])
    ]


def get_component_meta(meta: dict[str, Any]) -> dict[str, Any]:
    if isinstance(meta.get("spatial"), dict):
        return meta["spatial"]
    if isinstance(meta.get("debug"), dict):
        return meta["debug"]
    raise KeyError("meta must contain either a spatial or debug section")


def get_transform(meta: dict[str, Any]) -> dict[str, Any]:
    section = get_component_meta(meta)
    transform = section.get("canonical_transform")
    if not isinstance(transform, dict):
        raise KeyError("meta section does not contain canonical_transform")
    return transform


def camera_location(meta: dict[str, Any]) -> Vec3:
    camera = meta.get("camera") or {}
    location = camera.get("location")
    if location is None:
        raise KeyError("meta.camera.location is missing")
    return to_vec3(location)


def camera_axes(meta: dict[str, Any]) -> tuple[Vec3, Vec3, Vec3]:
    transform = get_transform(meta)
    right = to_vec3(transform["x_axis_world"])
    forward = to_vec3(transform["y_axis_world"])
    up = to_vec3(transform["z_axis_world"])
    return right, forward, up


def choose_bbox(meta: dict[str, Any], source: str) -> tuple[str, Vec3, Vec3]:
    obj = meta.get("object") or {}
    if source == "full-scene" and obj.get("full_scene_bbox_min") and obj.get("full_scene_bbox_max"):
        return source, to_vec3(obj["full_scene_bbox_min"]), to_vec3(obj["full_scene_bbox_max"])
    if source == "auto" and obj.get("full_scene_bbox_min") and obj.get("full_scene_bbox_max"):
        return "full-scene", to_vec3(obj["full_scene_bbox_min"]), to_vec3(obj["full_scene_bbox_max"])
    if obj.get("bbox_min") and obj.get("bbox_max"):
        return "object", to_vec3(obj["bbox_min"]), to_vec3(obj["bbox_max"])
    raise KeyError("meta.object bbox fields are missing")


def depth_bounds_for_bbox(meta: dict[str, Any], source: str, min_depth: float) -> dict[str, Any]:
    used_source, bbox_min, bbox_max = choose_bbox(meta, source)
    _right, forward, _up = camera_axes(meta)
    cam = camera_location(meta)
    depths = [dot(vsub(corner, cam), forward) for corner in bbox_corners(bbox_min, bbox_max)]
    raw_start = min(depths)
    raw_end = max(depths)
    if raw_end <= min_depth:
        start = min_depth
        end = min_depth + 1e-6
    elif raw_start <= min_depth:
        start = min_depth
        end = raw_end
    else:
        start = raw_start
        end = raw_end
    if end <= start:
        end = start + max(min_depth, 1e-6)
    return {
        "source": used_source,
        "bbox_min": vec_to_list(bbox_min),
        "bbox_max": vec_to_list(bbox_max),
        "corner_depths": depths,
        "start_depth": start,
        "end_depth": end,
        "depth_range": end - start,
    }


def current_cube(meta: dict[str, Any]) -> dict[str, Any]:
    transform = get_transform(meta)
    scale = float(transform.get("scale", 0.0))
    center_depth = float(transform.get("camera_distance", 0.0))
    center = to_vec3(transform.get("target_center", [0.0, 0.0, 0.0]))
    return {
        "center_world": vec_to_list(center),
        "scale": scale,
        "center_depth": center_depth,
        "front_depth": center_depth - scale,
        "back_depth": center_depth + scale,
        "center_depth_over_scale": center_depth / scale if scale > 0 else None,
        "front_depth_over_scale": (center_depth - scale) / scale if scale > 0 else None,
        "back_depth_over_scale": (center_depth + scale) / scale if scale > 0 else None,
    }


def camera_coords(meta: dict[str, Any], point: Vec3) -> dict[str, float]:
    right, forward, up = camera_axes(meta)
    rel = vsub(point, camera_location(meta))
    return {
        "x": dot(rel, right),
        "y_depth": dot(rel, forward),
        "z": dot(rel, up),
    }


def source_center(meta: dict[str, Any], mode: str) -> Vec3:
    if mode == "object":
        center = (meta.get("object") or {}).get("center")
        if center is not None:
            return to_vec3(center)
    section = get_component_meta(meta)
    center = section.get("light_volume_center")
    if center is not None:
        return to_vec3(center)
    transform = get_transform(meta)
    return to_vec3(transform["target_center"])


def scene_depth_cube(meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    bounds = depth_bounds_for_bbox(meta, args.depth_bounds, args.min_depth)
    start = float(bounds["start_depth"])
    end = float(bounds["end_depth"])
    front_depth = start + float(args.front_depth_ratio) * (end - start)
    front_depth = max(front_depth, args.min_depth)

    front_over_scale = float(args.tokenlight_front_over_scale)
    center_over_scale = float(args.tokenlight_center_over_scale)
    scale = front_depth / front_over_scale
    center_depth = center_over_scale * scale
    back_depth = center_depth + (center_depth - front_depth)

    cam = camera_location(meta)
    right, forward, up = camera_axes(meta)
    src = source_center(meta, args.center_source)
    src_cam = camera_coords(meta, src)

    if args.lateral_mode == "camera-axis":
        x = 0.0
        z = 0.0
    elif args.lateral_mode == "preserve-world":
        x = src_cam["x"]
        z = src_cam["z"]
    else:
        source_depth = max(src_cam["y_depth"], args.min_depth)
        x = src_cam["x"] * center_depth / source_depth
        z = src_cam["z"] * center_depth / source_depth

    center_world = vadd(vadd(vadd(cam, vmul(forward, center_depth)), vmul(right, x)), vmul(up, z))
    front_world = vadd(center_world, vmul(forward, -(center_depth - front_depth)))
    back_world = vadd(center_world, vmul(forward, back_depth - center_depth))

    return {
        "schema": "scene_depth_cube_v1",
        "scene_id": meta.get("scene_id"),
        "scene_type": meta.get("scene_type"),
        "source": {
            "center_source": args.center_source,
            "lateral_mode": args.lateral_mode,
            "source_center_world": vec_to_list(src),
            "source_center_camera": src_cam,
        },
        "depth_bounds": bounds,
        "ratios": {
            "scene_front_depth_ratio": float(args.front_depth_ratio),
            "tokenlight_front_depth_over_scale": front_over_scale,
            "tokenlight_center_depth_over_scale": center_over_scale,
            "tokenlight_back_depth_over_scale": (back_depth / scale) if scale > 0 else None,
        },
        "computed": {
            "center_world": vec_to_list(center_world),
            "front_world": vec_to_list(front_world),
            "back_world": vec_to_list(back_world),
            "scale": scale,
            "center_depth": center_depth,
            "front_depth": front_depth,
            "back_depth": back_depth,
            "center_depth_over_scale": center_depth / scale if scale > 0 else None,
            "front_depth_over_scale": front_depth / scale if scale > 0 else None,
            "back_depth_over_scale": back_depth / scale if scale > 0 else None,
            "scene_front_depth_ratio": (front_depth - start) / (end - start) if end > start else None,
            "scene_center_depth_ratio": (center_depth - start) / (end - start) if end > start else None,
            "scene_back_depth_ratio": (back_depth - start) / (end - start) if end > start else None,
            "camera_axes": {
                "right": vec_to_list(right),
                "forward": vec_to_list(forward),
                "up": vec_to_list(up),
            },
        },
        "current": current_cube(meta),
        "object_coverage": object_coverage(meta, center_world, scale),
    }


def object_coverage(meta: dict[str, Any], cube_center: Vec3, scale: float) -> dict[str, Any]:
    obj = meta.get("object") or {}
    if not obj.get("bbox_min") or not obj.get("bbox_max") or scale <= 0:
        return {}
    right, forward, up = camera_axes(meta)
    bbox_min = to_vec3(obj["bbox_min"])
    bbox_max = to_vec3(obj["bbox_max"])
    coords = []
    for corner in bbox_corners(bbox_min, bbox_max):
        rel = vsub(corner, cube_center)
        coords.append(
            {
                "x": dot(rel, right) / scale,
                "y_depth": dot(rel, forward) / scale,
                "z": dot(rel, up) / scale,
            }
        )
    mins = {axis: min(coord[axis] for coord in coords) for axis in ("x", "y_depth", "z")}
    maxs = {axis: max(coord[axis] for coord in coords) for axis in ("x", "y_depth", "z")}
    outside = any(mins[axis] < -1.0 or maxs[axis] > 1.0 for axis in mins)
    return {
        "canonical_bbox_min": mins,
        "canonical_bbox_max": maxs,
        "outside_unit_cube": outside,
    }


def find_meta_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    scene_metas = sorted(input_path.glob("scenes/*/meta.json"))
    if scene_metas:
        return scene_metas
    return sorted(input_path.rglob("meta.json"))


def relative_to_or_name(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a scene-depth-defined TokenLight-style cube from existing meta.json files. "
            "This does not modify renders; it writes sidecar JSON files for inspection."
        )
    )
    parser.add_argument("--input", required=True, help="Dataset root or a single scene meta.json.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where sidecar JSON files are written. Defaults to outputs/scene_depth_cubes/<input-name>.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N meta files.")
    parser.add_argument(
        "--depth-bounds",
        choices=["auto", "full-scene", "object"],
        default="auto",
        help="Depth range used as start/end. auto uses full_scene_bbox when present, else object bbox.",
    )
    parser.add_argument(
        "--front-depth-ratio",
        type=float,
        default=0.3,
        help="r_front in front_depth = start + r_front * (end - start).",
    )
    parser.add_argument("--tokenlight-front-over-scale", type=float, default=3.5)
    parser.add_argument("--tokenlight-center-over-scale", type=float, default=4.5)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument(
        "--center-source",
        choices=["light-volume", "object"],
        default="light-volume",
        help="Point whose screen/lateral location is reused for the new cube.",
    )
    parser.add_argument(
        "--lateral-mode",
        choices=["preserve-projection", "preserve-world", "camera-axis"],
        default="preserve-projection",
        help=(
            "How to place the cube center laterally. preserve-projection keeps the source point's "
            "2D projection at the new depth; camera-axis places the cube on image center."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("outputs/scene_depth_cubes") / input_path.stem
    metas = find_meta_files(input_path)
    if args.limit is not None:
        metas = metas[: max(args.limit, 0)]
    if not metas:
        raise SystemExit(f"No meta.json files found under {input_path}")

    written = []
    outside_count = 0
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result = scene_depth_cube(meta, args)
        rel = relative_to_or_name(meta_path, input_path if input_path.is_dir() else meta_path.parent)
        out_path = output_dir / rel.with_name("scene_depth_cube.json")
        write_json(out_path, result)
        written.append(out_path)
        if result.get("object_coverage", {}).get("outside_unit_cube"):
            outside_count += 1

    summary = {
        "schema": "scene_depth_cube_summary_v1",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "count": len(written),
        "object_outside_unit_cube_count": outside_count,
        "settings": {
            "depth_bounds": args.depth_bounds,
            "front_depth_ratio": args.front_depth_ratio,
            "tokenlight_front_over_scale": args.tokenlight_front_over_scale,
            "tokenlight_center_over_scale": args.tokenlight_center_over_scale,
            "center_source": args.center_source,
            "lateral_mode": args.lateral_mode,
        },
        "sidecars": [str(path) for path in written],
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote {len(written)} scene-depth cube sidecars to {output_dir}")
    print(f"Object outside computed unit cube: {outside_count}/{len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
