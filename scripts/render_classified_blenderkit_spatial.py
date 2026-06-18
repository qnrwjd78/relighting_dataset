from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_AGENT = "relighting-dataset-blenderkit-classified-spatial/0.1"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description=(
            "Re-download classified BlenderKit .blend scenes one at a time, render spatial "
            "ambient + point-light components, then delete the .blend."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--blend", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--item-json", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--classification", default="outputs/previews/blenderkit/blenderkit_scene_use_classification.txt")
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["single_scene_light_good", "background_good_for_portrait_or_object"],
        help="Classification section names to render.",
    )
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full.json")
    parser.add_argument("--output", default="outputs/blenderkit_classified_spatial")
    parser.add_argument("--preview-dir", default="outputs/previews/blenderkit_classified_spatial")
    parser.add_argument("--work-dir", default="outputs/work/blenderkit_classified_spatial")
    parser.add_argument("--download-dir", default="data/blenderkit_spatial_cache")
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default="png")
    parser.add_argument(
        "--ambient-source",
        choices=["scene", "hdri"],
        default="scene",
        help="Use the source .blend scene lights for ambient preview/components, or replace them with sampled HDRI lighting.",
    )
    parser.add_argument(
        "--point-light-mode",
        choices=["target", "component"],
        default="target",
        help="Render point-light PNGs as final targets with ambient lighting included, or isolated point-light components.",
    )
    parser.add_argument("--hdri-mode", choices=["on", "off", "random"], default="on")
    parser.add_argument("--global-diffuse", dest="global_diffuse", action="store_true", default=None)
    parser.add_argument("--no-global-diffuse", dest="global_diffuse", action="store_false")
    parser.add_argument("--per-light-diffuse", dest="per_light_diffuse", action="store_true", default=None)
    parser.add_argument("--no-per-light-diffuse", dest="per_light_diffuse", action="store_false")
    parser.add_argument("--debug", action="store_true", help="Render preview outputs only and skip point-light components.")
    parser.add_argument("--light-preview", action="store_true", default=True)
    parser.add_argument("--no-light-preview", action="store_false", dest="light_preview")
    parser.add_argument("--positions-per-scene", type=int, default=None)
    parser.add_argument(
        "--light-volume-placement",
        choices=["bbox-center", "camera-framed"],
        default="camera-framed",
        help="Place the spatial light cube at the subject bbox center or move it onto the current camera axis.",
    )
    parser.add_argument(
        "--light-volume-depth-over-scale",
        type=float,
        default=None,
        help="Target camera depth / cube scale for camera-framed placement. Defaults to the canonical rig camera distance.",
    )
    parser.add_argument(
        "--spatial-bbox-mode",
        choices=["auto", "candidates", "full"],
        default="auto",
        help="BBox used to place the 64 spatial lights. auto/candidates use preview metadata subject candidates; full uses all scene meshes.",
    )
    parser.add_argument("--subject-candidate-count", type=int, default=3)
    parser.add_argument("--candidate-padding", type=float, default=1.15)
    parser.add_argument("--candidate-outlier-factor", type=float, default=20.0)
    parser.add_argument("--overwrite-blend", action="store_true")
    parser.add_argument("--keep-blend", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args(argv)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def repo_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        return resolve_repo_path(args.api_key_file).read_text(encoding="utf-8").strip()
    return ""


def headers(api_key: str, user_agent: str) -> dict[str, str]:
    result = {"User-Agent": user_agent}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def request_json(url: str, api_key: str, user_agent: str) -> dict:
    req = urllib.request.Request(url, headers=headers(api_key, user_agent))
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_download_url(download_url: str, api_key: str, user_agent: str, scene_uuid: str) -> str:
    sep = "&" if "?" in download_url else "?"
    data = request_json(f"{download_url}{sep}scene_uuid={scene_uuid}", api_key, user_agent)
    for key in ("filePath", "file_path", "url", "downloadUrl", "download_url"):
        if data.get(key):
            return str(data[key])
    files = data.get("files")
    if isinstance(files, list):
        for item in files:
            for key in ("filePath", "url", "downloadUrl"):
                if item.get(key):
                    return str(item[key])
    raise RuntimeError(f"Download response did not contain a file URL: {data}")


def download_file(url: str, path: Path, api_key: str, user_agent: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"[BlenderKitSpatial] Exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers=headers(api_key, user_agent))
    with urllib.request.urlopen(req, timeout=240) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)
    print(f"[BlenderKitSpatial] Downloaded: {path}")


def section_slug(line: str) -> str | None:
    match = re.match(r"\[\d+\]\s+([^\s]+)", line.strip())
    return match.group(1) if match else None


def selected_ids(classification: Path, categories: set[str]) -> list[str]:
    ids: list[str] = []
    active = False
    for raw in classification.read_text(encoding="utf-8").splitlines():
        slug = section_slug(raw)
        if slug:
            active = slug in categories
            continue
        if not active:
            continue
        match = re.match(r"- blenderkit_(\d+)", raw)
        if match:
            ids.append(match.group(1).zfill(5))
    return ids


def load_selected_items(args: argparse.Namespace) -> list[dict]:
    ids = selected_ids(resolve_repo_path(args.classification), set(args.categories))
    index = json.loads(resolve_repo_path(args.index_json).read_text(encoding="utf-8"))
    by_id = {str(item.get("id")).zfill(5): item for item in index.get("items", [])}
    items = [by_id[item_id] for item_id in ids if item_id in by_id]
    return items[args.start : args.start + args.limit if args.limit is not None else None]


def safe_blend_name(item: dict) -> str:
    original = item.get("original_blend_path")
    if original:
        return Path(original).name
    base = item.get("asset_base_id") or item.get("asset_id") or item.get("id")
    return f"{base}.blend"


def write_worker_item(item: dict, work_dir: Path) -> Path:
    item_path = work_dir / "worker_item.json"
    item_path.parent.mkdir(parents=True, exist_ok=True)
    item_path.write_text(json.dumps(item, ensure_ascii=True), encoding="utf-8")
    return item_path


def worker_command(args: argparse.Namespace, blend_path: Path, item_path: Path) -> list[str]:
    cmd = shlex.split(args.blender_cmd) + [
        "-b",
        "--python",
        str(ROOT / "scripts" / "render_classified_blenderkit_spatial.py"),
        "--",
        "--worker",
        "--blend",
        str(blend_path),
        "--item-json",
        str(item_path),
        "--config",
        args.config,
        "--output",
        args.output,
        "--preview-dir",
        args.preview_dir,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
        "--component-format",
        args.component_format,
        "--ambient-source",
        args.ambient_source,
        "--point-light-mode",
        args.point_light_mode,
        "--hdri-mode",
        args.hdri_mode,
    ]
    if args.positions_per_scene is not None:
        cmd.extend(["--positions-per-scene", str(args.positions_per_scene)])
    cmd.extend(["--light-volume-placement", args.light_volume_placement])
    if args.light_volume_depth_over_scale is not None:
        cmd.extend(["--light-volume-depth-over-scale", str(args.light_volume_depth_over_scale)])
    cmd.extend(["--spatial-bbox-mode", args.spatial_bbox_mode])
    cmd.extend(["--subject-candidate-count", str(args.subject_candidate_count)])
    cmd.extend(["--candidate-padding", str(args.candidate_padding)])
    cmd.extend(["--candidate-outlier-factor", str(args.candidate_outlier_factor)])
    if args.debug:
        cmd.append("--debug")
    if args.light_preview:
        cmd.append("--light-preview")
    else:
        cmd.append("--no-light-preview")
    return cmd


def orchestrate(args: argparse.Namespace) -> int:
    api_key = load_api_key(args)
    if not api_key:
        raise SystemExit("Missing BlenderKit API key. Set BLENDERKIT_API_KEY or pass --api-key-file.")

    items = load_selected_items(args)
    if not items:
        raise SystemExit("No selected BlenderKit items matched the requested categories.")

    download_dir = resolve_repo_path(args.download_dir)
    work_dir = resolve_repo_path(args.work_dir)
    output_root = resolve_repo_path(args.output)
    scene_uuid = str(uuid.uuid4())

    print(f"[BlenderKitSpatial] Selected items: {len(items)}")
    for index, item in enumerate(items, 1):
        item_id = str(item.get("id")).zfill(5)
        scene_id = f"blenderkit_{item_id}"
        meta_path = output_root / "scenes" / scene_id / "meta.json"
        if args.skip_existing and meta_path.exists():
            print(f"[BlenderKitSpatial] Skip existing {scene_id}: {meta_path}")
            continue

        download_api_url = item.get("download_api_url") or item.get("record", {}).get("download_api_url")
        if not download_api_url:
            print(f"[BlenderKitSpatial] Skip missing download_api_url: {scene_id}")
            continue

        blend_path = download_dir / safe_blend_name(item)
        print(f"[BlenderKitSpatial] {index}/{len(items)} {scene_id} {item.get('name')}")
        try:
            resolved_url = resolve_download_url(download_api_url, api_key, args.user_agent, scene_uuid)
            download_file(resolved_url, blend_path, api_key, args.user_agent, args.overwrite_blend)
            item_path = write_worker_item(item, work_dir)
            cmd = worker_command(args, blend_path, item_path)
            print("[BlenderKitSpatial] Render:", " ".join(shlex.quote(part) for part in cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[BlenderKitSpatial] FAILED {scene_id}: {exc}", file=sys.stderr)
            continue
        finally:
            if not args.keep_blend:
                blend_path.unlink(missing_ok=True)
                blend_path.with_suffix(blend_path.suffix + ".part").unlink(missing_ok=True)
                print(f"[BlenderKitSpatial] Deleted blend: {blend_path}")
        time.sleep(args.sleep)
    return 0


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def load_runtime_config(config_path: Path, args: argparse.Namespace) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output_root"] = args.output
    config["render"]["resolution_x"] = args.width
    config["render"]["resolution_y"] = args.height
    config["render"]["samples"] = args.samples
    config["render"]["component_format"] = args.component_format
    if args.positions_per_scene is not None:
        config["spatial"]["positions_per_scene"] = args.positions_per_scene
    if args.global_diffuse is not None:
        config.setdefault("global_diffuse", {})["enabled"] = bool(args.global_diffuse)
    if args.per_light_diffuse is not None:
        config.setdefault("spatial", {}).setdefault("per_light_diffuse", {})["enabled"] = bool(args.per_light_diffuse)
    config["_component_format"] = args.component_format
    config["_ambient_source"] = args.ambient_source
    config["_point_light_mode"] = args.point_light_mode
    config["_hdri_mode"] = args.hdri_mode
    config["_light_preview"] = bool(args.light_preview)
    config["_render_pbr"] = False
    hdri_manifest_value = config.get("hdri_manifest")
    hdri_manifest = resolve_repo_path(hdri_manifest_value) if hdri_manifest_value else None
    config["_runtime"] = {
        "hdris": [str(resolve_repo_path(path)) for path in load_lines(hdri_manifest)] if hdri_manifest else [],
        "receiver_bounds": None,
        "receiver_materials": [],
    }
    return config


def copy_spatial_previews(args: argparse.Namespace, scene_id: str, scene_dir: Path, item: dict) -> None:
    if args.component_format == "exr":
        return
    preview_dir = resolve_repo_path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    generated_preview_dir = scene_dir.parent / "preview"
    sources = {
        "ambient": scene_dir / "spatial" / "ambient.png",
        "light_positions": generated_preview_dir / f"{scene_id}_light_positions.png",
    }
    # render_spatial_components writes ../preview relative to scene_dir.
    sources["ambient_preview"] = generated_preview_dir / f"{scene_id}_ambient.png"
    for name, src in sources.items():
        if not src.exists():
            continue
        dst = preview_dir / f"{scene_id}_{name}.png"
        shutil.copyfile(src, dst)
        copied[name] = repo_relative(dst)
    if copied:
        index_path = preview_dir / "index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            rows = data.get("items", [])
        else:
            rows = []
        rows = [row for row in rows if row.get("scene_id") != scene_id]
        rows.append(
            {
                "scene_id": scene_id,
                "name": item.get("name"),
                "source_preview_png": item.get("preview_png"),
                "source_metadata_json": item.get("metadata_json"),
                **copied,
            }
        )
        index_path.write_text(
            json.dumps({"schema": "blenderkit_classified_spatial_preview_v1", "count": len(rows), "items": rows}, indent=2),
            encoding="utf-8",
        )


def vector_from_list(values, Vector):
    return Vector((float(values[0]), float(values[1]), float(values[2])))


def bbox_max_extent(bounds: tuple) -> float:
    bbox_min, bbox_max = bounds
    extent = bbox_max - bbox_min
    return max(float(extent.x), float(extent.y), float(extent.z))


def padded_bbox(bounds: tuple, padding: float) -> tuple:
    bbox_min, bbox_max = bounds
    center = (bbox_min + bbox_max) * 0.5
    half = (bbox_max - bbox_min) * 0.5 * max(float(padding), 1.0)
    return center - half, center + half


def union_bboxes(bounds_list: list[tuple]):
    bbox_min = bounds_list[0][0].copy()
    bbox_max = bounds_list[0][1].copy()
    for cur_min, cur_max in bounds_list[1:]:
        bbox_min.x = min(bbox_min.x, cur_min.x)
        bbox_min.y = min(bbox_min.y, cur_min.y)
        bbox_min.z = min(bbox_min.z, cur_min.z)
        bbox_max.x = max(bbox_max.x, cur_max.x)
        bbox_max.y = max(bbox_max.y, cur_max.y)
        bbox_max.z = max(bbox_max.z, cur_max.z)
    return bbox_min, bbox_max


def candidate_bounds(candidate: dict, Vector):
    if not candidate.get("bbox_min") or not candidate.get("bbox_max"):
        return None
    return vector_from_list(candidate["bbox_min"], Vector), vector_from_list(candidate["bbox_max"], Vector)


def looks_like_large_support_object(name: str) -> bool:
    value = name.lower()
    return any(token in value for token in ("base", "ground", "terrain", "sky", "background", "floor", "plane", "cloud"))


def select_spatial_bounds(item: dict, full_bounds: tuple, args: argparse.Namespace, Vector) -> tuple:
    if args.spatial_bbox_mode == "full":
        return full_bounds[0], full_bounds[1], "full_scene_mesh_bbox"

    candidates = []
    for candidate in item.get("subject_candidates", []):
        bounds = candidate_bounds(candidate, Vector)
        if bounds is None:
            continue
        extent = bbox_max_extent(bounds)
        if extent <= 1e-8:
            continue
        candidates.append((candidate, bounds, extent))

    if not candidates:
        return full_bounds[0], full_bounds[1], "full_scene_mesh_bbox_no_candidates"

    extents = sorted(extent for _candidate, _bounds, extent in candidates)
    median_extent = extents[len(extents) // 2]
    filtered = []
    for candidate, bounds, extent in candidates:
        name = str(candidate.get("name", ""))
        if args.spatial_bbox_mode == "auto":
            if median_extent > 0.0 and extent > median_extent * float(args.candidate_outlier_factor):
                continue
            if (
                looks_like_large_support_object(name)
                and len(candidates) > 1
                and median_extent > 0.0
                and extent > median_extent * 3.0
            ):
                continue
        filtered.append((candidate, bounds, extent))
        if len(filtered) >= max(int(args.subject_candidate_count), 1):
            break

    if not filtered:
        filtered = candidates[: max(int(args.subject_candidate_count), 1)]

    bounds = padded_bbox(union_bboxes([bounds for _candidate, bounds, _extent in filtered]), float(args.candidate_padding))
    names = [str(candidate.get("name", "")) for candidate, _bounds, _extent in filtered]
    return bounds[0], bounds[1], f"{args.spatial_bbox_mode}_subject_candidates:{names}"


def canonical_depth_over_scale(config: dict) -> float:
    cam_cfg = config.get("camera", {})
    canonical_cfg = config.get("canonical", {})
    canonical_center = canonical_cfg.get("center", [0.0, 0.0, 0.0])
    position = cam_cfg.get("canonical_position")
    if position is not None:
        return max(abs(float(position[1]) - float(canonical_center[1])), 1e-6)
    if cam_cfg.get("canonical_distance") is not None:
        return max(abs(float(cam_cfg["canonical_distance"])), 1e-6)
    lo, hi = cam_cfg.get("canonical_distance_range", [4.5, 4.5])
    return max((abs(float(lo)) + abs(float(hi))) * 0.5, 1e-6)


def canonical_forward_front_extent(config: dict) -> float:
    cam_cfg = config.get("camera", {})
    canonical_cfg = config.get("canonical", {})
    canonical_center = canonical_cfg.get("center", [0.0, 0.0, 0.0])
    center_y = float(canonical_center[1])
    y_min, y_max = canonical_cfg.get("position_range", {}).get("y", [-1.0, 1.0])
    position = cam_cfg.get("canonical_position")
    camera_y = float(position[1]) if position is not None else center_y - 1.0
    if camera_y <= center_y:
        return max(center_y - float(y_min), 0.0)
    return max(float(y_max) - center_y, 0.0)


def safe_camera_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def camera_fov_snapshot(camera) -> dict:
    data = camera.data
    snapshot = {
        "type": getattr(data, "type", None),
        "angle_degrees": None,
        "angle_x_degrees": None,
        "angle_y_degrees": None,
        "lens": safe_camera_float(getattr(data, "lens", None)),
        "sensor_fit": getattr(data, "sensor_fit", None),
        "sensor_width": safe_camera_float(getattr(data, "sensor_width", None)),
        "sensor_height": safe_camera_float(getattr(data, "sensor_height", None)),
        "shift_x": safe_camera_float(getattr(data, "shift_x", None)),
        "shift_y": safe_camera_float(getattr(data, "shift_y", None)),
        "ortho_scale": safe_camera_float(getattr(data, "ortho_scale", None)),
    }
    for key, attr in (
        ("angle_degrees", "angle"),
        ("angle_x_degrees", "angle_x"),
        ("angle_y_degrees", "angle_y"),
    ):
        value = safe_camera_float(getattr(data, attr, None))
        if value is not None:
            snapshot[key] = math.degrees(value)
    return snapshot


def camera_framed_light_volume(
    config: dict,
    camera,
    bbox_center,
    relight,
    Vector,
    target_depth_over_scale: float | None,
    scene=None,
):
    _right, _up, forward = relight.camera_basis(camera)
    camera_location = Vector(camera.location)
    bbox_depth = float((bbox_center - camera_location).dot(forward))
    depth = max(bbox_depth, 0.1)
    target_ratio = float(target_depth_over_scale) if target_depth_over_scale is not None else canonical_depth_over_scale(config)
    target_ratio = max(target_ratio, 1e-6)
    canonical_fov = math.radians(float(config.get("camera", {}).get("fov_degrees", 39.6)))
    current_fov = float(getattr(camera.data, "angle_x", 0.0) or getattr(camera.data, "angle", canonical_fov))
    current_fov_y = float(getattr(camera.data, "angle_y", 0.0) or getattr(camera.data, "angle", current_fov))
    fov_scale_x = math.tan(current_fov * 0.5) / max(math.tan(canonical_fov * 0.5), 1e-6)
    fov_scale_y = math.tan(current_fov_y * 0.5) / max(math.tan(canonical_fov * 0.5), 1e-6)
    fov_scale = fov_scale_x
    front_extent = canonical_forward_front_extent(config)
    center_plane_scale = max((depth / target_ratio) * fov_scale, 1e-6)
    projected_bbox_denominator = max(target_ratio - front_extent + fov_scale * front_extent, 1e-6)
    scale = max((depth * fov_scale) / projected_bbox_denominator, 1e-6)
    light_center = camera_location + forward * depth
    if scene is not None:
        right, up, _forward = relight.camera_basis(camera)
        for _ in range(3):
            co = relight.world_to_camera_view(scene, camera, light_center)
            dx = 0.5 - float(co.x)
            dy = 0.5 - float(co.y)
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                break
            half_width = depth * math.tan(current_fov * 0.5)
            half_height = depth * math.tan(current_fov_y * 0.5)
            light_center = light_center + right * (dx * 2.0 * half_width) + up * (dy * 2.0 * half_height)
    shift = light_center - bbox_center

    config.setdefault("_runtime", {})["canonical_scale"] = scale
    adjustment = {
        "mode": "camera-framed",
        "bbox_center": relight.vec_to_list(bbox_center),
        "adjusted_center": relight.vec_to_list(light_center),
        "center_shift": relight.vec_to_list(shift),
        "camera_depth": depth,
        "bbox_camera_depth": bbox_depth,
        "target_depth_over_scale": target_ratio,
        "canonical_fov_degrees": math.degrees(canonical_fov),
        "current_fov_degrees": math.degrees(current_fov),
        "current_fov_y_degrees": math.degrees(current_fov_y),
        "fov_scale_x": fov_scale_x,
        "fov_scale_y": fov_scale_y,
        "fov_scale_axis": "x",
        "fov_scale": fov_scale,
        "scale_match": "projected_bbox_x",
        "canonical_front_extent": front_extent,
        "center_plane_scale": center_plane_scale,
        "projected_bbox_scale_multiplier": scale / max(center_plane_scale, 1e-6),
        "scale": scale,
    }
    config["_runtime"]["light_volume_center_source"] = "camera_axis_at_bbox_depth"
    config["_runtime"]["light_volume_adjustment"] = adjustment
    return light_center, adjustment


def render_debug_preview(
    scene_dir: Path,
    config: dict,
    rng: random.Random,
    camera,
    light_center,
    relight,
) -> dict:
    if config.get("_ambient_source") == "scene":
        source = relight.scene_ambient_source_meta()
    else:
        ambient = config["ambient"]
        hdri_path, hdri_mode = relight.choose_hdri_path(config, rng)
        hdri_strength = rng.uniform(*ambient.get("hdri_strength_range", [0.8, 1.2]))
        hdri_rotation = rng.random() * 2.0 * math.pi if ambient.get("hdri_rotation_random", True) else 0.0
        source = relight.set_hdri_world(hdri_path, hdri_strength, hdri_rotation, ambient.get("fallback_color", [0.78, 0.78, 0.78]))
        source["hdri_mode"] = hdri_mode
        relight.remove_all_lights()

    ambient_preview = f"../preview/{scene_dir.name}_ambient.png"
    relight.render_png(scene_dir / ambient_preview)

    positions = relight.sample_spatial_positions(config, rng)
    light_position_preview = None
    if config.get("_light_preview", False):
        light_position_preview = relight.render_light_position_preview(scene_dir, positions, config, camera, light_center)

    spatial = config["spatial"]
    return {
        "debug_preview_only": True,
        "ambient_png": ambient_preview,
        "light_position_preview": light_position_preview,
        "ambient_source": source,
        "light_volume_center": relight.vec_to_list(light_center),
        "light_volume_center_source": config.get("_runtime", {}).get("light_volume_center_source", "bbox_center"),
        "light_volume_adjustment": config.get("_runtime", {}).get("light_volume_adjustment"),
        "canonical_transform": relight.canonical_transform_meta(config, camera, light_center),
        "positions_per_scene": int(spatial.get("positions_per_scene", 64)),
        "position_sampling": spatial.get("sampling", "stratified_random"),
        "grid_resolution": spatial.get("grid_resolution"),
        "jitter": spatial.get("jitter"),
        "min_position_distance": spatial.get("min_position_distance"),
        "valid_point_light_count": 0,
        "invalid_point_light_count": 0,
        "point_lights": [],
    }


def worker(args: argparse.Namespace) -> int:
    script_dir = ROOT / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        import bpy
        from mathutils import Vector

        import render_object_relighting as relight
    except ModuleNotFoundError as exc:
        raise SystemExit("Worker must run inside Blender Python.") from exc

    item = json.loads(resolve_repo_path(args.item_json).read_text(encoding="utf-8"))
    item_id = str(item.get("id")).zfill(5)
    scene_id = f"blenderkit_{item_id}"
    config = load_runtime_config(resolve_repo_path(args.config), args)

    bpy.ops.wm.open_mainfile(filepath=str(resolve_repo_path(args.blend)))
    relight.setup_render_settings(config)

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects in {args.blend}")
    full_bbox_min, full_bbox_max = relight.mesh_bbox(mesh_objects)
    bbox_min, bbox_max, bbox_source = select_spatial_bounds(item, (full_bbox_min, full_bbox_max), args, Vector)
    bbox_center = (bbox_min + bbox_max) * 0.5
    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)

    camera = None
    if item.get("camera") and item["camera"] in bpy.data.objects:
        camera = bpy.data.objects[item["camera"]]
        bpy.context.scene.camera = camera
    elif bpy.context.scene.camera:
        camera = bpy.context.scene.camera
    else:
        rng_for_camera = random.Random(int(config["seed"]) + int(item_id))
        camera, _camera_meta = relight.create_camera(config, rng_for_camera, bbox_center)
    bpy.context.view_layer.update()

    light_center = bbox_center
    if args.light_volume_placement == "camera-framed":
        light_center, _light_volume_adjustment = camera_framed_light_volume(
            config,
            camera,
            bbox_center,
            relight,
            Vector,
            args.light_volume_depth_over_scale,
            bpy.context.scene,
        )
    else:
        config.setdefault("_runtime", {})["light_volume_center_source"] = "bbox_center"
        config["_runtime"]["light_volume_adjustment"] = {
            "mode": "bbox-center",
            "bbox_center": relight.vec_to_list(bbox_center),
            "adjusted_center": relight.vec_to_list(light_center),
            "center_shift": [0.0, 0.0, 0.0],
            "scale": relight.canonical_world_scale(config),
        }

    rng = random.Random(int(config["seed"]) + int(item_id))
    output_root = resolve_repo_path(args.output)
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        spatial_meta = render_debug_preview(scene_dir, config, rng, camera, light_center, relight)
    else:
        spatial_meta = relight.render_spatial_components(scene_dir, config, rng, camera, light_center)
    meta = {
        "schema": "tokenlight_synthetic_components_v1",
        "scene_id": scene_id,
        "scene_type": "blenderkit_scene_spatial",
        "source_blend": str(resolve_repo_path(args.blend)),
        "source_item": {
            "name": item.get("name"),
            "asset_id": item.get("asset_id"),
            "asset_base_id": item.get("asset_base_id"),
            "preview_png": item.get("preview_png"),
            "metadata_json": item.get("metadata_json"),
        },
        "object": {
            "path": str(resolve_repo_path(args.blend)),
            "primitive": None,
            "bbox_min": relight.vec_to_list(bbox_min),
            "bbox_max": relight.vec_to_list(bbox_max),
            "bbox_source": bbox_source,
            "full_scene_bbox_min": relight.vec_to_list(full_bbox_min),
            "full_scene_bbox_max": relight.vec_to_list(full_bbox_max),
            "center": relight.vec_to_list(bbox_center),
        },
        "camera": {
            "name": camera.name if camera else None,
            "location": relight.vec_to_list(Vector(camera.location)) if camera else None,
            "fov": camera_fov_snapshot(camera) if camera else None,
        },
        "render": {
            "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
            "samples": int(config["render"].get("samples", 128)),
            "engine": bpy.context.scene.render.engine,
            "linear_rgb": True,
            "tone_mapping_applied": False,
        },
        "spatial": spatial_meta,
    }
    relight.write_json(scene_dir / "meta.json", meta)
    copy_spatial_previews(args, scene_id, scene_dir, item)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
