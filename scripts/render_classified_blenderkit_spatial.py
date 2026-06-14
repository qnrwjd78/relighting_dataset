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
    parser.add_argument("--hdri-mode", choices=["on", "off", "random"], default="on")
    parser.add_argument("--light-preview", action="store_true", default=True)
    parser.add_argument("--no-light-preview", action="store_false", dest="light_preview")
    parser.add_argument("--positions-per-scene", type=int, default=None)
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
        "--hdri-mode",
        args.hdri_mode,
    ]
    if args.positions_per_scene is not None:
        cmd.extend(["--positions-per-scene", str(args.positions_per_scene)])
    cmd.extend(["--spatial-bbox-mode", args.spatial_bbox_mode])
    cmd.extend(["--subject-candidate-count", str(args.subject_candidate_count)])
    cmd.extend(["--candidate-padding", str(args.candidate_padding)])
    cmd.extend(["--candidate-outlier-factor", str(args.candidate_outlier_factor)])
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
    config["_component_format"] = args.component_format
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
    center = (bbox_min + bbox_max) * 0.5
    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)

    camera = None
    if item.get("camera") and item["camera"] in bpy.data.objects:
        camera = bpy.data.objects[item["camera"]]
        bpy.context.scene.camera = camera
    elif bpy.context.scene.camera:
        camera = bpy.context.scene.camera
    else:
        rng_for_camera = random.Random(int(config["seed"]) + int(item_id))
        camera, _camera_meta = relight.create_camera(config, rng_for_camera, center)

    rng = random.Random(int(config["seed"]) + int(item_id))
    output_root = resolve_repo_path(args.output)
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    spatial_meta = relight.render_spatial_components(scene_dir, config, rng, camera, center)
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
            "center": relight.vec_to_list(center),
        },
        "camera": {
            "name": camera.name if camera else None,
            "location": relight.vec_to_list(Vector(camera.location)) if camera else None,
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
