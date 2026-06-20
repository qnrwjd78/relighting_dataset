from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_AGENT = "relighting-dataset-blenderkit-manual-preview/0.1"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description=(
            "Render three BlenderKit manual previews: source-camera colored subject, "
            "Objaverse-canonical camera, and canonical camera with TokenLight light-volume preview."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--blend", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--decision-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--decision-kind", choices=["object", "background"], default=None, help=argparse.SUPPRESS)

    parser.add_argument("--data-dir", default="data/blenderkit")
    parser.add_argument("--decisions-dir", default=None)
    parser.add_argument("--index-json", default=None)
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full.json")
    parser.add_argument("--output", default="outputs/previews/blenderkit_manual_render")
    parser.add_argument("--work-dir", default="outputs/work/blenderkit_manual_preview")
    parser.add_argument("--download-dir", default="data/blenderkit_manual_preview_cache")
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--modes", nargs="+", choices=["object", "background"], default=["object", "background"])
    parser.add_argument("--scene-id", action="append", default=None, help="Render only these ids, e.g. blenderkit_00059.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--positions-per-scene", type=int, default=None)
    parser.add_argument("--object-modes", nargs="+", default=["object_group_manual"])
    parser.add_argument("--object-confidence", nargs="+", default=["high", "medium"])
    parser.add_argument("--background-usability", nargs="+", default=["yes"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite-blend", action="store_true")
    parser.add_argument("--keep-blend", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args(argv)


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        return resolve_repo_path(args.api_key_file).read_text(encoding="utf-8").strip()
    return ""


def scene_id_number(scene_id: str) -> str:
    return scene_id.replace("blenderkit_", "").zfill(5)


def safe_blend_name(item: dict) -> str:
    original = item.get("original_blend_path")
    if original:
        return Path(original).name
    base = item.get("asset_base_id") or item.get("asset_id") or item.get("id")
    return f"{base}.blend"


def load_index(path: Path) -> dict[str, dict]:
    data = load_json(path)
    return {str(item.get("id")).zfill(5): item for item in data.get("items", [])}


def load_decisions(args: argparse.Namespace) -> list[tuple[str, dict]]:
    data_dir = resolve_repo_path(args.data_dir)
    decisions_dir = resolve_repo_path(args.decisions_dir) if args.decisions_dir else data_dir / "manual_scene_decisions"
    selected = set(args.scene_id or [])
    rows: list[tuple[str, dict]] = []

    if "object" in args.modes:
        path = decisions_dir / "single_scene_light_good_object_decisions.json"
        for row in load_json(path):
            if selected and row.get("scene_id") not in selected:
                continue
            if row.get("mode") not in set(args.object_modes):
                continue
            if row.get("confidence") not in set(args.object_confidence):
                continue
            rows.append(("object", row))

    if "background" in args.modes:
        path = decisions_dir / "background_good_placement_decisions.json"
        for row in load_json(path):
            if selected and row.get("scene_id") not in selected:
                continue
            if row.get("usable_as_background") not in set(args.background_usability):
                continue
            rows.append(("background", row))

    return rows[args.start : args.start + args.limit if args.limit is not None else None]


def headers(api_key: str, user_agent: str) -> dict[str, str]:
    result = {"User-Agent": user_agent}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def request_json(url: str, api_key: str, user_agent: str) -> dict:
    import urllib.request

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
    import urllib.request

    if path.exists() and not overwrite:
        print(f"[BlenderKitManualPreview] Exists: {path}")
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
    print(f"[BlenderKitManualPreview] Downloaded: {path}")


def worker_command(args: argparse.Namespace, kind: str, blend_path: Path, decision_path: Path) -> list[str]:
    cmd = shlex.split(args.blender_cmd) + [
        "-b",
        "--python",
        str(ROOT / "scripts" / "render_blenderkit_manual_previews.py"),
        "--",
        "--worker",
        "--blend",
        str(blend_path),
        "--decision-json",
        str(decision_path),
        "--decision-kind",
        kind,
        "--config",
        args.config,
        "--output",
        args.output,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
    ]
    if args.positions_per_scene is not None:
        cmd.extend(["--positions-per-scene", str(args.positions_per_scene)])
    return cmd


def orchestrate(args: argparse.Namespace) -> int:
    api_key = load_api_key(args)
    if not api_key:
        raise SystemExit("Missing BlenderKit API key. Set BLENDERKIT_API_KEY or pass --api-key-file.")

    data_dir = resolve_repo_path(args.data_dir)
    index_path = resolve_repo_path(args.index_json) if args.index_json else data_dir / "blenderkit_index.json"
    by_id = load_index(index_path)
    rows = load_decisions(args)
    if not rows:
        raise SystemExit("No manual decisions matched the requested filters.")

    download_dir = resolve_repo_path(args.download_dir)
    work_dir = resolve_repo_path(args.work_dir)
    output_root = resolve_repo_path(args.output)
    scene_uuid = str(uuid.uuid4())

    print(f"[BlenderKitManualPreview] Selected decisions: {len(rows)}")
    for index, (kind, decision) in enumerate(rows, 1):
        scene_id = str(decision["scene_id"])
        item_id = scene_id_number(scene_id)
        item = by_id.get(item_id)
        if not item:
            print(f"[BlenderKitManualPreview] Skip missing index item: {scene_id}", file=sys.stderr)
            continue
        scene_dir = output_root / kind / scene_id
        meta_path = scene_dir / "preview_meta.json"
        if args.skip_existing and meta_path.exists():
            print(f"[BlenderKitManualPreview] Skip existing {kind}/{scene_id}: {meta_path}")
            continue
        download_api_url = item.get("download_api_url") or item.get("record", {}).get("download_api_url")
        if not download_api_url:
            print(f"[BlenderKitManualPreview] Skip missing download_api_url: {scene_id}", file=sys.stderr)
            continue

        blend_path = download_dir / safe_blend_name(item)
        decision_payload = {"kind": kind, "decision": decision, "index_item": item}
        decision_path = work_dir / f"{kind}_{scene_id}_decision.json"
        write_json(decision_path, decision_payload)
        print(f"[BlenderKitManualPreview] {index}/{len(rows)} {kind} {scene_id} {item.get('name')}")
        try:
            resolved_url = resolve_download_url(download_api_url, api_key, args.user_agent, scene_uuid)
            download_file(resolved_url, blend_path, api_key, args.user_agent, args.overwrite_blend)
            cmd = worker_command(args, kind, blend_path, decision_path)
            print("[BlenderKitManualPreview] Render:", " ".join(shlex.quote(part) for part in cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[BlenderKitManualPreview] FAILED {kind}/{scene_id}: {exc}", file=sys.stderr)
            continue
        finally:
            if not args.keep_blend:
                blend_path.unlink(missing_ok=True)
                blend_path.with_suffix(blend_path.suffix + ".part").unlink(missing_ok=True)
                print(f"[BlenderKitManualPreview] Deleted blend: {blend_path}")
        time.sleep(args.sleep)
    return 0


def load_lines(path: Path) -> list[str]:
    if not path or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def load_runtime_config(config_path: Path, args: argparse.Namespace) -> dict:
    script_dir = ROOT / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import render_object_relighting as relight

    config = load_json(config_path)
    config["output_root"] = args.output
    config["render"]["resolution_x"] = args.width
    config["render"]["resolution_y"] = args.height
    config["render"]["samples"] = args.samples
    config["render"]["component_format"] = "png"
    if args.positions_per_scene is not None:
        config["spatial"]["positions_per_scene"] = args.positions_per_scene
    config["_component_format"] = "png"
    config["_ambient_source"] = "scene"
    config["_point_light_mode"] = "target"
    config["_hdri_mode"] = str(config.get("ambient", {}).get("hdri_mode", "on")).lower()
    config["_light_preview"] = True
    config["_render_pbr"] = False

    object_manifest = relight.resolve_path(ROOT, config.get("object_manifest"))
    hdri_manifest = relight.resolve_path(ROOT, config.get("hdri_manifest"))
    receiver_texture_manifest = relight.resolve_path(
        ROOT,
        config.get("receiver_texture_manifest") or config.get("layout", {}).get("receiver_texture_manifest"),
    )
    config["_runtime"] = {
        "objects": relight.load_path_lines(object_manifest, ROOT) if object_manifest else [],
        "hdris": relight.load_path_lines(hdri_manifest, ROOT) if hdri_manifest else [],
        "receiver_textures": relight.load_receiver_texture_manifest(receiver_texture_manifest, ROOT)
        if receiver_texture_manifest
        else [],
        "receiver_bounds": None,
        "receiver_materials": [],
    }
    return config


def make_debug_material(name: str):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.85, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.35
        bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def colorized_render(path: Path, subject_objects: list) -> None:
    import bpy
    import render_object_relighting as relight

    debug_mat = make_debug_material("TL_manual_subject_magenta")
    originals = {obj.name: list(obj.data.materials) for obj in subject_objects}
    try:
        for obj in subject_objects:
            obj.data.materials.clear()
            obj.data.materials.append(debug_mat)
        relight.render_png(path)
    finally:
        for obj in subject_objects:
            obj.data.materials.clear()
            for mat in originals.get(obj.name, []):
                obj.data.materials.append(mat)


def select_objects_by_name(names: list[str]) -> list:
    import bpy

    result = []
    missing = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "MESH":
            result.append(obj)
        else:
            missing.append(name)
    if not result:
        raise RuntimeError(f"No selected candidate mesh objects were found. Missing: {missing}")
    return result


def translate_subject_to_placement(subject_objects: list, placement_world: list[float]) -> None:
    from mathutils import Matrix, Vector
    import render_object_relighting as relight

    bbox_min, _bbox_max = relight.mesh_bbox(subject_objects)
    placement = Vector((float(placement_world[0]), float(placement_world[1]), float(placement_world[2])))
    shift = placement - Vector((0.0, 0.0, float(bbox_min.z)))
    relight.apply_world_transform(subject_objects, Matrix.Translation(shift))


def render_manual_worker(args: argparse.Namespace) -> int:
    script_dir = ROOT / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        import bpy
        from mathutils import Vector
        import render_object_relighting as relight
    except ModuleNotFoundError as exc:
        raise SystemExit("Worker must run inside Blender Python.") from exc

    payload = load_json(resolve_repo_path(args.decision_json))
    decision = payload["decision"]
    item = payload.get("index_item", {})
    kind = args.decision_kind or payload["kind"]
    scene_id = str(decision["scene_id"])
    item_id = int(scene_id_number(scene_id))
    rng = random.Random(int(load_json(resolve_repo_path(args.config)).get("seed", 0)) + item_id)
    config = load_runtime_config(resolve_repo_path(args.config), args)

    bpy.ops.wm.open_mainfile(filepath=str(resolve_repo_path(args.blend)))
    relight.setup_render_settings(config)
    scene_dir = resolve_repo_path(args.output) / kind / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if kind == "object":
        subject_objects = select_objects_by_name(list(decision.get("candidate_objects") or []))
        object_meta = {
            "source": "blenderkit_scene_candidate_objects",
            "candidate_objects": [obj.name for obj in subject_objects],
            "manual_mode": decision.get("mode"),
            "manual_confidence": decision.get("confidence"),
        }
    else:
        objects = config["_runtime"].get("objects") or []
        asset = rng.choice(objects) if objects else None
        primitive = rng.choice(config.get("object", {}).get("primitive_fallbacks", ["sphere"]))
        subject_objects = relight.import_asset_or_primitive(asset, primitive, rng, config)
        translate_subject_to_placement(subject_objects, decision["placement_world"])
        object_meta = {
            "source": "random_objaverse_object",
            "path": asset,
            "primitive": None if asset else primitive,
            "placement_world": decision.get("placement_world"),
            "anchor_mesh": decision.get("anchor_mesh"),
            "usable_as_background": decision.get("usable_as_background"),
            "orientation": config.get("_runtime", {}).get("object_orientation"),
            "import_adjustments": config.get("_runtime", {}).get("object_import_adjustments"),
        }

    if bpy.context.scene.camera is None:
        bbox_min, bbox_max = relight.mesh_bbox(subject_objects)
        center = (bbox_min + bbox_max) * 0.5
        relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)
        relight.create_camera(config, rng, center)

    source_camera_name = bpy.context.scene.camera.name if bpy.context.scene.camera else None
    colored_path = scene_dir / "preview_000_source_camera_colored_object.png"
    colorized_render(colored_path, subject_objects)

    bbox_min, bbox_max = relight.mesh_bbox(subject_objects)
    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)
    center = (bbox_min + bbox_max) * 0.5
    canonical_camera, camera_meta = relight.create_camera(config, rng, center)
    bpy.context.scene.camera = canonical_camera
    bpy.context.view_layer.update()

    canonical_path = scene_dir / "preview_001_objaverse_canonical.png"
    relight.render_png(canonical_path)

    positions = relight.sample_spatial_positions(config, rng)
    light_rel = relight.render_light_position_preview(scene_dir, positions, config, canonical_camera, center)
    light_src = (scene_dir / light_rel).resolve()
    light_dst = scene_dir / "preview_002_objaverse_canonical_light_preview.png"
    if light_src.exists():
        shutil.copyfile(light_src, light_dst)

    full_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    full_bbox_min, full_bbox_max = relight.mesh_bbox(full_meshes)
    meta = {
        "schema": "blenderkit_manual_preview_v1",
        "scene_id": scene_id,
        "kind": kind,
        "source_blend": str(resolve_repo_path(args.blend)),
        "source_item": {
            "name": item.get("name"),
            "asset_id": item.get("asset_id"),
            "asset_base_id": item.get("asset_base_id"),
            "preview_png": item.get("preview_png"),
        },
        "manual_decision": decision,
        "object": {
            **object_meta,
            "bbox_min": relight.vec_to_list(bbox_min),
            "bbox_max": relight.vec_to_list(bbox_max),
            "center": relight.vec_to_list(center),
        },
        "full_scene_bbox": {
            "min": relight.vec_to_list(full_bbox_min),
            "max": relight.vec_to_list(full_bbox_max),
        },
        "source_camera": {"name": source_camera_name},
        "canonical_camera": camera_meta,
        "canonical": config["canonical"],
        "spatial_preview": {
            "positions_per_scene": len(positions),
            "light_position_preview": repo_relative(light_dst),
            "light_volume_center": relight.vec_to_list(center),
            "canonical_transform": relight.canonical_transform_meta(config, canonical_camera, center),
        },
        "renders": {
            "source_camera_colored_object": repo_relative(colored_path),
            "objaverse_canonical": repo_relative(canonical_path),
            "objaverse_canonical_light_preview": repo_relative(light_dst),
        },
        "render": {
            "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
            "samples": int(config["render"].get("samples", 128)),
            "engine": bpy.context.scene.render.engine,
            "fov_degrees": config["camera"].get("fov_degrees", 39.6),
        },
    }
    relight.write_json(scene_dir / "preview_meta.json", meta)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return render_manual_worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
