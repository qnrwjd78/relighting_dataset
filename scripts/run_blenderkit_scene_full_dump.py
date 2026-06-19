from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_AGENT = "relighting-dataset-blenderkit-scene-full-dump/0.1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_classified_blenderkit_spatial import (  # noqa: E402
    download_file,
    resolve_download_url,
    safe_blend_name,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "For BlenderKit scenes, create scene_full_dump.json and object_id_overlay.png. "
            "In normal mode this downloads one .blend at a time, runs Blender, and deletes the .blend."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--dump-only", action="store_true", help="Write scene_full_dump.json only; skip object_id_overlay.png rendering.")

    parser.add_argument("--classification-csv", default="outputs/previews/blenderkit/blenderkit_preview_3way_candidates.csv")
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument("--category", default="object_candidate")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional ids like blenderkit_00059 or 00059.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-root", default="outputs/previews/blenderkit/scene_full_dump")
    parser.add_argument("--download-dir", default="outputs/work/blenderkit_scene_full_dump/blends")
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default="blenderkit_key.txt")
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--overwrite-blend", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-blend", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args(argv)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def normalize_id(value: str) -> str:
    value = value.strip()
    if value.startswith("blenderkit_"):
        value = value[len("blenderkit_") :]
    return value.zfill(5)


def scene_id(item_id: str) -> str:
    return f"blenderkit_{normalize_id(item_id)}"


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        path = resolve_repo_path(args.api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def load_classified_ids(path: Path, category: str) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("category") == category:
                ids.append(normalize_id(row["id"]))
    return ids


def load_items(args: argparse.Namespace) -> list[dict]:
    index = json.loads(resolve_repo_path(args.index_json).read_text(encoding="utf-8"))
    by_id = {normalize_id(str(item.get("id"))): item for item in index.get("items", [])}
    if args.ids:
        ids = [normalize_id(value) for value in args.ids]
    else:
        ids = load_classified_ids(resolve_repo_path(args.classification_csv), args.category)
    items = [by_id[item_id] for item_id in ids if item_id in by_id]
    end = args.start + args.limit if args.limit is not None else None
    return items[args.start : end]


def run_command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def worker_command(args: argparse.Namespace, blend_path: Path, item: dict, out_dir: Path) -> list[str]:
    item_id = normalize_id(str(item.get("id")))
    metadata_json = item.get("metadata_json") or f"outputs/previews/blenderkit/metadata/blenderkit_{item_id}.json"
    cmd = shlex.split(args.blender_cmd) + [
        "--background",
        str(blend_path),
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--worker",
        "--scene-id",
        scene_id(item_id),
        "--metadata-json",
        str(resolve_repo_path(metadata_json)),
        "--out-dir",
        str(out_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
    ]
    if args.dump_only:
        cmd.append("--dump-only")
    return cmd


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene_id", "name", "status", "message", "out_dir", "scene_full_dump", "object_id_overlay", "blend_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def process_item(args: argparse.Namespace, item: dict, api_key: str, scene_uuid: str) -> dict:
    item_id = normalize_id(str(item.get("id")))
    sid = scene_id(item_id)
    out_dir = resolve_repo_path(args.out_root) / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "scene_id": sid,
        "name": item.get("name", ""),
        "status": "pending",
        "message": "",
        "out_dir": out_dir.as_posix(),
        "scene_full_dump": (out_dir / "scene_full_dump.json").as_posix(),
        "object_id_overlay": (out_dir / "object_id_overlay.png").as_posix(),
        "blend_path": "",
    }
    dump_exists = (out_dir / "scene_full_dump.json").exists()
    overlay_exists = (out_dir / "object_id_overlay.png").exists()
    existing_outputs_ready = dump_exists if args.dump_only else dump_exists and overlay_exists
    if args.skip_existing and not args.overwrite and existing_outputs_ready:
        row["status"] = "ok"
        row["message"] = "skip_existing"
        return row

    download_api_url = item.get("download_api_url") or item.get("record", {}).get("download_api_url")
    if not download_api_url:
        row["status"] = "failed"
        row["message"] = "missing_download_api_url"
        return row

    blend_path = out_dir / "source.blend" if args.keep_blend else resolve_repo_path(args.download_dir) / safe_blend_name(item)
    row["blend_path"] = blend_path.as_posix()
    cmd = worker_command(args, blend_path, item, out_dir)
    if args.dry_run:
        row["status"] = "dry_run"
        row["message"] = run_command_text(cmd)
        return row

    try:
        resolved_url = resolve_download_url(download_api_url, api_key, args.user_agent, scene_uuid)
        download_file(resolved_url, blend_path, api_key, args.user_agent, args.overwrite_blend)
        print(f"[SceneDump] Worker {sid}: {run_command_text(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)
        row["status"] = "ok"
        row["message"] = "dumped"
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, subprocess.CalledProcessError) as exc:
        row["status"] = "failed"
        row["message"] = str(exc)
        print(f"[SceneDump] FAILED {sid}: {exc}", file=sys.stderr)
    finally:
        if not args.keep_blend and not args.dry_run:
            blend_path.unlink(missing_ok=True)
            blend_path.with_suffix(blend_path.suffix + ".part").unlink(missing_ok=True)
            print(f"[SceneDump] Deleted blend: {blend_path}")
    return row


def orchestrate(args: argparse.Namespace) -> int:
    api_key = load_api_key(args)
    if not api_key and not args.dry_run:
        raise SystemExit("Missing BlenderKit API key. Set BLENDERKIT_API_KEY or pass --api-key-file.")
    items = load_items(args)
    if not items:
        raise SystemExit("No items selected.")

    out_root = resolve_repo_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "scene_full_dump_manifest.csv"
    scene_uuid = str(uuid.uuid4())
    rows: list[dict] = []

    print(f"[SceneDump] Selected items: {len(items)}")
    for index, item in enumerate(items, 1):
        sid = scene_id(str(item.get("id")))
        print(f"[SceneDump] {index}/{len(items)} {sid} {item.get('name', '')}")
        rows.append(process_item(args, item, api_key, scene_uuid))
        write_manifest(manifest, rows)
        time.sleep(args.sleep)
    write_manifest(manifest, rows)
    print(f"[SceneDump] Wrote manifest: {manifest}")
    return 0


def run_worker(args: argparse.Namespace) -> int:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if not scene.camera:
        cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]
        if cameras:
            scene.camera = cameras[0]
    active_camera = scene.camera

    def vec(value) -> list[float]:
        return [float(x) for x in value]

    def matrix_rows(matrix) -> list[list[float]]:
        return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]

    def world_bbox(obj) -> tuple[Vector, Vector, list[Vector]]:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        points = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
        return (
            Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
            Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
            points,
        )

    def screen_bbox(points: list[Vector], camera) -> dict | None:
        if not camera:
            return None
        coords = [world_to_camera_view(scene, camera, point) for point in points]
        xs = [float(c.x) for c in coords]
        ys = [float(c.y) for c in coords]
        zs = [float(c.z) for c in coords]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return {
            "normalized_bbox": [x0, y0, x1, y1],
            "normalized_center": [(x0 + x1) * 0.5, (y0 + y1) * 0.5],
            "normalized_area": max(0.0, x1 - x0) * max(0.0, y1 - y0),
            "depth_min": min(zs),
            "depth_max": max(zs),
            "any_corner_in_frame": any(0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.z > 0.0 for c in coords),
        }

    def material_info(mat) -> dict:
        info = {
            "name": mat.name if mat else None,
            "uses_nodes": bool(mat and mat.use_nodes),
            "diffuse_color": vec(mat.diffuse_color) if mat else None,
            "base_color": None,
            "metallic": None,
            "roughness": None,
            "alpha": None,
            "has_emission": False,
            "emission_color": None,
            "emission_strength": 0.0,
            "image_textures": [],
        }
        if not mat or not mat.use_nodes:
            return info
        for node in mat.node_tree.nodes:
            if node.bl_idname == "ShaderNodeBsdfPrincipled":
                for input_name, out_key in (("Base Color", "base_color"), ("Metallic", "metallic"), ("Roughness", "roughness"), ("Alpha", "alpha")):
                    socket = node.inputs.get(input_name)
                    if socket:
                        value = socket.default_value
                        info[out_key] = vec(value) if hasattr(value, "__len__") else float(value)
                emission_strength = node.inputs.get("Emission Strength")
                if emission_strength and float(emission_strength.default_value) > 0.0:
                    info["has_emission"] = True
                    info["emission_strength"] = float(emission_strength.default_value)
                emission_color = node.inputs.get("Emission Color")
                if emission_color:
                    info["emission_color"] = vec(emission_color.default_value)
            elif node.bl_idname == "ShaderNodeEmission":
                info["has_emission"] = True
                color = node.inputs.get("Color")
                strength = node.inputs.get("Strength")
                info["emission_color"] = vec(color.default_value) if color else None
                info["emission_strength"] = float(strength.default_value) if strength else 0.0
            elif node.bl_idname == "ShaderNodeTexImage":
                image = node.image
                info["image_textures"].append(
                    {
                        "node_name": node.name,
                        "image_name": image.name if image else None,
                        "filepath": image.filepath if image else None,
                        "size": list(image.size) if image else None,
                    }
                )
        return info

    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    mesh_entries = []
    scene_points = []
    for mesh_id, obj in enumerate(meshes, 1):
        bbox_min, bbox_max, points = world_bbox(obj)
        scene_points.extend(points)
        extent = bbox_max - bbox_min
        materials = [material_info(slot.material) for slot in obj.material_slots]
        entry = {
            "mesh_id": mesh_id,
            "name": obj.name,
            "data_name": obj.data.name if obj.data else None,
            "type": obj.type,
            "collection_names": [collection.name for collection in obj.users_collection],
            "parent": obj.parent.name if obj.parent else None,
            "children": [child.name for child in obj.children],
            "visible_viewport": bool(obj.visible_get()),
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
            "location": vec(obj.location),
            "rotation_euler": vec(obj.rotation_euler),
            "scale": vec(obj.scale),
            "matrix_world": matrix_rows(obj.matrix_world),
            "bbox_min_world": vec(bbox_min),
            "bbox_max_world": vec(bbox_max),
            "bbox_center_world": vec((bbox_min + bbox_max) * 0.5),
            "bbox_extent_world": vec(extent),
            "bbox_volume_world": float(max(extent.x, 0.0) * max(extent.y, 0.0) * max(extent.z, 0.0)),
            "vertex_count": len(obj.data.vertices) if obj.data else 0,
            "edge_count": len(obj.data.edges) if obj.data else 0,
            "face_count": len(obj.data.polygons) if obj.data else 0,
            "material_slot_names": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "materials": materials,
            "active_camera_projection": screen_bbox(points, active_camera),
            "distance_to_active_camera": float(((bbox_min + bbox_max) * 0.5 - active_camera.location).length) if active_camera else None,
        }
        mesh_entries.append(entry)

    if scene_points:
        scene_bbox_min = Vector((min(p.x for p in scene_points), min(p.y for p in scene_points), min(p.z for p in scene_points)))
        scene_bbox_max = Vector((max(p.x for p in scene_points), max(p.y for p in scene_points), max(p.z for p in scene_points)))
    else:
        scene_bbox_min = Vector((0.0, 0.0, 0.0))
        scene_bbox_max = Vector((0.0, 0.0, 0.0))

    cameras = []
    for obj in scene.objects:
        if obj.type != "CAMERA":
            continue
        cameras.append(
            {
                "name": obj.name,
                "is_active_scene_camera": obj == active_camera,
                "location": vec(obj.location),
                "rotation_euler": vec(obj.rotation_euler),
                "scale": vec(obj.scale),
                "matrix_world": matrix_rows(obj.matrix_world),
                "lens": float(obj.data.lens),
                "angle": float(obj.data.angle),
                "angle_degrees": math.degrees(float(obj.data.angle)),
                "angle_x": float(obj.data.angle_x),
                "angle_y": float(obj.data.angle_y),
                "clip_start": float(obj.data.clip_start),
                "clip_end": float(obj.data.clip_end),
                "sensor_width": float(obj.data.sensor_width),
                "sensor_height": float(obj.data.sensor_height),
            }
        )

    lights = []
    for obj in scene.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        lights.append(
            {
                "name": obj.name,
                "light_type": data.type,
                "location": vec(obj.location),
                "rotation_euler": vec(obj.rotation_euler),
                "scale": vec(obj.scale),
                "matrix_world": matrix_rows(obj.matrix_world),
                "energy": float(data.energy),
                "color": vec(data.color),
                "size": float(getattr(data, "size", 0.0)),
                "spot_size": float(getattr(data, "spot_size", 0.0)),
                "spot_blend": float(getattr(data, "spot_blend", 0.0)),
            }
        )

    metadata = {}
    if args.metadata_json and Path(args.metadata_json).exists():
        metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))

    dump = {
        "scene_id": args.scene_id,
        "source_blend": bpy.data.filepath,
        "render_resolution": [int(args.width), int(args.height)],
        "scene_bbox": {
            "min": vec(scene_bbox_min),
            "max": vec(scene_bbox_max),
            "center": vec((scene_bbox_min + scene_bbox_max) * 0.5),
            "extent": vec(scene_bbox_max - scene_bbox_min),
        },
        "active_camera": active_camera.name if active_camera else None,
        "cameras": cameras,
        "lights": lights,
        "meshes": mesh_entries,
        "source_metadata": metadata,
    }
    dump_path = out_dir / "scene_full_dump.json"
    dump_path.write_text(json.dumps(dump, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.dump_only:
        print(json.dumps({"scene_full_dump": dump_path.as_posix(), "object_id_overlay": None}, indent=2))
        return 0

    # Render an object-id overlay with one flat emission color per mesh. The JSON stores
    # the exact mesh_id -> color -> Blender object-name mapping.
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    try:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        for device in bpy.context.preferences.addons["cycles"].preferences.devices:
            device.use = True
    except Exception:
        scene.cycles.device = "CPU"
    if active_camera:
        scene.camera = active_camera

    original_materials = {obj.name: list(obj.data.materials) for obj in meshes}
    original_hide_render = {obj.name: obj.hide_render for obj in scene.objects}
    overlay_entries = []
    try:
        for obj in scene.objects:
            if obj.type not in {"MESH", "CAMERA"}:
                obj.hide_render = True
        for entry, obj in zip(mesh_entries, meshes):
            hue = ((entry["mesh_id"] * 0.61803398875) % 1.0)
            color = hsv_to_rgb(hue, 0.82, 1.0)
            mat = bpy.data.materials.new(f"TL_object_id_{entry['mesh_id']:04d}_{obj.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
            emission.inputs["Strength"].default_value = 1.0
            output = nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            entry["object_id_overlay"] = {
                "mesh_id": entry["mesh_id"],
                "color_rgb": color,
                "color_rgb_255": [int(round(channel * 255.0)) for channel in color],
            }
            overlay_entries.append({"mesh_id": entry["mesh_id"], "name": obj.name, "color_rgb": color})
        scene.world.color = (0.0, 0.0, 0.0)
        scene.render.filepath = str(out_dir / "object_id_overlay.png")
        bpy.ops.render.render(write_still=True)
    finally:
        for obj in meshes:
            obj.data.materials.clear()
            for mat in original_materials[obj.name]:
                obj.data.materials.append(mat)
        for obj in scene.objects:
            obj.hide_render = original_hide_render.get(obj.name, obj.hide_render)

    dump["meshes"] = mesh_entries
    dump["object_id_overlay"] = {
        "path": (out_dir / "object_id_overlay.png").as_posix(),
        "entries": overlay_entries,
    }
    dump_path.write_text(json.dumps(dump, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"scene_full_dump": dump_path.as_posix(), "object_id_overlay": (out_dir / "object_id_overlay.png").as_posix()}, indent=2))
    return 0


def hsv_to_rgb(h: float, s: float, v: float) -> list[float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return [float(r), float(g), float(b)]


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
