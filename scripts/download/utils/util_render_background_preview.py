from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("Run this script with Blender Python.") from exc


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render a camera preview PNG and metadata for one .blend scene.")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["current", "eevee", "cycles", "workbench"], default="current")
    parser.add_argument("--hdri-manifest", default=None)
    parser.add_argument("--hdri-strength", type=float, default=1.0)
    parser.add_argument("--hdri-seed", type=int, default=0)
    return parser.parse_args(argv)


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "scene"


def vec_to_list(v: Vector) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def read_hdri_manifest(path: str | Path) -> list[Path]:
    manifest = resolve_path(path)
    paths = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hdri = resolve_path(line)
        if hdri.suffix.lower() in {".hdr", ".exr"} and hdri.exists():
            paths.append(hdri)
    return paths


def select_hdri(manifest: str | None, blend: Path, seed: int) -> Path | None:
    if not manifest:
        return None
    paths = read_hdri_manifest(manifest)
    if not paths:
        raise RuntimeError(f"No existing HDRI files found in manifest: {manifest}")
    digest = hashlib.sha256(f"{seed}:{blend}".encode("utf-8")).hexdigest()
    return paths[int(digest[:16], 16) % len(paths)]


def setup_hdri_world(hdri: Path, strength: float) -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("TL_preview_world")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(hdri), check_existing=True)
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def object_world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector] | None:
    if obj.type != "MESH" or not obj.bound_box:
        return None
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxs = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return mins, maxs


def scene_bbox(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    boxes = [object_world_bbox(obj) for obj in meshes]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    mins = Vector((min(box[0].x for box in boxes), min(box[0].y for box in boxes), min(box[0].z for box in boxes)))
    maxs = Vector((max(box[1].x for box in boxes), max(box[1].y for box in boxes), max(box[1].z for box in boxes)))
    return mins, maxs


def bbox_volume(bounds: tuple[Vector, Vector]) -> float:
    extent = bounds[1] - bounds[0]
    return max(float(extent.x * extent.y * extent.z), 0.0)


def bbox_center(bounds: tuple[Vector, Vector]) -> Vector:
    return (bounds[0] + bounds[1]) * 0.5


def ranked_subject_candidates(meshes: list[bpy.types.Object], limit: int = 12) -> list[dict]:
    candidates = []
    for obj in meshes:
        bounds = object_world_bbox(obj)
        if not bounds:
            continue
        volume = bbox_volume(bounds)
        if volume <= 0.0:
            continue
        center = bbox_center(bounds)
        distance_to_origin = math.sqrt(center.x * center.x + center.y * center.y + center.z * center.z)
        candidates.append((volume / (1.0 + distance_to_origin), obj, bounds, volume, center))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "name": obj.name,
            "volume": float(volume),
            "bbox_min": vec_to_list(bounds[0]),
            "bbox_max": vec_to_list(bounds[1]),
            "center": vec_to_list(center),
        }
        for _, obj, bounds, volume, center in candidates[:limit]
    ]


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_camera(meshes: list[bpy.types.Object], bounds: tuple[Vector, Vector] | None) -> tuple[bpy.types.Object | None, bool]:
    if bpy.context.scene.camera:
        return bpy.context.scene.camera, False
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if cameras:
        bpy.context.scene.camera = cameras[0]
        return cameras[0], False
    if not bounds:
        return None, False

    center = bbox_center(bounds)
    extent = bounds[1] - bounds[0]
    radius = max(extent.length * 0.5, 1.0)
    location = center + Vector((0.0, -radius * 2.6, radius * 0.7))
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "TL_preview_camera"
    camera.data.lens = 35.0
    look_at(camera, center)
    bpy.context.scene.camera = camera
    return camera, True


def setup_render(width: int, height: int, samples: int, engine: str) -> None:
    scene = bpy.context.scene
    if engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif engine == "cycles":
        scene.render.engine = "CYCLES"
    elif engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    elif scene.render.engine == "BLENDER_EEVEE_NEXT" and hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = samples


def main() -> int:
    args = parse_args()
    blend = Path(args.blend).resolve()
    preview = Path(args.preview).resolve()
    metadata = Path(args.metadata).resolve()
    preview.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    lights = [obj.name for obj in bpy.data.objects if obj.type == "LIGHT"]
    cameras = [obj.name for obj in bpy.data.objects if obj.type == "CAMERA"]
    bounds = scene_bbox(meshes)
    camera, camera_created = ensure_camera(meshes, bounds)
    setup_render(args.width, args.height, args.samples, args.engine)
    hdri = select_hdri(args.hdri_manifest, blend, args.hdri_seed)
    if hdri is not None:
        setup_hdri_world(hdri, args.hdri_strength)

    row = {
        "source_blend": str(blend),
        "scene_id": slug(blend.stem),
        "preview_png": str(preview),
        "render_size": [args.width, args.height],
        "camera": camera.name if camera else None,
        "camera_created": camera_created,
        "cameras": cameras,
        "lights": lights,
        "mesh_count": len(meshes),
        "bbox_min": vec_to_list(bounds[0]) if bounds else None,
        "bbox_max": vec_to_list(bounds[1]) if bounds else None,
        "hdri": str(hdri) if hdri else None,
        "hdri_strength": args.hdri_strength if hdri else None,
        "subject_candidates": ranked_subject_candidates(meshes),
    }

    if camera:
        bpy.context.scene.render.filepath = str(preview)
        bpy.ops.render.render(write_still=True)
    else:
        row["render_error"] = "no_camera_or_mesh_bounds"

    metadata.write_text(json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[Preview] Wrote {preview}")
    print(f"[Preview] Wrote {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
