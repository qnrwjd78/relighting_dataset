from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Debug BlenderKit object candidates in a .blend scene.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full.json")
    return parser.parse_args(argv)


def world_bbox(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points: list[Vector] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        eval_obj = obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            points.append(eval_obj.matrix_world @ Vector(corner))
    if not points:
        raise RuntimeError("No bbox points for selected objects.")
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def vec(v: Vector) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.45) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def make_emission_mat(name: str, color: tuple[float, float, float, float], strength: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def is_excluded_mesh_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("plane", "floor", "wall", "ceiling", "backdrop", "background", "room"))


def choose_candidate_objects() -> tuple[list[bpy.types.Object], list[bpy.types.Object], str]:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.visible_get()]
    candidates = [obj for obj in meshes if not is_excluded_mesh_name(obj.name)]
    support_like = []
    for obj in candidates:
        bbox_min, bbox_max = world_bbox([obj])
        extent = bbox_max - bbox_min
        horizontal = max(float(extent.x), float(extent.y), 1e-6)
        if float(extent.z) < horizontal * 0.28 and horizontal > 0.12:
            support_like.append(obj)
    if len(candidates) - len(support_like) >= 1:
        candidates = [obj for obj in candidates if obj not in support_like]
    if not candidates:
        candidates = meshes
        reason = "fallback_all_visible_meshes"
    else:
        reason = "visible_meshes_excluding_environment_and_flat_support_like_meshes"
    excluded = [obj for obj in meshes if obj not in candidates]
    return candidates, excluded, reason


def setup_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        for device in bpy.context.preferences.addons["cycles"].preferences.devices:
            device.use = True
    except Exception:
        scene.cycles.device = "CPU"


def render_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def colorize_candidate_preview(candidates: list[bpy.types.Object], out_path: Path) -> None:
    debug_mat = make_mat("TL_candidate_debug_magenta", (1.0, 0.0, 0.85, 1.0))
    originals = {obj.name: list(obj.data.materials) for obj in candidates}
    try:
        for obj in candidates:
            obj.data.materials.clear()
            obj.data.materials.append(debug_mat)
        render_png(out_path)
    finally:
        for obj in candidates:
            obj.data.materials.clear()
            for mat in originals[obj.name]:
                obj.data.materials.append(mat)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def duplicate_candidate_group(candidates: list[bpy.types.Object], target_center: Vector, target_max_extent: float) -> list[bpy.types.Object]:
    bbox_min, bbox_max = world_bbox(candidates)
    center = (bbox_min + bbox_max) * 0.5
    max_extent = max((bbox_max - bbox_min).x, (bbox_max - bbox_min).y, (bbox_max - bbox_min).z, 1e-6)
    scale = target_max_extent / max_extent
    duplicates: list[bpy.types.Object] = []
    for obj in candidates:
        dup = obj.copy()
        dup.data = obj.data.copy()
        dup.animation_data_clear()
        dup.matrix_world = (
            Matrix.Translation(target_center)
            @ Matrix.Scale(scale, 4)
            @ Matrix.Translation(-center)
            @ obj.matrix_world
        )
        bpy.context.collection.objects.link(dup)
        duplicates.append(dup)
    return duplicates


def hide_all_except(keep: list[bpy.types.Object]) -> dict[str, tuple[bool, bool]]:
    keep_set = set(keep)
    states = {}
    for obj in bpy.context.scene.objects:
        states[obj.name] = (obj.hide_viewport, obj.hide_render)
        obj.hide_viewport = obj not in keep_set and obj.type != "CAMERA"
        obj.hide_render = obj not in keep_set and obj.type != "CAMERA"
    return states


def restore_visibility(states: dict[str, tuple[bool, bool]]) -> None:
    for name, (hide_viewport, hide_render) in states.items():
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_viewport = hide_viewport
            obj.hide_render = hide_render


def create_debug_cube(size: float = 2.0) -> list[bpy.types.Object]:
    mat = make_emission_mat("TL_light_cube_debug_white", (1.0, 1.0, 1.0, 1.0), strength=1.2)
    half = size * 0.5
    corners = [
        Vector((x, y, z))
        for x in (-half, half)
        for y in (-half, half)
        for z in (-half, half)
    ]
    index = {(round(p.x, 6), round(p.y, 6), round(p.z, 6)): i for i, p in enumerate(corners)}
    edges = []
    for a in corners:
        for axis in range(3):
            b = Vector(a)
            b[axis] *= -1
            ia = index[(round(a.x, 6), round(a.y, 6), round(a.z, 6))]
            ib = index[(round(b.x, 6), round(b.y, 6), round(b.z, 6))]
            if ia < ib:
                edges.append((a, b))
    objects = []
    for i, (a, b) in enumerate(edges):
        curve = bpy.data.curves.new(f"TL_light_cube_edge_{i:02d}", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = 0.01
        curve.materials.append(mat)
        spl = curve.splines.new("POLY")
        spl.points.add(1)
        spl.points[0].co = (a.x, a.y, a.z, 1.0)
        spl.points[1].co = (b.x, b.y, b.z, 1.0)
        obj = bpy.data.objects.new(curve.name, curve)
        bpy.context.collection.objects.link(obj)
        objects.append(obj)
    return objects


def render_objaverse_like(candidates: list[bpy.types.Object], out_path: Path) -> dict:
    duplicates = duplicate_candidate_group(candidates, Vector((0.0, 0.0, 0.0)), 1.35)
    cube_edges = create_debug_cube(size=2.0)
    keep = duplicates + cube_edges
    states = hide_all_except(keep)
    mat = make_mat("TL_objaverse_candidate_cyan", (0.0, 0.75, 1.0, 1.0))
    for obj in duplicates:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    light_data = bpy.data.lights.new("TL_objaverse_key_light", "AREA")
    light_data.energy = 450.0
    light_data.size = 4.0
    light = bpy.data.objects.new("TL_objaverse_key_light", light_data)
    light.location = (0.0, -3.0, 3.0)
    bpy.context.collection.objects.link(light)
    keep.append(light)
    cam_data = bpy.data.cameras.new("TL_objaverse_camera")
    cam = bpy.data.objects.new("TL_objaverse_camera", cam_data)
    cam.location = (0.0, -4.5, 0.65)
    cam.data.angle = math.radians(39.6)
    bpy.context.collection.objects.link(cam)
    look_at(cam, Vector((0.0, 0.0, 0.0)))
    old_camera = bpy.context.scene.camera
    bpy.context.scene.camera = cam
    try:
        render_png(out_path)
    finally:
        bpy.context.scene.camera = old_camera
        restore_visibility(states)
        for obj in keep + [cam]:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
    return {
        "canonical_object_max_extent": 1.35,
        "light_cube_min": [-1.0, -1.0, -1.0],
        "light_cube_max": [1.0, 1.0, 1.0],
        "camera_location": [0.0, -4.5, 0.65],
        "camera_fov_degrees": 39.6,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_render(args.width, args.height, args.samples)
    candidates, excluded, reason = choose_candidate_objects()
    if not bpy.context.scene.camera:
        cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
        if cameras:
            bpy.context.scene.camera = cameras[0]
    if not bpy.context.scene.camera:
        raise RuntimeError("Scene has no camera.")
    bbox_min, bbox_max = world_bbox(candidates)
    preview_path = out_dir / "preview_render_scene_000000.png"
    objaverse_path = out_dir / "objaverse_like_render_000000.png"
    colorize_candidate_preview(candidates, preview_path)
    objaverse_meta = render_objaverse_like(candidates, objaverse_path)
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    result = {
        "scene_id": args.scene_id,
        "source_blend": bpy.data.filepath,
        "preview_camera": bpy.context.scene.camera.name,
        "candidate_selection": {
            "mode": "single_group",
            "reason": reason,
            "candidate_objects": [obj.name for obj in candidates],
            "excluded_objects": [obj.name for obj in excluded],
            "bbox_min": vec(bbox_min),
            "bbox_max": vec(bbox_max),
            "bbox_center": vec((bbox_min + bbox_max) * 0.5),
            "bbox_extent": vec(bbox_max - bbox_min),
        },
        "source_metadata_subject_candidates": metadata.get("subject_candidates", []),
        "outputs": {
            "candidate_preview": preview_path.as_posix(),
            "objaverse_like_render": objaverse_path.as_posix(),
        },
        "objaverse_like": objaverse_meta,
    }
    (out_dir / "candidate_objects.json").write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
