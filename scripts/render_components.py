from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable

try:
    import bpy
    from mathutils import Matrix, Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("scripts/render_components.py must be run by Blender Python.") from exc


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render TokenLight-style linear EXR components.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--only", choices=["all", "spatial", "diffuse", "fixtures"], default="all")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def load_path_lines(path: Path, root: Path) -> list[str]:
    paths = []
    for line in load_lines(path):
        paths.append(str(resolve_path(root, line)))
    return paths


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def normalize_fixture_rows(rows: list[dict], root: Path) -> list[dict]:
    for row in rows:
        if row.get("blend_path"):
            row["blend_path"] = str(resolve_path(root, row["blend_path"]))
        if row.get("hdri_path"):
            row["hdri_path"] = str(resolve_path(root, row["hdri_path"]))
    return rows


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def setup_render_settings(config: dict) -> None:
    scene = bpy.context.scene
    render_cfg = config["render"]
    scene.render.engine = render_cfg.get("engine", "CYCLES")

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(render_cfg.get("samples", 128))
        scene.cycles.use_denoising = bool(render_cfg.get("denoise", True))
        if hasattr(scene.cycles, "tile_size"):
            scene.cycles.tile_size = int(render_cfg.get("tile_size", 256))
        if hasattr(scene.cycles, "max_bounces"):
            scene.cycles.max_bounces = int(render_cfg.get("max_bounces", 8))
        if render_cfg.get("device", "GPU").upper() == "GPU":
            scene.cycles.device = "GPU"
            prefs = bpy.context.preferences.addons.get("cycles")
            if prefs:
                cprefs = prefs.preferences
                for compute_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        cprefs.compute_device_type = compute_type
                        cprefs.get_devices()
                        for device in cprefs.devices:
                            device.use = True
                        break
                    except Exception:
                        continue

    resolution = render_cfg.get("resolution", 960)
    if isinstance(resolution, (list, tuple)):
        resolution_x, resolution_y = int(resolution[0]), int(resolution[1])
    else:
        resolution_x = int(render_cfg.get("resolution_x", resolution))
        resolution_y = int(render_cfg.get("resolution_y", resolution))
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100

    scene.view_settings.view_transform = render_cfg.get("view_transform", "Standard")
    scene.view_settings.look = render_cfg.get("look", "None")
    scene.view_settings.exposure = float(render_cfg.get("exposure", 0.0))
    scene.view_settings.gamma = float(render_cfg.get("gamma", 1.0))

    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = str(render_cfg.get("exr_color_depth", "16"))


def make_principled_mat(
    name: str,
    color: tuple[float, float, float],
    roughness: float = 0.75,
    metallic: float = 0.0,
    alpha: float = 1.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        set_input(bsdf, "Base Color", (color[0], color[1], color[2], alpha))
        set_input(bsdf, "Roughness", roughness)
        set_input(bsdf, "Metallic", metallic)
        set_input(bsdf, "Alpha", alpha)
        if alpha < 1.0:
            mat.blend_method = "BLEND"
    return mat


def make_emission_mat(name: str, color: tuple[float, float, float], strength: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    emission.inputs["Strength"].default_value = strength
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def set_input(node, name: str, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


def random_material(rng: random.Random, family_names: list[str]) -> bpy.types.Material:
    family = rng.choice(family_names or ["matte"])
    hue = rng.random()
    sat = rng.uniform(0.25, 0.85)
    val = rng.uniform(0.35, 0.9)
    color = hsv_to_rgb(hue, sat, val)
    if family == "metal":
        return make_principled_mat("tl_rand_metal", color, roughness=rng.uniform(0.18, 0.55), metallic=1.0)
    if family == "glossy":
        return make_principled_mat("tl_rand_glossy", color, roughness=rng.uniform(0.08, 0.28), metallic=0.0)
    if family == "plastic":
        return make_principled_mat("tl_rand_plastic", color, roughness=rng.uniform(0.28, 0.62), metallic=0.0)
    return make_principled_mat("tl_rand_matte", color, roughness=rng.uniform(0.62, 0.95), metallic=0.0)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def import_asset_or_primitive(asset_path: str | None, primitive: str, rng: random.Random, config: dict) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    if asset_path:
        path = Path(asset_path)
        ext = path.suffix.lower()
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(path))
        elif ext == ".obj":
            if hasattr(bpy.ops.import_scene, "obj"):
                bpy.ops.import_scene.obj(filepath=str(path))
            else:
                bpy.ops.wm.obj_import(filepath=str(path))
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(path))
        elif ext == ".stl":
            if hasattr(bpy.ops.import_mesh, "stl"):
                bpy.ops.import_mesh.stl(filepath=str(path))
            else:
                bpy.ops.wm.stl_import(filepath=str(path))
        elif ext == ".ply":
            if hasattr(bpy.ops.import_mesh, "ply"):
                bpy.ops.import_mesh.ply(filepath=str(path))
            else:
                bpy.ops.wm.ply_import(filepath=str(path))
        else:
            raise ValueError(f"Unsupported object asset extension: {path}")
    else:
        create_primitive(primitive)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"Asset produced no mesh objects: {asset_path or primitive}")

    if config["object"].get("randomize_materials", True):
        families = config["object"].get("material_families", ["matte", "plastic", "glossy", "metal"])
        for obj in mesh_objects:
            if not obj.data.materials or rng.random() < 0.35:
                obj.data.materials.clear()
                obj.data.materials.append(random_material(rng, families))

    normalize_objects(mesh_objects, float(config["object"].get("target_size", 1.2)))
    tag_objects(mesh_objects, "TL_SUBJECT")
    return mesh_objects


def create_primitive(name: str) -> None:
    name = name.lower()
    if name == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    elif name == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=0.42, minor_radius=0.17, major_segments=96, minor_segments=24)
    elif name == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.45, depth=1.0)
    elif name == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=0.5, radius2=0.08, depth=1.1)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.5)
    bpy.context.object.name = f"primitive_{name}"


def tag_objects(objects: Iterable[bpy.types.Object], tag: str) -> None:
    for obj in objects:
        obj[tag] = True


def mesh_bbox(objects: Iterable[bpy.types.Object]) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            v = eval_obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, v.x)
            mins.y = min(mins.y, v.y)
            mins.z = min(mins.z, v.z)
            maxs.x = max(maxs.x, v.x)
            maxs.y = max(maxs.y, v.y)
            maxs.z = max(maxs.z, v.z)
            found = True
    if not found:
        raise RuntimeError("No mesh bounds found")
    return mins, maxs


def root_objects(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    object_set = set(objects)
    return [obj for obj in objects if obj.parent not in object_set]


def apply_world_transform(objects: list[bpy.types.Object], matrix: Matrix) -> None:
    for obj in root_objects(objects):
        obj.matrix_world = matrix @ obj.matrix_world
    bpy.context.view_layer.update()


def normalize_objects(objects: list[bpy.types.Object], target_size: float) -> None:
    bbox_min, bbox_max = mesh_bbox(objects)
    center = (bbox_min + bbox_max) * 0.5
    max_dim = max((bbox_max - bbox_min).x, (bbox_max - bbox_min).y, (bbox_max - bbox_min).z)
    if max_dim <= 1e-6:
        raise RuntimeError("Object bbox is degenerate")
    scale = target_size / max_dim
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    apply_world_transform(objects, transform)

    bbox_min, bbox_max = mesh_bbox(objects)
    center_xy = Vector(((bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5, 0.0))
    lift = Vector((-center_xy.x, -center_xy.y, -bbox_min.z))
    apply_world_transform(objects, Matrix.Translation(lift))


def create_receivers(config: dict, rng: random.Random) -> list[bpy.types.Object]:
    layout = config["layout"]
    receivers: list[bpy.types.Object] = []
    if layout.get("ground", True):
        size = float(layout.get("ground_size", 8.0))
        bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
        ground = bpy.context.object
        ground.name = "TL_Ground"
        ground.data.materials.append(receiver_material("tl_ground_mat", rng, config))
        receivers.append(ground)

    if rng.random() < float(layout.get("wall_probability", 0.7)):
        receivers.append(create_wall("TL_BackWall", y=float(layout.get("wall_distance", 1.8)), config=config, rng=rng))
        if rng.random() < float(layout.get("corner_probability", 0.25)):
            wall = create_wall("TL_SideWall", y=0.0, config=config, rng=rng)
            wall.location.x = -float(layout.get("wall_distance", 1.8))
            wall.location.y = 0.0
            wall.rotation_euler[2] = math.radians(90)
            receivers.append(wall)

    tag_objects(receivers, "TL_RECEIVER")
    return receivers


def receiver_material(name: str, rng: random.Random, config: dict) -> bpy.types.Material:
    if config["layout"].get("randomize_receiver_material", True):
        v = rng.uniform(0.34, 0.72)
        color = (v * rng.uniform(0.85, 1.15), v * rng.uniform(0.85, 1.15), v * rng.uniform(0.85, 1.15))
    else:
        color = (0.55, 0.55, 0.55)
    return make_principled_mat(name, color, roughness=rng.uniform(0.55, 0.92), metallic=0.0)


def create_wall(name: str, y: float, config: dict, rng: random.Random) -> bpy.types.Object:
    layout = config["layout"]
    size = float(layout.get("ground_size", 8.0))
    height = float(layout.get("wall_height", 3.0))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, y, height * 0.5))
    wall = bpy.context.object
    wall.name = name
    wall.scale = (size * 0.5, height * 0.5, 1.0)
    wall.rotation_euler[0] = math.radians(90)
    wall.data.materials.append(receiver_material(f"{name}_mat", rng, config))
    return wall


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_camera(config: dict, rng: random.Random, center: Vector) -> tuple[bpy.types.Object, dict]:
    cam_cfg = config["camera"]
    fov = math.radians(float(cam_cfg.get("fov_degrees", 39.6)))
    distance = rng.uniform(*cam_cfg.get("distance_range", [2.8, 3.6]))
    az = math.radians(rng.uniform(*cam_cfg.get("azimuth_degrees_range", [-35.0, 35.0])))
    el = math.radians(rng.uniform(*cam_cfg.get("elevation_degrees_range", [4.0, 24.0])))
    jitter = float(cam_cfg.get("look_at_jitter", 0.06))
    look_target = center + Vector((rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)))

    location = Vector(
        (
            center.x + distance * math.cos(el) * math.sin(az),
            center.y - distance * math.cos(el) * math.cos(az),
            center.z + distance * math.sin(el),
        )
    )
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam_data.angle = fov
    look_at(cam, look_target)
    bpy.context.scene.camera = cam

    meta = {
        "location": vec_to_list(location),
        "look_at": vec_to_list(look_target),
        "fov_degrees": float(cam_cfg.get("fov_degrees", 39.6)),
        "distance": distance,
        "azimuth_degrees": math.degrees(az),
        "elevation_degrees": math.degrees(el),
    }
    return cam, meta


def camera_basis(camera: bpy.types.Object) -> tuple[Vector, Vector, Vector]:
    rot = camera.matrix_world.to_quaternion()
    right = (rot @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (rot @ Vector((0.0, 1.0, 0.0))).normalized()
    forward = (rot @ Vector((0.0, 0.0, -1.0))).normalized()
    return right, up, forward


def canonical_to_world(p: list[float], camera: bpy.types.Object, config: dict, center: Vector) -> Vector:
    canonical = config["canonical"]
    radius = float(canonical.get("light_volume_radius", 1.5))
    z_scale = float(canonical.get("z_scale", 1.2))
    right, _cam_up, forward = camera_basis(camera)
    world_up = Vector((0.0, 0.0, 1.0))
    return center + float(p[0]) * radius * right + float(p[1]) * radius * forward + float(p[2]) * z_scale * world_up


def set_hdri_world(hdri_path: str | None, strength: float, rotation_z: float, fallback_color: list[float]) -> dict:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    out = nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Strength"].default_value = strength
    if hdri_path and Path(hdri_path).exists():
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
        mapping.inputs["Rotation"].default_value[2] = rotation_z
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], bg.inputs["Color"])
        source = {"type": "hdri", "path": hdri_path, "strength": strength, "rotation_z": rotation_z}
    else:
        color = fallback_color
        bg.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
        source = {"type": "constant", "color": color, "strength": strength, "rotation_z": rotation_z}
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    return source


def set_black_world() -> None:
    set_constant_world((0.0, 0.0, 0.0), 0.0)


def set_constant_world(color: tuple[float, float, float], strength: float) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    out = nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    bg.inputs["Strength"].default_value = strength
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def remove_all_lights() -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def create_point_light(name: str, location: Vector, energy: float, radius: float, color: list[float]) -> bpy.types.Object:
    data = bpy.data.lights.new(name=name, type="POINT")
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    data.energy = energy
    data.color = tuple(color)
    data.shadow_soft_size = radius
    return obj


def create_area_light(name: str, location: Vector, target: Vector, energy: float, size: float) -> bpy.types.Object:
    data = bpy.data.lights.new(name=name, type="AREA")
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    data.energy = energy
    data.shape = "SQUARE"
    data.size = max(size, 0.001)
    look_at(obj, target)
    return obj


def render_exr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(write_still=True)


def render_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"


def sample_spatial_positions(config: dict, rng: random.Random) -> list[list[float]]:
    spatial = config["spatial"]
    count = int(spatial.get("positions_per_scene", 64))
    pr = config["canonical"]["position_range"]
    positions: list[list[float]] = []
    if spatial.get("fixed_debug_positions_first", True):
        positions.extend(
            [
                [-1.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, -1.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.25],
            ]
        )
    while len(positions) < count:
        positions.append(
            [
                rng.uniform(*pr["x"]),
                rng.uniform(*pr["y"]),
                rng.uniform(*pr["z"]),
            ]
        )
    return positions[:count]


def render_object_mask(scene_dir: Path, subject_objects: list[bpy.types.Object]) -> str:
    original_materials = {obj.name: list(obj.data.materials) for obj in bpy.data.objects if obj.type == "MESH"}
    original_lights = [(obj, obj.hide_render) for obj in bpy.data.objects if obj.type == "LIGHT"]
    white = make_emission_mat("TL_mask_white", (1.0, 1.0, 1.0), 1.0)
    black = make_emission_mat("TL_mask_black", (0.0, 0.0, 0.0), 1.0)
    subject_set = set(subject_objects)
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(white if obj in subject_set else black)
    for light, _state in original_lights:
        light.hide_render = True
    set_black_world()
    rel_path = "masks/object_mask.png"
    render_png(scene_dir / rel_path)
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name in original_materials:
            obj.data.materials.clear()
            for mat in original_materials[obj.name]:
                obj.data.materials.append(mat)
    for light, state in original_lights:
        light.hide_render = state
    return rel_path


def render_spatial_components(scene_dir: Path, config: dict, rng: random.Random, camera: bpy.types.Object, center: Vector) -> dict:
    ambient = config["ambient"]
    hdris = config["_runtime"]["hdris"]
    hdri_path = rng.choice(hdris) if hdris else None
    hdri_strength = rng.uniform(*ambient.get("hdri_strength_range", [0.8, 1.2]))
    hdri_rotation = rng.random() * 2.0 * math.pi if ambient.get("hdri_rotation_random", True) else 0.0
    source = set_hdri_world(hdri_path, hdri_strength, hdri_rotation, ambient.get("fallback_color", [0.78, 0.78, 0.78]))
    remove_all_lights()
    ambient_render = "spatial/ambient.exr"
    render_exr(scene_dir / ambient_render)

    set_black_world()
    positions = sample_spatial_positions(config, rng)
    lights_meta = []
    spatial = config["spatial"]
    radius = float(spatial.get("fixed_radius", 0.06))
    for light_index, p_can in enumerate(positions):
        p_world = canonical_to_world(p_can, camera, config, center)
        light = create_point_light(
            f"TL_Point_{light_index:03d}",
            p_world,
            float(spatial.get("base_energy", 500.0)),
            radius,
            spatial.get("color", [1.0, 1.0, 1.0]),
        )
        rel_path = f"spatial/point_lights/light_{light_index:03d}.exr"
        render_exr(scene_dir / rel_path)
        bpy.data.objects.remove(light, do_unlink=True)
        lights_meta.append(
            {
                "id": light_index,
                "canonical_position": p_can,
                "world_position": vec_to_list(p_world),
                "energy": float(spatial.get("base_energy", 500.0)),
                "canonical_radius": radius,
                "world_radius": radius,
                "render": rel_path,
            }
        )
    return {
        "ambient_render": ambient_render,
        "ambient_source": source,
        "positions_per_scene": int(spatial.get("positions_per_scene", 64)),
        "fixed_radius": radius,
        "point_lights": lights_meta,
    }


def render_diffuse_components(scene_dir: Path, config: dict, rng: random.Random, camera: bpy.types.Object, center: Vector) -> dict:
    diffuse = config["diffuse"]
    color = tuple(diffuse.get("constant_ambient_color", [0.72, 0.72, 0.72]))
    strength = float(diffuse.get("constant_ambient_strength", 0.35))
    set_constant_world(color, strength)
    remove_all_lights()
    ambient_render = "diffuse/ambient_constant.exr"
    render_exr(scene_dir / ambient_render)

    set_black_world()
    p_can = diffuse.get("area_light_canonical_position", [-0.45, -0.65, 1.15])
    p_world = canonical_to_world(p_can, camera, config, center)
    spread_count = int(diffuse.get("spread_count", 6))
    spread_min, spread_max = diffuse.get("spread_degrees_range", [6.0, 70.0])
    norm_min, norm_max = diffuse.get("normalize_spread_to", [-1.0, 1.0])
    spreads = []
    for i in range(spread_count):
        t = 0.0 if spread_count == 1 else i / (spread_count - 1)
        spread_deg = spread_min + t * (spread_max - spread_min)
        distance = max((p_world - center).length, 0.1)
        area_size = 2.0 * distance * math.tan(math.radians(spread_deg) * 0.5)
        light = create_area_light(
            f"TL_Diffuse_Area_{i:03d}",
            p_world,
            center,
            float(diffuse.get("area_light_energy", 650.0)),
            area_size,
        )
        rel_path = f"diffuse/spread_{i:03d}.exr"
        render_exr(scene_dir / rel_path)
        bpy.data.objects.remove(light, do_unlink=True)
        normalized = norm_min + t * (norm_max - norm_min)
        spreads.append(
            {
                "id": i,
                "spread_degrees": spread_deg,
                "normalized_spread": normalized,
                "area_size": area_size,
                "canonical_position": p_can,
                "world_position": vec_to_list(p_world),
                "render": rel_path,
            }
        )
    return {
        "ambient_render": ambient_render,
        "ambient_source": {
            "type": "constant",
            "color": list(color),
            "strength": strength,
        },
        "spreads": spreads,
    }


def render_object_scene(scene_index: int, config: dict, root: Path, only: str) -> dict:
    rng = random.Random(int(config["seed"]) + scene_index)
    clear_scene()
    setup_render_settings(config)
    output_root = resolve_path(root, config["output_root"]) or (root / "outputs/tokenlight_synthetic")
    scene_id = f"scene_{scene_index:06d}"
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    objects = config["_runtime"]["objects"]
    primitives = config["object"].get("primitive_fallbacks", ["sphere"])
    asset = rng.choice(objects) if objects else None
    primitive = rng.choice(primitives)
    subject_objects = import_asset_or_primitive(asset, primitive, rng, config)
    create_receivers(config, rng)

    bbox_min, bbox_max = mesh_bbox(subject_objects)
    center = (bbox_min + bbox_max) * 0.5
    camera, camera_meta = create_camera(config, rng, center)
    mask_path = render_object_mask(scene_dir, subject_objects)

    meta = {
        "schema": "tokenlight_synthetic_components_v1",
        "scene_id": scene_id,
        "scene_type": "object_centric",
        "object": {
            "path": asset,
            "primitive": None if asset else primitive,
            "target_size": float(config["object"].get("target_size", 1.2)),
            "bbox_min": vec_to_list(bbox_min),
            "bbox_max": vec_to_list(bbox_max),
            "center": vec_to_list(center),
        },
        "camera": camera_meta,
        "render": {
            "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
            "samples": int(config["render"].get("samples", 128)),
            "engine": bpy.context.scene.render.engine,
            "linear_rgb": True,
            "tone_mapping_applied": False,
        },
        "canonical": config["canonical"],
        "masks": {"object": mask_path},
    }

    if only in ("all", "spatial") and config["spatial"].get("enabled", True):
        meta["spatial"] = render_spatial_components(scene_dir, config, rng, camera, center)
    if only in ("all", "diffuse") and config["diffuse"].get("enabled", True):
        meta["diffuse"] = render_diffuse_components(scene_dir, config, rng, camera, center)

    write_json(scene_dir / "meta.json", meta)
    return meta


def fixture_objects_for(prefixes: list[str]) -> list[bpy.types.Object]:
    matches = []
    for obj in bpy.data.objects:
        if any(obj.name.startswith(prefix) for prefix in prefixes):
            matches.append(obj)
    return matches


def has_emission_material(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "EMISSION":
                    return True
    return False


def apply_fixture_states(
    fixtures: list[dict],
    active_fixture_id: str | None,
    original_materials: dict[str, list[bpy.types.Material]],
    off_material: bpy.types.Material,
) -> None:
    for fixture in fixtures:
        enabled = active_fixture_id is not None and fixture["id"] == active_fixture_id
        prefixes = fixture.get("prefixes", [])
        light_prefixes = fixture.get("light_prefixes", [])
        for obj in bpy.data.objects:
            if any(obj.name.startswith(prefix) for prefix in light_prefixes):
                obj.hide_render = not enabled
            if obj.type == "MESH" and any(obj.name.startswith(prefix) for prefix in prefixes):
                obj.hide_render = False
                obj.data.materials.clear()
                if enabled:
                    for mat in original_materials.get(obj.name, []):
                        obj.data.materials.append(mat)
                else:
                    obj.data.materials.append(off_material)


def render_fixture_mask(scene_dir: Path, fixture: dict) -> str:
    original_materials = {obj.name: list(obj.data.materials) for obj in bpy.data.objects if obj.type == "MESH"}
    original_lights = [(obj, obj.hide_render) for obj in bpy.data.objects if obj.type == "LIGHT"]
    white = make_emission_mat(f"TL_mask_{fixture['id']}_white", (1.0, 1.0, 1.0), 1.0)
    black = make_emission_mat(f"TL_mask_{fixture['id']}_black", (0.0, 0.0, 0.0), 1.0)
    fixture_set = set(fixture_objects_for(fixture.get("prefixes", [])))
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(white if obj in fixture_set else black)
    for light, _state in original_lights:
        light.hide_render = True
    set_black_world()
    rel_path = f"fixtures/fixture_{fixture['id']}/mask.png"
    render_png(scene_dir / rel_path)
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name in original_materials:
            obj.data.materials.clear()
            for mat in original_materials[obj.name]:
                obj.data.materials.append(mat)
    for light, state in original_lights:
        light.hide_render = state
    return rel_path


def render_fixture_scene(row: dict, scene_index: int, config: dict, root: Path) -> dict:
    blend_path = resolve_path(root, row["blend_path"])
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    setup_render_settings(config)
    output_root = resolve_path(root, config["output_root"]) or (root / "outputs/tokenlight_synthetic")
    scene_id = row.get("scene_id") or f"fixture_{scene_index:06d}"
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if row.get("camera") and row["camera"] in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects[row["camera"]]

    ambient = config["ambient"]
    hdri_path = row.get("hdri_path")
    set_hdri_world(
        hdri_path,
        float(row.get("hdri_strength", ambient.get("fallback_strength", 0.8))),
        float(row.get("hdri_rotation_z", 0.0)),
        ambient.get("fallback_color", [0.78, 0.78, 0.78]),
    )

    fixtures = row.get("fixtures", [])
    original_materials = {obj.name: list(obj.data.materials) for obj in bpy.data.objects if obj.type == "MESH"}
    off_material = make_principled_mat("TL_fixture_off_material", (0.02, 0.02, 0.02), roughness=0.85, metallic=0.0)
    apply_fixture_states(fixtures, None, original_materials, off_material)

    environment_render = "fixtures/environment.exr"
    render_exr(scene_dir / environment_render)

    fixtures_meta = []
    for fixture in fixtures:
        apply_fixture_states(fixtures, fixture["id"], original_materials, off_material)
        set_black_world()
        rel = f"fixtures/fixture_{fixture['id']}/contribution.exr"
        render_exr(scene_dir / rel)
        mask = render_fixture_mask(scene_dir, fixture)
        fixtures_meta.append(
            {
                "id": fixture["id"],
                "prefixes": fixture.get("prefixes", []),
                "light_prefixes": fixture.get("light_prefixes", []),
                "contribution_render": rel,
                "mask_render": mask,
            }
        )

    meta = {
        "schema": "tokenlight_synthetic_components_v1",
        "scene_id": scene_id,
        "scene_type": "visible_fixture",
        "source_blend": str(blend_path),
        "camera": {
            "name": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
        },
        "render": {
            "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
            "samples": int(config["render"].get("samples", 128)),
            "engine": bpy.context.scene.render.engine,
            "linear_rgb": True,
            "tone_mapping_applied": False,
        },
        "fixtures": {
            "environment_render": environment_render,
            "max_non_selected_fixtures_in_ambient": int(config["fixtures"].get("max_non_selected_fixtures_in_ambient", 5)),
            "fixtures": fixtures_meta,
        },
    }
    write_json(scene_dir / "meta.json", meta)
    return meta


def vec_to_list(v: Vector) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_json(config_path.resolve())
    if args.resolution is not None:
        config["render"]["resolution"] = args.resolution
        config["render"].pop("resolution_x", None)
        config["render"].pop("resolution_y", None)
    if args.width is not None:
        config["render"]["resolution_x"] = args.width
    if args.height is not None:
        config["render"]["resolution_y"] = args.height
    if args.samples is not None:
        config["render"]["samples"] = args.samples

    object_manifest = resolve_path(root, config.get("object_manifest"))
    hdri_manifest = resolve_path(root, config.get("hdri_manifest"))
    fixture_manifest = resolve_path(root, config.get("fixture_scene_manifest"))
    config["_runtime"] = {
        "objects": load_path_lines(object_manifest, root) if object_manifest else [],
        "hdris": load_path_lines(hdri_manifest, root) if hdri_manifest else [],
        "fixture_scenes": normalize_fixture_rows(load_jsonl(fixture_manifest), root) if fixture_manifest else [],
    }

    scene_count = int(config.get("scene_count", 1))
    if args.max_scenes is not None:
        scene_count = min(scene_count, args.max_scenes)

    output_root = resolve_path(root, config["output_root"]) or (root / "outputs/tokenlight_synthetic")
    metas = []
    if args.only in ("all", "spatial", "diffuse"):
        for i in range(args.start_index, args.start_index + scene_count):
            print(f"[TokenLight] Rendering object-centric scene {i}", flush=True)
            metas.append(render_object_scene(i, config, root, args.only))

    if args.only in ("all", "fixtures") and config["fixtures"].get("enabled", True):
        fixture_rows = config["_runtime"]["fixture_scenes"]
        if fixture_rows and config["fixtures"].get("render_if_manifest_exists", True):
            for j, row in enumerate(fixture_rows):
                print(f"[TokenLight] Rendering fixture scene {row.get('scene_id', j)}", flush=True)
                metas.append(render_fixture_scene(row, j, config, root))
        else:
            print("[TokenLight] No fixture scene manifest found; skipping visible-fixture synthetic renders.", flush=True)

    manifest = {
        "schema": "tokenlight_synthetic_dataset_manifest_v1",
        "paper_defaults": {
            "spatial_positions_per_scene": config["spatial"].get("positions_per_scene", 64),
            "diffuse_spread_count": config["diffuse"].get("spread_count", 6),
            "fov_degrees": config["camera"].get("fov_degrees", 39.6),
            "linear_rgb_components": True,
        },
        "scene_count_written": len(metas),
        "scenes": [{"scene_id": m["scene_id"], "scene_type": m["scene_type"], "meta": f"scenes/{m['scene_id']}/meta.json"} for m in metas],
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    print(f"[TokenLight] Wrote manifest: {output_root / 'dataset_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
