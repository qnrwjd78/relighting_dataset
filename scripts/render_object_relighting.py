from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from array import array
from pathlib import Path
from typing import Iterable

DATASET_UTILS = Path(__file__).resolve().parents[1] / "dataset"
if str(DATASET_UTILS) not in sys.path:
    sys.path.insert(0, str(DATASET_UTILS))
from utils.util_progress import progress_bar, progress_write

try:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("scripts/render_object_relighting.py must be run by Blender Python.") from exc


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render object/portrait relighting components.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug", action="store_true", help="Render preview outputs only and skip component EXR/mask renders.")
    parser.add_argument("--light-preview", action="store_true", help="Render a PNG showing the sampled spatial light positions.")
    parser.add_argument("--debug-light-preview", action="store_true", help="Deprecated alias for --light-preview.")
    parser.add_argument("--pbr", action="store_true", help="Render extra PBR maps: depth, normal, albedo, roughness.")
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default=None)
    parser.add_argument("--output-format", choices=["exr", "png", "both"], dest="component_format")
    parser.add_argument("--hdri-mode", choices=["on", "off", "random"], default=None)
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


def load_receiver_texture_manifest(path: Path, root: Path) -> list[dict]:
    if not path.exists():
        return []
    data = load_json(path)
    rows = data.get("textures", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []

    textures = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        maps = row.get("maps", {})
        resolved_maps = {}
        for map_name, info in maps.items():
            if isinstance(info, dict):
                path_value = info.get("path")
                if path_value:
                    map_info = dict(info)
                    map_info["path"] = str(resolve_path(root, path_value))
                    resolved_maps[map_name] = map_info
            elif isinstance(info, str):
                resolved_maps[map_name] = {"path": str(resolve_path(root, info))}
        if resolved_maps.get("albedo"):
            tex = dict(row)
            tex["maps"] = resolved_maps
            textures.append(tex)
    return textures


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
    scene["tl_exr_color_depth"] = str(render_cfg.get("exr_color_depth", "16"))


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


def clamp_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, float(channel))) for channel in color)


def mix_color(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, float(t)))
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def jitter_color(
    rng: random.Random,
    color: tuple[float, float, float],
    strength: float,
) -> tuple[float, float, float]:
    return clamp_color(tuple(channel * rng.uniform(1.0 - strength, 1.0 + strength) for channel in color))


def muted_receiver_color(rng: random.Random, value_range: tuple[float, float] = (0.38, 0.78)) -> tuple[float, float, float]:
    hue = rng.random()
    saturation = rng.uniform(0.02, 0.24)
    value = rng.uniform(*value_range)
    return hsv_to_rgb(hue, saturation, value)


def get_principled_bsdf(mat: bpy.types.Material):
    if not mat.use_nodes:
        mat.use_nodes = True
    return mat.node_tree.nodes.get("Principled BSDF")


def make_receiver_principled_mat(
    name: str,
    color: tuple[float, float, float],
    roughness: float,
    specular: float,
) -> bpy.types.Material:
    mat = make_principled_mat(name, clamp_color(color), roughness=roughness, metallic=0.0)
    bsdf = get_principled_bsdf(mat)
    if bsdf:
        set_input(bsdf, "Specular IOR Level", specular)
        set_input(bsdf, "Specular", specular)
    return mat


def add_color_ramp_texture(
    mat: bpy.types.Material,
    bsdf,
    factor_socket,
    color_a: tuple[float, float, float],
    color_b: tuple[float, float, float],
) -> None:
    ramp = mat.node_tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (*clamp_color(color_a), 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (*clamp_color(color_b), 1.0)
    mat.node_tree.links.new(factor_socket, ramp.inputs["Fac"])
    mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


def add_noise_base_color(
    mat: bpy.types.Material,
    rng: random.Random,
    bsdf,
    color: tuple[float, float, float],
    texture_strength: float,
    scale_range: tuple[float, float],
    detail: float = 8.0,
) -> None:
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    set_input(noise, "Scale", rng.uniform(*scale_range))
    set_input(noise, "Detail", detail)
    set_input(noise, "Roughness", rng.uniform(0.48, 0.68))
    color_a = mix_color(color, (0.0, 0.0, 0.0), texture_strength)
    color_b = mix_color(color, (1.0, 1.0, 1.0), texture_strength)
    add_color_ramp_texture(mat, bsdf, noise.outputs["Fac"], color_a, color_b)


def add_checker_base_color(
    mat: bpy.types.Material,
    rng: random.Random,
    bsdf,
    color: tuple[float, float, float],
    texture_strength: float,
    scale_range: tuple[float, float],
) -> None:
    checker = mat.node_tree.nodes.new("ShaderNodeTexChecker")
    set_input(checker, "Scale", rng.uniform(*scale_range))
    color_a = mix_color(color, (0.0, 0.0, 0.0), texture_strength)
    color_b = mix_color(color, (1.0, 1.0, 1.0), texture_strength)
    set_input(checker, "Color1", (*clamp_color(color_a), 1.0))
    set_input(checker, "Color2", (*clamp_color(color_b), 1.0))
    mat.node_tree.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])


def add_wave_base_color(
    mat: bpy.types.Material,
    rng: random.Random,
    bsdf,
    color_a: tuple[float, float, float],
    color_b: tuple[float, float, float],
    scale_range: tuple[float, float],
) -> None:
    wave = mat.node_tree.nodes.new("ShaderNodeTexWave")
    set_input(wave, "Scale", rng.uniform(*scale_range))
    set_input(wave, "Distortion", rng.uniform(4.0, 12.0))
    factor = wave.outputs["Fac"] if "Fac" in wave.outputs else wave.outputs["Color"]
    add_color_ramp_texture(mat, bsdf, factor, color_a, color_b)


def add_brick_base_color(
    mat: bpy.types.Material,
    rng: random.Random,
    bsdf,
    color: tuple[float, float, float],
    texture_strength: float,
    scale_range: tuple[float, float],
) -> None:
    brick = mat.node_tree.nodes.new("ShaderNodeTexBrick")
    set_input(brick, "Scale", rng.uniform(*scale_range))
    set_input(brick, "Mortar Size", rng.uniform(0.012, 0.045))
    set_input(brick, "Color1", (*clamp_color(mix_color(color, (0.0, 0.0, 0.0), texture_strength)), 1.0))
    set_input(brick, "Color2", (*clamp_color(mix_color(color, (1.0, 1.0, 1.0), texture_strength)), 1.0))
    set_input(brick, "Mortar", (*clamp_color(mix_color(color, (0.1, 0.1, 0.1), texture_strength * 1.3)), 1.0))
    mat.node_tree.links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])


def add_noise_bump(
    mat: bpy.types.Material,
    rng: random.Random,
    bsdf,
    bump_strength: float,
    scale_range: tuple[float, float],
) -> None:
    if bump_strength <= 0.0:
        return
    noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
    set_input(noise, "Scale", rng.uniform(*scale_range))
    set_input(noise, "Detail", rng.uniform(3.0, 10.0))
    bump = mat.node_tree.nodes.new("ShaderNodeBump")
    set_input(bump, "Strength", bump_strength)
    set_input(bump, "Distance", rng.uniform(0.02, 0.08))
    mat.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    mat.node_tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def set_image_colorspace(image: bpy.types.Image, colorspace: str) -> None:
    try:
        image.colorspace_settings.name = colorspace
    except Exception:
        pass


def texture_categories(entry: dict) -> set[str]:
    categories = set()
    for key in ("download_category", "category"):
        value = entry.get(key)
        if value:
            categories.add(str(value).lower())
    for value in entry.get("asset_categories", entry.get("categories", [])) or []:
        categories.add(str(value).lower())
    return categories


def choose_receiver_texture(config: dict, rng: random.Random, role: str) -> dict | None:
    layout = config["layout"]
    textures = config.get("_runtime", {}).get("receiver_textures", [])
    if not textures:
        return None

    probability = float(layout.get(f"{role}_texture_probability", layout.get("receiver_texture_probability", 0.75)))
    if rng.random() >= probability:
        return None

    allowed = {str(c).lower() for c in layout.get(f"{role}_texture_categories", [])}
    candidates = textures
    if allowed:
        candidates = [entry for entry in textures if texture_categories(entry) & allowed]
    if not candidates:
        candidates = textures
    return rng.choice(candidates) if candidates else None


def make_image_texture_node(
    mat: bpy.types.Material,
    map_info: dict,
    colorspace: str,
    vector_socket,
):
    image_path = map_info.get("path")
    if not image_path or not Path(image_path).exists():
        return None
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path), check_existing=True)
    set_image_colorspace(tex.image, colorspace)
    mat.node_tree.links.new(vector_socket, tex.inputs["Vector"])
    return tex


def set_receiver_material_meta(
    mat: bpy.types.Material,
    source: str,
    family: str,
    texture_entry: dict | None = None,
    texture_scale: float | None = None,
) -> None:
    mat["tl_receiver_source"] = source
    mat["tl_receiver_family"] = family
    if texture_entry:
        maps = {
            name: info.get("path")
            for name, info in texture_entry.get("maps", {}).items()
            if isinstance(info, dict) and info.get("path")
        }
        mat["tl_receiver_texture"] = json.dumps(
            {
                "id": texture_entry.get("id"),
                "name": texture_entry.get("name"),
                "download_category": texture_entry.get("download_category"),
                "asset_categories": texture_entry.get("asset_categories", []),
                "scale": texture_scale,
                "maps": maps,
            },
            ensure_ascii=False,
        )


def make_receiver_texture_material(
    name: str,
    texture_entry: dict,
    rng: random.Random,
    config: dict,
    role: str,
) -> bpy.types.Material | None:
    maps = texture_entry.get("maps", {})
    albedo = maps.get("albedo")
    albedo_path = albedo.get("path") if isinstance(albedo, dict) else None
    if not albedo_path or not Path(albedo_path).exists():
        return None

    layout = config["layout"]
    scale = sample_range(layout.get("receiver_texture_scale_range"), 3.0, rng)
    normal_strength = sample_range(layout.get("receiver_texture_normal_strength_range"), 0.28, rng)
    roughness_fallback = rng.uniform(0.45, 0.9) if role == "floor" else rng.uniform(0.68, 0.96)
    specular = rng.uniform(0.12, 0.42) if role == "floor" else rng.uniform(0.04, 0.22)
    mat = make_receiver_principled_mat(name, (0.65, 0.65, 0.65), roughness=roughness_fallback, specular=specular)
    bsdf = get_principled_bsdf(mat)
    if not bsdf:
        return None

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    set_input(mapping, "Scale", (scale, scale, 1.0))
    set_input(mapping, "Rotation", (0.0, 0.0, rng.random() * math.tau))
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

    albedo_tex = make_image_texture_node(mat, albedo, "sRGB", mapping.outputs["Vector"])
    if not albedo_tex:
        return None
    links.new(albedo_tex.outputs["Color"], bsdf.inputs["Base Color"])

    roughness = maps.get("roughness")
    if isinstance(roughness, dict):
        roughness_tex = make_image_texture_node(mat, roughness, "Non-Color", mapping.outputs["Vector"])
        if roughness_tex:
            roughness_bw = nodes.new("ShaderNodeRGBToBW")
            links.new(roughness_tex.outputs["Color"], roughness_bw.inputs["Color"])
            links.new(roughness_bw.outputs["Val"], bsdf.inputs["Roughness"])

    normal = maps.get("normal")
    if isinstance(normal, dict) and normal_strength > 0.0:
        normal_tex = make_image_texture_node(mat, normal, "Non-Color", mapping.outputs["Vector"])
        if normal_tex:
            normal_map = nodes.new("ShaderNodeNormalMap")
            set_input(normal_map, "Strength", normal_strength)
            links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    set_receiver_material_meta(mat, "polyhaven_texture", role, texture_entry, scale)
    return mat


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
        if ext == ".blend":
            with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
                data_to.objects = list(data_from.objects)
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        elif ext in (".glb", ".gltf"):
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


def sample_range(config_value, fallback: float, rng: random.Random) -> float:
    if isinstance(config_value, list) and len(config_value) == 2:
        return rng.uniform(float(config_value[0]), float(config_value[1]))
    return float(fallback)


def horizontal_camera_axes(camera: bpy.types.Object) -> tuple[Vector, Vector]:
    right, _up, forward = camera_basis(camera)
    forward.z = 0.0
    if forward.length <= 1e-6:
        forward = Vector((0.0, 1.0, 0.0))
    forward.normalize()
    right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
    return right, forward


def create_oriented_quad(
    name: str,
    center: Vector,
    axis_u: Vector,
    axis_v: Vector,
    width: float,
    height: float,
) -> bpy.types.Object:
    u = axis_u.normalized() * (width * 0.5)
    v = axis_v.normalized() * (height * 0.5)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata([center - u - v, center + u - v, center + u + v, center - u + v], [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="TL_UV")
    for poly in mesh.polygons:
        for loop_index, uv in zip(poly.loop_indices, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
            uv_layer.data[loop_index].uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def mesh_visible_to_camera(obj: bpy.types.Object, camera: bpy.types.Object, samples: int = 7, margin: float = 0.03) -> bool:
    if obj.type != "MESH" or not obj.data.polygons:
        return False
    scene = bpy.context.scene
    world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    for poly in obj.data.polygons:
        verts = [world_verts[i] for i in poly.vertices]
        if len(verts) < 4:
            points = verts
        else:
            a, b, c, d = verts[:4]
            points = []
            for iy in range(samples):
                ty = iy / max(samples - 1, 1)
                left = a.lerp(d, ty)
                right = b.lerp(c, ty)
                for ix in range(samples):
                    tx = ix / max(samples - 1, 1)
                    points.append(left.lerp(right, tx))
        for point in points:
            co = world_to_camera_view(scene, camera, point)
            if co.z > 0.0 and -margin <= co.x <= 1.0 + margin and -margin <= co.y <= 1.0 + margin:
                return True
    return False


def remove_camera_invisible_walls(receivers: list[bpy.types.Object], camera: bpy.types.Object) -> list[bpy.types.Object]:
    bpy.context.view_layer.update()
    kept = []
    for obj in receivers:
        if obj.name in {"TL_Ground", "TL_BackWall"} or mesh_visible_to_camera(obj, camera):
            kept.append(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
    return kept


def receiver_bounds_to_meta(bounds: dict) -> dict:
    return {
        "origin": vec_to_list(bounds["origin"]),
        "right": vec_to_list(bounds["right"]),
        "forward": vec_to_list(bounds["forward"]),
        "front_y": float(bounds["front_y"]),
        "back_y": float(bounds["back_y"]),
        "half_width": float(bounds["half_width"]),
        "floor_z": float(bounds["floor_z"]),
        "wall_height": float(bounds["wall_height"]),
        "kept_walls": list(bounds.get("kept_walls", [])),
    }


def receiver_material_to_meta(obj: bpy.types.Object, role: str) -> dict:
    mat = obj.data.materials[0] if obj.type == "MESH" and obj.data.materials else None
    meta = {
        "object": obj.name,
        "role": role,
        "material": mat.name if mat else None,
        "source": mat.get("tl_receiver_source") if mat else None,
        "family": mat.get("tl_receiver_family") if mat else None,
    }
    texture_json = mat.get("tl_receiver_texture") if mat else None
    if texture_json:
        try:
            meta["texture"] = json.loads(texture_json)
        except json.JSONDecodeError:
            meta["texture"] = texture_json
    return meta


def point_inside_receiver_bounds(point: Vector, bounds: dict | None, radius: float, margin: float = 0.02) -> tuple[bool, str | None]:
    if not bounds:
        return True, None
    safety = radius + margin
    rel = point - bounds["origin"]
    x = rel.dot(bounds["right"])
    y = rel.dot(bounds["forward"])
    z = point.z
    if z < bounds["floor_z"] + safety:
        return False, "below_floor"
    if z > bounds["floor_z"] + bounds["wall_height"] - safety:
        return False, "above_wall_height"
    if y > bounds["back_y"] - safety:
        return False, "behind_back_wall"
    if y < bounds["front_y"] + safety:
        return False, "outside_ground_front"
    if abs(x) > bounds["half_width"] - safety:
        return False, "outside_side_bounds"
    return True, None


def create_receivers(config: dict, rng: random.Random, camera: bpy.types.Object, center: Vector) -> list[bpy.types.Object]:
    layout = config["layout"]
    receivers: list[bpy.types.Object] = []
    receiver_roles: dict[str, str] = {}
    right, forward = horizontal_camera_axes(camera)
    origin = Vector((center.x, center.y, 0.0))
    room_width = max(sample_range(layout.get("ground_size_range"), float(layout.get("ground_size", 10.0)), rng), 24.0)
    back_y = max(sample_range(layout.get("wall_distance_range"), float(layout.get("wall_distance", 2.4)), rng), 2.4)
    front_y = -max(room_width * 0.75, 14.0)
    depth = back_y - front_y
    wall_height = max(sample_range(layout.get("wall_height_range"), float(layout.get("wall_height", 3.0)), rng), 10.0)
    up = Vector((0.0, 0.0, 1.0))

    if layout.get("ground", True):
        ground = create_oriented_quad(
            "TL_Ground",
            center=origin + forward * ((front_y + back_y) * 0.5),
            axis_u=right,
            axis_v=forward,
            width=room_width,
            height=depth,
        )
        ground.data.materials.append(floor_material("tl_ground_mat", rng, config))
        receivers.append(ground)
        receiver_roles[ground.name] = "floor"

    if rng.random() < float(layout.get("wall_probability", 1.0)):
        back_wall = create_oriented_quad(
            "TL_BackWall",
            center=origin + forward * back_y + up * (wall_height * 0.5),
            axis_u=right,
            axis_v=up,
            width=room_width,
            height=wall_height,
        )
        back_wall.data.materials.append(wall_material("TL_BackWall_mat", rng, config))
        receivers.append(back_wall)
        receiver_roles[back_wall.name] = "wall"

        for side_name, side_sign in (("TL_LeftWall", -1.0), ("TL_RightWall", 1.0)):
            side_wall = create_oriented_quad(
                side_name,
                center=origin + right * (side_sign * room_width * 0.5) + forward * ((front_y + back_y) * 0.5) + up * (wall_height * 0.5),
                axis_u=forward,
                axis_v=up,
                width=depth,
                height=wall_height,
            )
            side_wall.data.materials.append(wall_material(f"{side_name}_mat", rng, config))
            receivers.append(side_wall)
            receiver_roles[side_wall.name] = "wall"

    tag_objects(receivers, "TL_RECEIVER")
    kept = remove_camera_invisible_walls(receivers, camera)
    config["_runtime"]["receiver_materials"] = [
        receiver_material_to_meta(obj, receiver_roles.get(obj.name, "receiver"))
        for obj in kept
    ]
    config["_runtime"]["receiver_bounds"] = {
        "origin": origin,
        "right": right,
        "forward": forward,
        "front_y": front_y,
        "back_y": back_y,
        "half_width": room_width * 0.5,
        "floor_z": 0.0,
        "wall_height": wall_height,
        "kept_walls": [obj.name for obj in kept if obj.name != "TL_Ground"],
    }
    return kept


def receiver_texture_ranges(config: dict, rng: random.Random) -> tuple[float, float]:
    layout = config["layout"]
    texture_strength = sample_range(layout.get("receiver_texture_strength_range"), 0.14, rng)
    bump_strength = sample_range(layout.get("receiver_bump_strength_range"), 0.018, rng)
    return max(0.0, texture_strength), max(0.0, bump_strength)


def floor_material(name: str, rng: random.Random, config: dict) -> bpy.types.Material:
    layout = config["layout"]
    if not layout.get("randomize_receiver_material", True):
        mat = make_receiver_principled_mat(name, (0.55, 0.55, 0.55), roughness=0.75, specular=0.25)
        set_receiver_material_meta(mat, "procedural", "matte_neutral")
        return mat

    texture_entry = choose_receiver_texture(config, rng, "floor")
    if texture_entry:
        texture_mat = make_receiver_texture_material(name, texture_entry, rng, config, "floor")
        if texture_mat:
            return texture_mat

    families = layout.get(
        "floor_material_families",
        ["matte_concrete", "glossy_concrete", "wood", "tile", "subtle_checker", "painted"],
    )
    family = rng.choice(families or ["matte_concrete"])
    texture_strength, bump_strength = receiver_texture_ranges(config, rng)

    if family == "glossy_concrete":
        color = muted_receiver_color(rng, (0.34, 0.62))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.24, 0.52), specular=rng.uniform(0.35, 0.62))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_noise_base_color(mat, rng, bsdf, color, texture_strength, (8.0, 36.0), detail=10.0)
            add_noise_bump(mat, rng, bsdf, bump_strength * 0.7, (18.0, 55.0))
        return mat

    if family == "wood":
        base = hsv_to_rgb(rng.uniform(0.065, 0.13), rng.uniform(0.18, 0.42), rng.uniform(0.34, 0.62))
        grain = jitter_color(rng, mix_color(base, (0.95, 0.78, 0.45), 0.28), 0.12)
        mat = make_receiver_principled_mat(name, base, roughness=rng.uniform(0.38, 0.72), specular=rng.uniform(0.22, 0.45))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_wave_base_color(mat, rng, bsdf, base, grain, (6.0, 18.0))
            add_noise_bump(mat, rng, bsdf, bump_strength, (18.0, 70.0))
        return mat

    if family == "tile":
        color = muted_receiver_color(rng, (0.42, 0.78))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.34, 0.78), specular=rng.uniform(0.18, 0.46))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_brick_base_color(mat, rng, bsdf, color, texture_strength, (4.0, 12.0))
            add_noise_bump(mat, rng, bsdf, bump_strength * 0.55, (25.0, 80.0))
        return mat

    if family == "subtle_checker":
        color = muted_receiver_color(rng, (0.42, 0.72))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.48, 0.86), specular=rng.uniform(0.16, 0.36))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_checker_base_color(mat, rng, bsdf, color, texture_strength * 0.75, (3.0, 10.0))
            add_noise_bump(mat, rng, bsdf, bump_strength * 0.5, (20.0, 60.0))
        return mat

    if family == "painted":
        color = muted_receiver_color(rng, (0.40, 0.78))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.42, 0.84), specular=rng.uniform(0.12, 0.35))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_noise_base_color(mat, rng, bsdf, color, texture_strength * 0.7, (5.0, 18.0), detail=5.0)
            add_noise_bump(mat, rng, bsdf, bump_strength * 0.45, (16.0, 42.0))
        return mat

    color = muted_receiver_color(rng, (0.36, 0.68))
    mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.62, 0.94), specular=rng.uniform(0.08, 0.28))
    set_receiver_material_meta(mat, "procedural", family)
    bsdf = get_principled_bsdf(mat)
    if bsdf:
        add_noise_base_color(mat, rng, bsdf, color, texture_strength, (10.0, 42.0), detail=9.0)
        add_noise_bump(mat, rng, bsdf, bump_strength, (18.0, 65.0))
    return mat


def wall_material(name: str, rng: random.Random, config: dict) -> bpy.types.Material:
    layout = config["layout"]
    if not layout.get("randomize_receiver_material", True):
        mat = make_receiver_principled_mat(name, (0.58, 0.58, 0.58), roughness=0.86, specular=0.18)
        set_receiver_material_meta(mat, "procedural", "matte_neutral")
        return mat

    texture_entry = choose_receiver_texture(config, rng, "wall")
    if texture_entry:
        texture_mat = make_receiver_texture_material(name, texture_entry, rng, config, "wall")
        if texture_mat:
            return texture_mat

    families = layout.get(
        "wall_material_families",
        ["matte_plaster", "painted_wall", "subtle_noise", "large_panels", "concrete"],
    )
    family = rng.choice(families or ["matte_plaster"])
    texture_strength, bump_strength = receiver_texture_ranges(config, rng)
    wall_texture_strength = texture_strength * 0.65
    wall_bump_strength = bump_strength * 0.55

    if family == "large_panels":
        color = muted_receiver_color(rng, (0.52, 0.84))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.68, 0.94), specular=rng.uniform(0.08, 0.24))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_checker_base_color(mat, rng, bsdf, color, wall_texture_strength, (1.0, 3.2))
            add_noise_bump(mat, rng, bsdf, wall_bump_strength * 0.35, (10.0, 32.0))
        return mat

    if family == "concrete":
        color = muted_receiver_color(rng, (0.38, 0.70))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.74, 0.96), specular=rng.uniform(0.06, 0.22))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_noise_base_color(mat, rng, bsdf, color, wall_texture_strength, (12.0, 46.0), detail=10.0)
            add_noise_bump(mat, rng, bsdf, wall_bump_strength, (18.0, 75.0))
        return mat

    if family == "subtle_noise":
        color = muted_receiver_color(rng, (0.50, 0.86))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.70, 0.98), specular=rng.uniform(0.05, 0.20))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_noise_base_color(mat, rng, bsdf, color, wall_texture_strength, (5.0, 24.0), detail=6.0)
            add_noise_bump(mat, rng, bsdf, wall_bump_strength * 0.45, (16.0, 48.0))
        return mat

    if family == "painted_wall":
        color = muted_receiver_color(rng, (0.55, 0.90))
        mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.62, 0.92), specular=rng.uniform(0.08, 0.26))
        set_receiver_material_meta(mat, "procedural", family)
        bsdf = get_principled_bsdf(mat)
        if bsdf:
            add_noise_base_color(mat, rng, bsdf, color, wall_texture_strength * 0.55, (3.0, 12.0), detail=4.0)
            add_noise_bump(mat, rng, bsdf, wall_bump_strength * 0.3, (12.0, 34.0))
        return mat

    color = muted_receiver_color(rng, (0.54, 0.88))
    mat = make_receiver_principled_mat(name, color, roughness=rng.uniform(0.76, 0.98), specular=rng.uniform(0.04, 0.18))
    set_receiver_material_meta(mat, "procedural", family)
    bsdf = get_principled_bsdf(mat)
    if bsdf:
        add_noise_base_color(mat, rng, bsdf, color, wall_texture_strength * 0.7, (7.0, 28.0), detail=7.0)
        add_noise_bump(mat, rng, bsdf, wall_bump_strength * 0.5, (16.0, 50.0))
    return mat


def receiver_material(name: str, rng: random.Random, config: dict) -> bpy.types.Material:
    return floor_material(name, rng, config)


def create_wall(name: str, y: float, config: dict, rng: random.Random) -> bpy.types.Object:
    layout = config["layout"]
    size = float(layout.get("ground_size", 8.0))
    height = float(layout.get("wall_height", 3.0))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, y, height * 0.5))
    wall = bpy.context.object
    wall.name = name
    wall.scale = (size * 0.5, height * 0.5, 1.0)
    wall.rotation_euler[0] = math.radians(90)
    wall.data.materials.append(wall_material(f"{name}_mat", rng, config))
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


def objects_fit_camera(objects: list[bpy.types.Object], camera: bpy.types.Object, margin: float) -> bool:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            co = world_to_camera_view(scene, camera, eval_obj.matrix_world @ Vector(corner))
            if co.z <= 0.0:
                return False
            if co.x < margin or co.x > 1.0 - margin or co.y < margin or co.y > 1.0 - margin:
                return False
    return True


def points_fit_camera(points: list[Vector], camera: bpy.types.Object, margin: float) -> bool:
    scene = bpy.context.scene
    for point in points:
        co = world_to_camera_view(scene, camera, point)
        if co.z <= 0.0:
            return False
        if co.x < margin or co.x > 1.0 - margin or co.y < margin or co.y > 1.0 - margin:
            return False
    return True


def fit_camera_to_objects(
    camera: bpy.types.Object,
    objects: list[bpy.types.Object],
    look_target: Vector,
    margin: float = 0.08,
    max_steps: int = 48,
) -> dict:
    adjusted = False
    bpy.context.view_layer.update()
    for _ in range(max_steps):
        if objects_fit_camera(objects, camera, margin):
            break
        direction = Vector(camera.location) - look_target
        if direction.length <= 1e-6:
            break
        camera.location = look_target + direction * 1.08
        look_at(camera, look_target)
        bpy.context.view_layer.update()
        adjusted = True
    return {
        "adjusted": adjusted,
        "location": vec_to_list(Vector(camera.location)),
        "distance": float((Vector(camera.location) - look_target).length),
        "margin": margin,
    }


def fit_camera_to_points(
    camera: bpy.types.Object,
    points: list[Vector],
    look_target: Vector,
    margin: float = 0.04,
    max_steps: int = 64,
) -> None:
    bpy.context.view_layer.update()
    for _ in range(max_steps):
        if points_fit_camera(points, camera, margin):
            break
        direction = Vector(camera.location) - look_target
        if direction.length <= 1e-6:
            break
        camera.location = look_target + direction * 1.08
        look_at(camera, look_target)
        bpy.context.view_layer.update()


def camera_basis(camera: bpy.types.Object) -> tuple[Vector, Vector, Vector]:
    rot = camera.matrix_world.to_quaternion()
    right = (rot @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (rot @ Vector((0.0, 1.0, 0.0))).normalized()
    forward = (rot @ Vector((0.0, 0.0, -1.0))).normalized()
    return right, up, forward


def canonical_to_world(p: list[float], camera: bpy.types.Object, config: dict, center: Vector) -> Vector:
    canonical = config["canonical"]
    right, up, forward = camera_basis(camera)
    camera_location = Vector(camera.location)
    center_depth = max((center - camera_location).dot(forward), 0.1)
    image_fraction = float(canonical.get("image_plane_fraction", 0.68))
    depth_fraction = float(canonical.get("depth_fraction", 0.28))
    depth = max(0.1, center_depth + float(p[2]) * center_depth * depth_fraction)
    half_width = depth * math.tan(camera.data.angle_x * 0.5) * image_fraction
    half_height = depth * math.tan(camera.data.angle_y * 0.5) * image_fraction
    return camera_location + forward * depth + right * (float(p[0]) * half_width) + up * (float(p[1]) * half_height)


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


def choose_hdri_path(config: dict, rng: random.Random, explicit_path: str | None = None) -> tuple[str | None, str]:
    mode = str(config.get("_hdri_mode", config.get("ambient", {}).get("hdri_mode", "on"))).lower()
    if mode == "off":
        return None, "off"
    if mode == "random":
        probability = float(config.get("ambient", {}).get("hdri_probability", 0.5))
        if rng.random() > probability:
            return None, "random_off"
    if explicit_path:
        return explicit_path, mode
    hdris = config.get("_runtime", {}).get("hdris", [])
    return (rng.choice(hdris) if hdris else None), mode


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


def random_float_range(rng: random.Random, value: list[float] | tuple[float, ...] | None, fallback: float) -> tuple[float, float]:
    if value is None or len(value) < 2:
        return fallback, fallback
    return float(value[0]), float(value[1])


def sample_spatial_light_color(spatial: dict, rng: random.Random) -> list[float]:
    color_range = spatial.get("color_range")
    if color_range and len(color_range) >= 2:
        lo, hi = color_range[0], color_range[1]
        if not isinstance(lo, (list, tuple)):
            lo = [float(lo)] * 3
        if not isinstance(hi, (list, tuple)):
            hi = [float(hi)] * 3
        return [rng.uniform(float(lo[i]), float(hi[i])) for i in range(3)]
    color = spatial.get("color", [1.0, 1.0, 1.0])
    return [float(color[0]), float(color[1]), float(color[2])]


def sample_spatial_light_settings(spatial: dict, rng: random.Random) -> dict:
    base_energy = float(spatial.get("base_energy", 500.0))
    intensity_lo, intensity_hi = random_float_range(rng, spatial.get("intensity_range"), 1.0)
    radius_lo, radius_hi = random_float_range(rng, spatial.get("radius_range"), float(spatial.get("fixed_radius", 0.06)))
    intensity_scalar = rng.uniform(intensity_lo, intensity_hi)
    radius = rng.uniform(radius_lo, radius_hi)
    return {
        "color": sample_spatial_light_color(spatial, rng),
        "intensity_scalar": intensity_scalar,
        "base_energy": base_energy,
        "energy": base_energy * intensity_scalar,
        "radius": radius,
    }


def render_exr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = str(scene.get("tl_exr_color_depth", "16"))
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


def component_formats(config: dict) -> set[str]:
    fmt = str(config.get("_component_format", config.get("render", {}).get("component_format", "exr"))).lower()
    if fmt == "both":
        return {"exr", "png"}
    return {fmt}


def primary_component_format(config: dict) -> str:
    fmt = str(config.get("_component_format", config.get("render", {}).get("component_format", "exr"))).lower()
    return "png" if fmt == "png" else "exr"


def tonemap_exr_to_png(exr_path: Path, png_path: Path, gamma: float = 2.2) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    src = bpy.data.images.load(str(exr_path), check_existing=False)
    dst = None
    try:
        width, height = int(src.size[0]), int(src.size[1])
        channels = int(src.channels)
        src_pixels = array("f", [0.0]) * (width * height * channels)
        src.pixels.foreach_get(src_pixels)

        inv_gamma = 1.0 / gamma if gamma > 0.0 else 1.0
        dst_pixels = array("f", [0.0]) * (width * height * 4)
        for pixel_index in range(width * height):
            src_offset = pixel_index * channels
            dst_offset = pixel_index * 4
            for channel in range(3):
                value = src_pixels[src_offset + channel] if channel < channels else 0.0
                value = max(float(value), 0.0)
                value = value / (1.0 + value)
                if gamma > 0.0:
                    value = value ** inv_gamma
                dst_pixels[dst_offset + channel] = min(max(value, 0.0), 1.0)
            dst_pixels[dst_offset + 3] = 1.0

        dst = bpy.data.images.new(f"TL_tonemap_{png_path.stem}", width=width, height=height, alpha=True, float_buffer=False)
        dst.pixels.foreach_set(dst_pixels)
        dst.filepath_raw = str(png_path)
        dst.file_format = "PNG"
        dst.save()
    finally:
        if dst is not None:
            bpy.data.images.remove(dst)
        bpy.data.images.remove(src)


def render_component(scene_dir: Path, rel_base: str, config: dict) -> dict:
    formats = component_formats(config)
    primary = primary_component_format(config)
    exr_rel = f"{rel_base}.exr"
    png_rel = f"{rel_base}.png"
    exr_path = scene_dir / exr_rel
    png_path = scene_dir / png_rel

    temp_exr = exr_path
    remove_temp = False
    if "exr" not in formats:
        temp_exr = exr_path.with_name(f".{exr_path.stem}.tmp.exr")
        remove_temp = True

    render_exr(temp_exr)
    result = {"primary": png_rel if primary == "png" else exr_rel}
    if "png" in formats:
        tonemap_exr_to_png(temp_exr, png_path, float(config.get("render", {}).get("component_png_gamma", 2.2)))
        result["png"] = png_rel
    if "exr" in formats:
        result["exr"] = exr_rel
    if remove_temp:
        temp_exr.unlink(missing_ok=True)
    return result


def copy_component(scene_dir: Path, src_base: str, dst_base: str, config: dict) -> dict:
    formats = component_formats(config)
    primary = primary_component_format(config)
    result = {"primary": f"{dst_base}.png" if primary == "png" else f"{dst_base}.exr"}
    for fmt in formats:
        src = scene_dir / f"{src_base}.{fmt}"
        dst = scene_dir / f"{dst_base}.{fmt}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        result[fmt] = f"{dst_base}.{fmt}"
    return result


def write_component_preview_png(scene_dir: Path, output: dict, preview_rel: str, config: dict) -> str:
    preview_path = scene_dir / preview_rel
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if output.get("png"):
        shutil.copyfile(scene_dir / output["png"], preview_path)
        return preview_rel
    if output.get("exr"):
        tonemap_exr_to_png(scene_dir / output["exr"], preview_path, float(config.get("render", {}).get("component_png_gamma", 2.2)))
        return preview_rel
    raise RuntimeError(f"Cannot write preview PNG for component output: {output}")


def component_meta(output: dict) -> dict:
    meta = {"render": output["primary"]}
    if "exr" in output:
        meta["render_exr"] = output["exr"]
    if "png" in output:
        meta["render_png"] = output["png"]
    return meta


def first_existing_socket(outputs, names: list[str]):
    for name in names:
        if name in outputs:
            return outputs[name]
    return None


def add_file_output_node(
    tree,
    name: str,
    base_path: Path,
    prefix: str,
    file_format: str = "OPEN_EXR",
    color_mode: str = "RGB",
    color_depth: str = "16",
):
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = name
    node.label = name
    node.base_path = str(base_path)
    node.file_slots[0].path = f"{prefix}_"
    node.format.file_format = file_format
    node.format.color_mode = color_mode
    node.format.color_depth = color_depth
    return node


def move_compositor_output(tmp_dir: Path, prefix: str, target: Path) -> str:
    matches = sorted(tmp_dir.glob(f"{prefix}_*{target.suffix}"))
    if not matches:
        raise RuntimeError(f"Compositor did not write {prefix} EXR in {tmp_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.move(str(matches[-1]), str(target))
    return str(target)


def material_principled_input(mat: bpy.types.Material | None, input_name: str, fallback):
    if not mat or not mat.use_nodes:
        return fallback
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf or input_name not in bsdf.inputs:
        return fallback
    value = bsdf.inputs[input_name].default_value
    if input_name == "Base Color":
        return tuple(float(value[i]) for i in range(3))
    return float(value)


def make_property_override_material(name: str, value: float) -> bpy.types.Material:
    v = max(0.0, min(1.0, float(value)))
    return make_emission_mat(name, (v, v, v), strength=1.0)


def object_material_snapshot() -> dict[str, list[bpy.types.Material]]:
    return {obj.name: list(obj.data.materials) for obj in bpy.data.objects if obj.type == "MESH"}


def restore_object_materials(snapshot: dict[str, list[bpy.types.Material]]) -> None:
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name in snapshot:
            obj.data.materials.clear()
            for mat in snapshot[obj.name]:
                obj.data.materials.append(mat)


def render_material_property_map(scene_dir: Path, rel_path: str, property_name: str, fallback: float) -> str:
    snapshot = object_material_snapshot()
    cache: dict[float, bpy.types.Material] = {}
    try:
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            original = snapshot.get(obj.name, [])
            obj.data.materials.clear()
            source_materials = original or [None]
            for slot_index, mat in enumerate(source_materials):
                value = material_principled_input(mat, property_name, fallback)
                key = round(float(value), 4)
                if key not in cache:
                    safe_name = property_name.lower().replace(" ", "_")
                    cache[key] = make_property_override_material(f"TL_pbr_{safe_name}_{slot_index}_{key:.4f}", key)
                obj.data.materials.append(cache[key])
        set_black_world()
        render_exr(scene_dir / rel_path)
    finally:
        restore_object_materials(snapshot)
    return rel_path


def lerp_color(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def depth_preview_color(t: float) -> tuple[float, float, float]:
    stops = [
        (0.00, (0.95, 0.05, 0.18)),
        (0.32, (1.00, 0.72, 0.18)),
        (0.55, (0.78, 0.95, 0.38)),
        (0.78, (0.00, 0.72, 0.86)),
        (1.00, (0.08, 0.10, 0.55)),
    ]
    t = max(0.0, min(1.0, float(t)))
    for i in range(len(stops) - 1):
        left_t, left_color = stops[i]
        right_t, right_color = stops[i + 1]
        if left_t <= t <= right_t:
            local_t = 0.0 if right_t == left_t else (t - left_t) / (right_t - left_t)
            return lerp_color(left_color, right_color, local_t)
    return stops[-1][1]


def render_depth_preview_png(exr_path: Path, png_path: Path) -> dict:
    image = bpy.data.images.load(str(exr_path), check_existing=False)
    try:
        width, height = image.size
        pixels = list(image.pixels)
        values = [
            float(pixels[i])
            for i in range(0, len(pixels), 4)
            if math.isfinite(float(pixels[i])) and 0.0 < float(pixels[i]) < 1.0e6
        ]
        if not values:
            depth_min, depth_max = 0.0, 1.0
        else:
            values.sort()
            low_index = int((len(values) - 1) * 0.01)
            high_index = int((len(values) - 1) * 0.99)
            depth_min = values[low_index]
            depth_max = values[high_index]
            if depth_max <= depth_min + 1e-6:
                depth_max = depth_min + 1.0

        denom = depth_max - depth_min
        out_pixels = []
        for i in range(0, len(pixels), 4):
            depth = float(pixels[i])
            if not math.isfinite(depth) or depth <= 0.0:
                t = 1.0
            else:
                t = (depth - depth_min) / denom
            color = depth_preview_color(t)
            out_pixels.extend((color[0], color[1], color[2], 1.0))

        png_path.parent.mkdir(parents=True, exist_ok=True)
        preview = bpy.data.images.new(f"{exr_path.stem}_preview", width=width, height=height, alpha=True, float_buffer=False)
        try:
            preview.pixels.foreach_set(out_pixels)
            preview.filepath_raw = str(png_path)
            preview.file_format = "PNG"
            preview.save()
        finally:
            bpy.data.images.remove(preview)
        return {"min_meters": float(depth_min), "max_meters": float(depth_max), "percentiles": [1.0, 99.0]}
    finally:
        bpy.data.images.remove(image)


def render_pbr_pass_maps(scene_dir: Path, config: dict) -> dict:
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    old_use_nodes = scene.use_nodes
    old_tree_nodes = None
    old_tree_links = None
    if scene.node_tree:
        old_tree_nodes = list(scene.node_tree.nodes)
        old_tree_links = list(scene.node_tree.links)

    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")

    tmp_dir = scene_dir / "pbr" / "_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    depth_out = add_file_output_node(tree, "TL_PBR_Depth", tmp_dir, "depth", color_mode="RGB")
    normal_out = add_file_output_node(tree, "TL_PBR_Normal", tmp_dir, "normal", color_mode="RGB")
    albedo_out = add_file_output_node(tree, "TL_PBR_Albedo", tmp_dir, "albedo", color_mode="RGB")

    depth_socket = first_existing_socket(render_layers.outputs, ["Depth", "Z"])
    normal_socket = first_existing_socket(render_layers.outputs, ["Normal"])
    albedo_socket = first_existing_socket(render_layers.outputs, ["DiffCol", "Diffuse Color", "Albedo"])
    if not all([depth_socket, normal_socket, albedo_socket]):
        raise RuntimeError("Required render pass sockets are not available for PBR map rendering.")

    raw_depth_rgb = tree.nodes.new("CompositorNodeCombRGBA")
    for channel in ("R", "G", "B"):
        tree.links.new(depth_socket, raw_depth_rgb.inputs[channel])
    raw_depth_rgb.inputs["A"].default_value = 1.0

    tree.links.new(raw_depth_rgb.outputs["Image"], depth_out.inputs[0])
    tree.links.new(normal_socket, normal_out.inputs[0])
    tree.links.new(albedo_socket, albedo_out.inputs[0])

    old_filepath = scene.render.filepath
    try:
        bpy.ops.render.render(write_still=False)
        outputs = {
            "depth": "pbr/depth.exr",
            "normal": "pbr/normal.exr",
            "albedo": "pbr/albedo.exr",
        }
        for key, rel_path in outputs.items():
            move_compositor_output(tmp_dir, key, scene_dir / rel_path)
        outputs["depth_png"] = "pbr/depth.png"
        outputs["depth_png_range"] = render_depth_preview_png(scene_dir / outputs["depth"], scene_dir / outputs["depth_png"])
    finally:
        scene.render.filepath = old_filepath
        tree.nodes.clear()
        if old_tree_nodes is not None:
            # The old compositor setup is not used by this script, so restore only the use_nodes state.
            pass
        scene.use_nodes = old_use_nodes
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    return outputs


def render_pbr_maps(scene_dir: Path, config: dict) -> dict:
    outputs = render_pbr_pass_maps(scene_dir, config)
    outputs["roughness"] = render_material_property_map(scene_dir, "pbr/roughness.exr", "Roughness", 0.75)
    return outputs


def sample_spatial_positions(config: dict, rng: random.Random) -> list[list[float]]:
    spatial = config["spatial"]
    count = int(spatial.get("positions_per_scene", 64))
    pr = config["canonical"]["position_range"]
    positions: list[list[float]] = []
    if spatial.get("fixed_debug_positions_first", False):
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
    remaining = count - len(positions)
    sampling = spatial.get("sampling", "stratified_random")
    if remaining > 0 and sampling in {"stratified_random", "grid"}:
        side = max(1, math.ceil(remaining ** (1.0 / 3.0)))
        cells = [(ix, iy, iz) for iz in range(side) for iy in range(side) for ix in range(side)]
        if sampling == "stratified_random":
            rng.shuffle(cells)
        cells = cells[:remaining]
        ranges = [pr["x"], pr["y"], pr["z"]]
        for ix, iy, iz in cells:
            coords = []
            for cell_index, axis_range in zip((ix, iy, iz), ranges):
                lo, hi = float(axis_range[0]), float(axis_range[1])
                cell_lo = lo + (hi - lo) * (cell_index / side)
                cell_hi = lo + (hi - lo) * ((cell_index + 1) / side)
                if sampling == "grid":
                    coords.append((cell_lo + cell_hi) * 0.5)
                else:
                    coords.append(rng.uniform(cell_lo, cell_hi))
            positions.append(coords)
    while len(positions) < count:
        positions.append([rng.uniform(*pr["x"]), rng.uniform(*pr["y"]), rng.uniform(*pr["z"])])
    return positions[:count]


def debug_light_material(layer_index: int) -> bpy.types.Material:
    palette = [
        (0.1, 0.35, 1.0),
        (0.1, 1.0, 1.0),
        (1.0, 0.9, 0.1),
        (1.0, 0.1, 0.1),
    ]
    color = palette[layer_index % len(palette)]
    return make_emission_mat(f"TL_debug_light_pos_layer_{layer_index:02d}_mat", color, strength=0.75)


def debug_light_z_layers(positions: list[list[float]], layer_count: int = 4) -> dict[int, int]:
    ordered = sorted(enumerate(positions), key=lambda item: item[1][2])
    if not ordered:
        return {}
    per_layer = max(1, math.ceil(len(ordered) / layer_count))
    layers = {}
    for rank, (index, _position) in enumerate(ordered):
        layers[index] = min(layer_count - 1, rank // per_layer)
    return layers


def create_debug_curve_line(name: str, start: Vector, end: Vector, material: bpy.types.Material, bevel_depth: float = 0.006) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 1
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (start.x, start.y, start.z, 1.0)
    spline.points[1].co = (end.x, end.y, end.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def add_debug_light_volume_bounds(config: dict, camera: bpy.types.Object, center: Vector) -> list[bpy.types.Object]:
    pr = config["canonical"]["position_range"]
    xs = [float(pr["x"][0]), float(pr["x"][1])]
    ys = [float(pr["y"][0]), float(pr["y"][1])]
    zs = [float(pr["z"][0]), float(pr["z"][1])]
    corners = {
        (ix, iy, iz): canonical_to_world([xs[ix], ys[iy], zs[iz]], camera, config, center)
        for ix in range(2)
        for iy in range(2)
        for iz in range(2)
    }
    edges = []
    for ix in range(2):
        for iy in range(2):
            edges.append(((ix, iy, 0), (ix, iy, 1)))
    for ix in range(2):
        for iz in range(2):
            edges.append(((ix, 0, iz), (ix, 1, iz)))
    for iy in range(2):
        for iz in range(2):
            edges.append(((0, iy, iz), (1, iy, iz)))

    material = make_emission_mat("TL_debug_light_volume_bounds_mat", (1.0, 1.0, 1.0), strength=1.0)
    return [
        create_debug_curve_line(f"TL_Debug_LightVolume_{i:02d}", corners[a], corners[b], material)
        for i, (a, b) in enumerate(edges)
    ]


def debug_light_volume_points(config: dict, camera: bpy.types.Object, center: Vector) -> list[Vector]:
    pr = config["canonical"]["position_range"]
    xs = [float(pr["x"][0]), float(pr["x"][1])]
    ys = [float(pr["y"][0]), float(pr["y"][1])]
    zs = [float(pr["z"][0]), float(pr["z"][1])]
    return [
        canonical_to_world([x, y, z], camera, config, center)
        for x in xs
        for y in ys
        for z in zs
    ]


def render_light_position_preview(
    scene_dir: Path,
    positions: list[list[float]],
    config: dict,
    camera: bpy.types.Object,
    center: Vector,
) -> str:
    original_location = Vector(camera.location)
    original_rotation = camera.rotation_euler.copy()
    debug_objects = add_debug_light_volume_bounds(config, camera, center)
    z_layers = debug_light_z_layers(positions)
    for i, p_can in enumerate(positions):
        p_world = canonical_to_world(p_can, camera, config, center)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.015, location=p_world)
        marker = bpy.context.object
        marker.name = f"TL_Debug_LightPos_{i:03d}"
        marker.data.materials.append(debug_light_material(z_layers.get(i, 0)))
        debug_objects.append(marker)
    rel_path = f"../preview/{scene_dir.name}_light_positions.png"
    right, up, _forward = camera_basis(camera)
    distance = max((original_location - center).length, 0.1)
    camera.location = original_location + right * (distance * 0.12) + up * (distance * 0.10) + (original_location - center).normalized() * (distance * 0.06)
    look_at(camera, center)
    bpy.context.view_layer.update()
    try:
        render_png(scene_dir / rel_path)
    finally:
        camera.location = original_location
        camera.rotation_euler = original_rotation
        bpy.context.view_layer.update()
        for obj in debug_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    return rel_path


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
    hdri_path, hdri_mode = choose_hdri_path(config, rng)
    hdri_strength = rng.uniform(*ambient.get("hdri_strength_range", [0.8, 1.2]))
    hdri_rotation = rng.random() * 2.0 * math.pi if ambient.get("hdri_rotation_random", True) else 0.0
    source = set_hdri_world(hdri_path, hdri_strength, hdri_rotation, ambient.get("fallback_color", [0.78, 0.78, 0.78]))
    source["hdri_mode"] = hdri_mode
    remove_all_lights()
    ambient_output = render_component(scene_dir, "spatial/ambient", config)
    ambient_render = ambient_output["primary"]
    ambient_png = f"../preview/{scene_dir.name}_ambient.png"
    write_component_preview_png(scene_dir, ambient_output, ambient_png, config)
    positions = sample_spatial_positions(config, rng)
    light_position_preview = None
    if config.get("_light_preview", False):
        light_position_preview = render_light_position_preview(scene_dir, positions, config, camera, center)
    pbr_maps = render_pbr_maps(scene_dir, config) if config.get("_render_pbr", False) else None

    lights_meta = []
    spatial = config["spatial"]
    include_hdri_in_point_lights = bool(spatial.get("include_hdri_in_point_lights", False))
    if not include_hdri_in_point_lights:
        set_black_world()
    receiver_bounds = config["_runtime"].get("receiver_bounds")
    receiver_materials = config["_runtime"].get("receiver_materials", [])
    invalid_reference_source: str | None = "spatial/ambient" if include_hdri_in_point_lights else None
    with progress_bar(positions, total=len(positions), desc=f"{scene_dir.name} point lights", unit="light") as pbar:
        for light_index, p_can in enumerate(pbar):
            pbar.set_postfix(light=f"{light_index:03d}")
            p_world = canonical_to_world(p_can, camera, config, center)
            rel_base = f"spatial/point_lights/light_{light_index:03d}"
            light_settings = sample_spatial_light_settings(spatial, rng)
            radius = float(light_settings["radius"])
            valid, skip_reason = point_inside_receiver_bounds(p_world, receiver_bounds, radius)
            copied_from = None
            if valid:
                light = create_point_light(
                    f"TL_Point_{light_index:03d}",
                    p_world,
                    float(light_settings["energy"]),
                    radius,
                    light_settings["color"],
                )
                light_output = render_component(scene_dir, rel_base, config)
                bpy.data.objects.remove(light, do_unlink=True)
            else:
                if invalid_reference_source is None:
                    light_output = render_component(scene_dir, rel_base, config)
                    invalid_reference_source = rel_base
                else:
                    light_output = copy_component(scene_dir, invalid_reference_source, rel_base, config)
                    copied_from = f"{invalid_reference_source}.{primary_component_format(config)}"
            light_meta = {
                "id": light_index,
                "canonical_position": p_can,
                "world_position": vec_to_list(p_world),
                "valid": valid,
                "skip_reason": skip_reason,
                "copied_from": copied_from,
                "base_energy": float(light_settings["base_energy"]),
                "intensity_scalar": float(light_settings["intensity_scalar"]),
                "energy": float(light_settings["energy"]),
                "color": [float(c) for c in light_settings["color"]],
                "canonical_radius": radius,
                "world_radius": radius,
            }
            light_meta.update(component_meta(light_output))
            lights_meta.append(light_meta)
    return {
        "ambient_render": ambient_render,
        "ambient_output": component_meta(ambient_output),
        "ambient_png": ambient_png,
        "light_position_preview": light_position_preview,
        "ambient_source": source,
        "pbr_maps": pbr_maps,
        "light_volume_center": vec_to_list(center),
        "light_volume_center_source": "camera_look_at",
        "receiver_bounds": receiver_bounds_to_meta(receiver_bounds) if receiver_bounds else None,
        "receiver_materials": receiver_materials,
        "positions_per_scene": int(spatial.get("positions_per_scene", 64)),
        "valid_point_light_count": sum(1 for light in lights_meta if light["valid"]),
        "invalid_point_light_count": sum(1 for light in lights_meta if not light["valid"]),
        "include_hdri_in_point_lights": include_hdri_in_point_lights,
        "color_range": spatial.get("color_range"),
        "intensity_range": spatial.get("intensity_range"),
        "radius_range": spatial.get("radius_range"),
        "point_lights": lights_meta,
    }


def render_diffuse_components(scene_dir: Path, config: dict, rng: random.Random, camera: bpy.types.Object, center: Vector) -> dict:
    diffuse = config["diffuse"]
    color = tuple(diffuse.get("constant_ambient_color", [0.72, 0.72, 0.72]))
    strength = float(diffuse.get("constant_ambient_strength", 0.35))
    set_constant_world(color, strength)
    remove_all_lights()
    ambient_output = render_component(scene_dir, "diffuse/ambient_constant", config)
    ambient_render = ambient_output["primary"]

    set_black_world()
    p_can = diffuse.get("area_light_canonical_position", [-0.45, -0.65, 1.15])
    p_world = canonical_to_world(p_can, camera, config, center)
    spread_count = int(diffuse.get("spread_count", 6))
    spread_min, spread_max = diffuse.get("spread_degrees_range", [6.0, 70.0])
    norm_min, norm_max = diffuse.get("normalize_spread_to", [-1.0, 1.0])
    spread_values = sorted(rng.uniform(float(spread_min), float(spread_max)) for _ in range(spread_count))
    spreads = []
    with progress_bar(spread_values, total=len(spread_values), desc=f"{scene_dir.name} diffuse", unit="spread") as pbar:
        for i, spread_deg in enumerate(pbar):
            pbar.set_postfix(spread=f"{i:03d}")
            t = 0.0 if float(spread_max) == float(spread_min) else (spread_deg - float(spread_min)) / (float(spread_max) - float(spread_min))
            distance = max((p_world - center).length, 0.1)
            area_size = 2.0 * distance * math.tan(math.radians(spread_deg) * 0.5)
            light = create_area_light(
                f"TL_Diffuse_Area_{i:03d}",
                p_world,
                center,
                float(diffuse.get("area_light_energy", 650.0)),
                area_size,
            )
            light_output = render_component(scene_dir, f"diffuse/spread_{i:03d}", config)
            bpy.data.objects.remove(light, do_unlink=True)
            normalized = norm_min + t * (norm_max - norm_min)
            spread_meta = {
                "id": i,
                "spread_degrees": spread_deg,
                "normalized_spread": normalized,
                "area_size": area_size,
                "canonical_position": p_can,
                "world_position": vec_to_list(p_world),
            }
            spread_meta.update(component_meta(light_output))
            spreads.append(spread_meta)
    return {
        "ambient_render": ambient_render,
        "ambient_output": component_meta(ambient_output),
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

    bbox_min, bbox_max = mesh_bbox(subject_objects)
    center = (bbox_min + bbox_max) * 0.5
    camera, camera_meta = create_camera(config, rng, center)
    look_target = Vector(camera_meta["look_at"])
    framing_meta = fit_camera_to_objects(camera, subject_objects, look_target)
    camera_meta["location"] = framing_meta["location"]
    camera_meta["distance"] = framing_meta["distance"]
    camera_meta["framing"] = framing_meta
    create_receivers(config, rng, camera, center)

    if config.get("_debug_preview_only", False):
        ambient = config["ambient"]
        hdri_path, hdri_mode = choose_hdri_path(config, rng)
        hdri_strength = rng.uniform(*ambient.get("hdri_strength_range", [0.8, 1.2]))
        hdri_rotation = rng.random() * 2.0 * math.pi if ambient.get("hdri_rotation_random", True) else 0.0
        source = set_hdri_world(hdri_path, hdri_strength, hdri_rotation, ambient.get("fallback_color", [0.78, 0.78, 0.78]))
        source["hdri_mode"] = hdri_mode
        remove_all_lights()
        light_position_preview = None
        if config.get("_light_preview", False):
            positions = sample_spatial_positions(config, rng)
            light_position_preview = render_light_position_preview(scene_dir, positions, config, camera, look_target)
        meta = {
            "schema": "tokenlight_synthetic_components_v1",
            "scene_id": scene_id,
            "scene_type": "object_centric_debug_preview",
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
            "debug": {
                "preview_only": True,
                "ambient_source": source,
                "light_position_preview": light_position_preview,
                "light_volume_center": vec_to_list(look_target),
                "light_volume_center_source": "camera_look_at",
                "receiver_materials": config["_runtime"].get("receiver_materials", []),
            },
        }
        write_json(scene_dir / "meta.json", meta)
        return meta

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
        meta["spatial"] = render_spatial_components(scene_dir, config, rng, camera, look_target)
    if only in ("all", "diffuse") and config["diffuse"].get("enabled", True):
        meta["diffuse"] = render_diffuse_components(scene_dir, config, rng, camera, look_target)

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
    rng = random.Random(int(config["seed"]) + scene_index)
    output_root = resolve_path(root, config["output_root"]) or (root / "outputs/tokenlight_synthetic")
    scene_id = row.get("scene_id") or f"fixture_{scene_index:06d}"
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if row.get("camera") and row["camera"] in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects[row["camera"]]

    ambient = config["ambient"]
    hdri_path, hdri_mode = choose_hdri_path(config, rng, row.get("hdri_path"))
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

    environment_output = render_component(scene_dir, "fixtures/environment", config)
    environment_render = environment_output["primary"]

    fixtures_meta = []
    for fixture in fixtures:
        apply_fixture_states(fixtures, fixture["id"], original_materials, off_material)
        set_black_world()
        contribution_output = render_component(scene_dir, f"fixtures/fixture_{fixture['id']}/contribution", config)
        mask = render_fixture_mask(scene_dir, fixture)
        fixture_meta = {
            "id": fixture["id"],
            "prefixes": fixture.get("prefixes", []),
            "light_prefixes": fixture.get("light_prefixes", []),
            "mask_render": mask,
        }
        fixture_meta["contribution_render"] = contribution_output["primary"]
        fixture_meta["contribution_output"] = component_meta(contribution_output)
        fixtures_meta.append(fixture_meta)

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
            "environment_output": component_meta(environment_output),
            "hdri_mode": hdri_mode,
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
    if args.output is not None:
        config["output_root"] = args.output
    if args.component_format is not None:
        config["render"]["component_format"] = args.component_format
    config["_component_format"] = str(config["render"].get("component_format", "exr")).lower()
    config["_hdri_mode"] = args.hdri_mode or str(config.get("ambient", {}).get("hdri_mode", "on")).lower()
    config["_debug_preview_only"] = bool(args.debug)
    config["_light_preview"] = bool(args.light_preview or args.debug_light_preview or args.debug)
    config["_render_pbr"] = bool(args.pbr)

    object_manifest = resolve_path(root, config.get("object_manifest"))
    hdri_manifest = resolve_path(root, config.get("hdri_manifest"))
    receiver_texture_manifest = resolve_path(
        root,
        config.get("receiver_texture_manifest") or config.get("layout", {}).get("receiver_texture_manifest"),
    )
    fixture_manifest = resolve_path(root, config.get("fixture_scene_manifest"))
    config["_runtime"] = {
        "objects": load_path_lines(object_manifest, root) if object_manifest else [],
        "hdris": load_path_lines(hdri_manifest, root) if hdri_manifest else [],
        "receiver_textures": load_receiver_texture_manifest(receiver_texture_manifest, root) if receiver_texture_manifest else [],
        "fixture_scenes": normalize_fixture_rows(load_jsonl(fixture_manifest), root) if fixture_manifest else [],
    }

    scene_count = int(config.get("scene_count", 1))
    if args.max_scenes is not None:
        scene_count = min(scene_count, args.max_scenes)

    output_root = resolve_path(root, config["output_root"]) or (root / "outputs/tokenlight_synthetic")
    metas = []
    if args.only in ("all", "spatial", "diffuse"):
        scene_indices = range(args.start_index, args.start_index + scene_count)
        with progress_bar(scene_indices, total=scene_count, desc="Object scenes", unit="scene") as pbar:
            for i in pbar:
                pbar.set_postfix(scene=f"{i:06d}")
                progress_write(f"[Relighting] Rendering object scene {i}")
                metas.append(render_object_scene(i, config, root, args.only))

    if not args.debug and args.only in ("all", "fixtures") and config["fixtures"].get("enabled", True):
        fixture_rows = config["_runtime"]["fixture_scenes"]
        if fixture_rows and config["fixtures"].get("render_if_manifest_exists", True):
            with progress_bar(fixture_rows, total=len(fixture_rows), desc="Fixture scenes", unit="scene") as pbar:
                for j, row in enumerate(pbar):
                    scene_name = row.get("scene_id", j)
                    pbar.set_postfix(scene=str(scene_name))
                    progress_write(f"[Relighting] Rendering fixture scene {scene_name}")
                    metas.append(render_fixture_scene(row, j, config, root))
        else:
            progress_write("[Relighting] No fixture scene manifest found; skipping visible-fixture synthetic renders.")

    manifest = {
        "schema": "tokenlight_synthetic_dataset_manifest_v1",
        "paper_defaults": {
            "spatial_positions_per_scene": config["spatial"].get("positions_per_scene", 64),
            "diffuse_spread_count": config["diffuse"].get("spread_count", 6),
            "fov_degrees": config["camera"].get("fov_degrees", 39.6),
            "linear_rgb_components": True,
            "component_format": config["_component_format"],
            "hdri_mode": config["_hdri_mode"],
        },
        "scene_count_written": len(metas),
        "scenes": [{"scene_id": m["scene_id"], "scene_type": m["scene_type"], "meta": f"scenes/{m['scene_id']}/meta.json"} for m in metas],
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    progress_write(f"[Relighting] Wrote manifest: {output_root / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
