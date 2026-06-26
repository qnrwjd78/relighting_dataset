from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from array import array
from contextlib import contextmanager
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
    parser.add_argument(
        "--pbr-white-shading-only",
        action="store_true",
        help="Legacy diagnostic: render only white diffuse point-light shading maps and skip RGB/PBR component maps.",
    )
    parser.add_argument(
        "--pbr-white-shading",
        dest="pbr_white_shading",
        action="store_true",
        default=None,
        help="Legacy diagnostic: with --pbr, render white diffuse shading maps for each spatial point light.",
    )
    parser.add_argument(
        "--no-pbr-white-shading",
        dest="pbr_white_shading",
        action="store_false",
        help="Disable white diffuse point-light shading maps even if enabled in config.",
    )
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default=None)
    parser.add_argument("--output-format", choices=["exr", "png", "both"], dest="component_format")
    parser.add_argument("--ambient-source", choices=["hdri", "scene"], default=None)
    parser.add_argument("--point-light-mode", choices=["component", "target"], default=None)
    parser.add_argument("--hdri-mode", choices=["on", "off", "random"], default=None)
    parser.add_argument("--positions-per-scene", type=int, default=None)
    parser.add_argument("--global-diffuse", dest="global_diffuse", action="store_true", default=None)
    parser.add_argument("--no-global-diffuse", dest="global_diffuse", action="store_false")
    parser.add_argument("--per-light-diffuse", dest="per_light_diffuse", action="store_true", default=None)
    parser.add_argument("--no-per-light-diffuse", dest="per_light_diffuse", action="store_false")
    parser.add_argument(
        "--soft-light-transport",
        action="store_true",
        help="Use the previous Cycles light transport settings instead of the default direct-limited transport.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed object scene instead of recording the failure and continuing.",
    )
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
    apply_light_transport_settings(config)


def direct_light_transport_settings(config: dict) -> dict:
    render_cfg = config.get("render", {})
    raw = render_cfg.get("direct_light_transport", {})
    if not isinstance(raw, dict):
        raw = {}
    defaults = {
        "max_bounces": 4,
        "diffuse_bounces": 0,
        "glossy_bounces": 1,
        "transmission_bounces": 4,
        "transparent_max_bounces": 4,
        "volume_bounces": 0,
        "caustics": True,
    }
    result = dict(defaults)
    result.update({key: raw[key] for key in defaults if key in raw})
    return result


def apply_light_transport_settings(config: dict) -> None:
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES" or bool(config.get("_soft_light_transport", False)):
        return
    settings = direct_light_transport_settings(config)
    cycles = scene.cycles
    for name in (
        "max_bounces",
        "diffuse_bounces",
        "glossy_bounces",
        "transmission_bounces",
        "transparent_max_bounces",
        "volume_bounces",
    ):
        if hasattr(cycles, name):
            setattr(cycles, name, int(settings[name]))
    if bool(settings.get("caustics", True)):
        for name in ("caustics_reflective", "caustics_refractive"):
            if hasattr(cycles, name):
                setattr(cycles, name, True)


def light_transport_meta(config: dict) -> dict:
    if bool(config.get("_soft_light_transport", False)):
        return {"mode": "soft", "preset": "cycles_config_default"}
    settings = direct_light_transport_settings(config)
    return {"mode": "direct_limited", **settings}


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



def first_obj_value(path: Path, prefix: str) -> str | None:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(prefix):
                value = line[len(prefix) :].strip()
                return value or None
    return None


def is_hsrd100_asset(path: Path) -> bool:
    return any(part.lower() == "hsrd100" for part in path.parts)


def find_hsrd100_diffuse_texture(asset: Path, material_name: str | None) -> Path | None:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    candidates: list[Path] = []
    if material_name:
        candidates.extend(path for path in asset.parent.glob(f"{material_name}_diffuse.*") if path.suffix.lower() in suffixes)
    candidates.extend(path for path in asset.parent.glob(f"{asset.stem}*_diffuse.*") if path.suffix.lower() in suffixes)
    candidates.extend(path for path in asset.parent.glob("*diffuse.*") if path.suffix.lower() in suffixes)
    return sorted(set(candidates), key=str)[0] if candidates else None


def objects_have_image_texture(objects: list[bpy.types.Object]) -> bool:
    for obj in objects:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                    return True
    return False


def apply_image_texture_material(objects: list[bpy.types.Object], material_name: str, texture: Path) -> None:
    mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(texture), check_existing=True)
    if image_node.image is not None:
        image_node.image.colorspace_settings.name = "sRGB"
    if bsdf:
        mat.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.75
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def apply_hsrd100_texture(objects: list[bpy.types.Object], asset: Path) -> dict:
    material_name = first_obj_value(asset, "usemtl ") or f"{asset.stem}_mat"
    texture = find_hsrd100_diffuse_texture(asset, material_name)
    meta = {
        "dataset": "hsrd100",
        "texture_found": texture is not None,
        "texture_path": str(texture) if texture else None,
        "material_name": material_name,
        "texture_applied": False,
        "texture_apply_reason": None,
    }
    if texture is None:
        meta["texture_apply_reason"] = "missing_diffuse_texture"
        return meta
    if objects_have_image_texture(objects):
        meta["texture_applied"] = True
        meta["texture_apply_reason"] = "already_had_image_texture"
        return meta
    apply_image_texture_material(objects, f"TL_HSRD100_{material_name}", texture)
    meta["texture_applied"] = True
    meta["texture_apply_reason"] = "manual_diffuse_texture"
    return meta

def call_first_import_operator(path: Path, operator_names: list[str]) -> None:
    errors = []
    for operator_name in operator_names:
        op = bpy.ops
        try:
            for part in operator_name.split("."):
                op = getattr(op, part)
            op(filepath=str(path))
            return
        except Exception as exc:
            errors.append(f"{operator_name}: {exc}")
    raise RuntimeError(f"Could not import {path} with any supported Blender operator: {'; '.join(errors)}")


def import_asset_or_primitive(asset_path: str | None, primitive: str, rng: random.Random, config: dict) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    hsrd100_asset = False
    hsrd100_import_meta = None
    if asset_path:
        path = Path(asset_path)
        ext = path.suffix.lower()
        hsrd100_asset = is_hsrd100_asset(path)
        if ext == ".blend":
            with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
                data_to.objects = list(data_from.objects)
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(path))
        elif ext == ".obj":
            call_first_import_operator(path, ["wm.obj_import", "import_scene.obj"])
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(path))
        elif ext == ".stl":
            call_first_import_operator(path, ["wm.stl_import", "import_mesh.stl"])
        elif ext == ".ply":
            call_first_import_operator(path, ["wm.ply_import", "import_mesh.ply"])
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

    if asset_path and hsrd100_asset:
        hsrd100_import_meta = apply_hsrd100_texture(mesh_objects, path)

    orientation_mode = str(config["object"].get("orientation_mode", "keep") if asset_path else "keep")
    if asset_path and hsrd100_asset and orientation_mode.lower() in {"keep", "none", "off"}:
        orientation_mode = "longest_axis_up"
    orientation_meta = normalize_objects(mesh_objects, float(config["object"].get("target_size", 1.2)), orientation_mode)
    runtime = config.setdefault("_runtime", {})
    runtime["object_orientation"] = orientation_meta
    if hsrd100_import_meta is not None:
        hsrd100_import_meta["upright_mode"] = orientation_mode
        runtime["object_import_adjustments"] = hsrd100_import_meta
    else:
        runtime.pop("object_import_adjustments", None)
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


def bbox_dimensions(objects: list[bpy.types.Object]) -> Vector:
    bbox_min, bbox_max = mesh_bbox(objects)
    return bbox_max - bbox_min


def object_orientation_candidates() -> list[tuple[str, Matrix]]:
    return [
        ("keep", Matrix.Identity(4)),
        ("x+90", Matrix.Rotation(math.radians(90.0), 4, "X")),
        ("x-90", Matrix.Rotation(math.radians(-90.0), 4, "X")),
        ("y+90", Matrix.Rotation(math.radians(90.0), 4, "Y")),
        ("y-90", Matrix.Rotation(math.radians(-90.0), 4, "Y")),
    ]


def restore_root_matrices(objects: list[bpy.types.Object], matrices: dict[str, Matrix]) -> None:
    for obj in root_objects(objects):
        if obj.name in matrices:
            obj.matrix_world = matrices[obj.name].copy()
    bpy.context.view_layer.update()


def dimensions_after_transform(objects: list[bpy.types.Object], transform: Matrix, matrices: dict[str, Matrix]) -> Vector:
    restore_root_matrices(objects, matrices)
    apply_world_transform(objects, transform)
    dims = bbox_dimensions(objects)
    restore_root_matrices(objects, matrices)
    return dims


def choose_auto_ground_orientation(objects: list[bpy.types.Object]) -> tuple[str, Matrix, str]:
    original = {obj.name: obj.matrix_world.copy() for obj in root_objects(objects)}
    current = bbox_dimensions(objects)
    values = [float(current.x), float(current.y), float(current.z)]
    shortest = max(min(values), 1e-6)
    longest = max(max(values), 1e-6)
    xy_min = max(min(float(current.x), float(current.y)), 1e-6)
    xy_max = max(float(current.x), float(current.y), 1e-6)
    z = float(current.z)

    if z <= shortest * 1.05:
        return "keep", Matrix.Identity(4), "z_already_shortest"
    if shortest / longest >= 0.55:
        return "keep", Matrix.Identity(4), "ambiguous_box"
    if z <= xy_min * 0.8:
        return "keep", Matrix.Identity(4), "already_low_profile"
    if z >= xy_max * 1.2 and xy_min / xy_max >= 0.35:
        return "keep", Matrix.Identity(4), "already_tall_profile"

    best_name = "keep"
    best_matrix = Matrix.Identity(4)
    best_score = (abs(z - min(float(current.x), float(current.y), z)), 0.0)
    for name, matrix in object_orientation_candidates():
        dims = dimensions_after_transform(objects, matrix, original)
        values = [float(dims.x), float(dims.y), float(dims.z)]
        shortest = max(min(values), 1e-6)
        footprint = max(values[0] * values[1], 1e-6)
        height_gap = abs(values[2] - shortest)
        score = (height_gap, -footprint)
        if score < best_score:
            best_name = name
            best_matrix = matrix
            best_score = score
    restore_root_matrices(objects, original)
    return best_name, best_matrix, "shortest_axis_to_ground_height"


def choose_object_orientation(objects: list[bpy.types.Object], mode: str) -> tuple[str, Matrix, str]:
    mode = mode.lower()
    if mode in {"keep", "none", "off"}:
        return "keep", Matrix.Identity(4), "disabled"
    if mode in {"longest_axis_up", "upright_longest_axis"}:
        dims = bbox_dimensions(objects)
        values = [float(dims.x), float(dims.y), float(dims.z)]
        longest_axis = max(range(3), key=lambda axis: values[axis])
        if longest_axis == 0:
            return "x_to_z", Matrix.Rotation(math.radians(-90.0), 4, "Y"), "longest_axis_up"
        if longest_axis == 1:
            return "y_to_z", Matrix.Rotation(math.radians(90.0), 4, "X"), "longest_axis_up"
        return "keep", Matrix.Identity(4), "longest_axis_already_up"
    if mode in {"shortest_axis_up", "ground_min_height"}:
        original = {obj.name: obj.matrix_world.copy() for obj in root_objects(objects)}
        best_name = "keep"
        best_matrix = Matrix.Identity(4)
        best_score = None
        for name, matrix in object_orientation_candidates():
            dims = dimensions_after_transform(objects, matrix, original)
            score = (float(dims.z), -float(dims.x * dims.y))
            if best_score is None or score < best_score:
                best_name = name
                best_matrix = matrix
                best_score = score
        restore_root_matrices(objects, original)
        return best_name, best_matrix, "shortest_axis_up"
    if mode == "auto_ground":
        return choose_auto_ground_orientation(objects)
    raise ValueError(f"Unsupported object.orientation_mode: {mode}")


def normalize_objects(objects: list[bpy.types.Object], target_size: float, orientation_mode: str = "keep") -> dict:
    original_dims = bbox_dimensions(objects)
    orientation_name, orientation_matrix, orientation_reason = choose_object_orientation(objects, orientation_mode)
    if orientation_name != "keep":
        apply_world_transform(objects, orientation_matrix)
    oriented_dims = bbox_dimensions(objects)

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
    final_dims = bbox_dimensions(objects)
    return {
        "mode": orientation_mode,
        "applied": orientation_name,
        "reason": orientation_reason,
        "original_dimensions": vec_to_list(original_dims),
        "oriented_dimensions": vec_to_list(oriented_dims),
        "final_dimensions": vec_to_list(final_dims),
    }


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


def set_camera_axes(camera: bpy.types.Object, right: Vector, up: Vector, forward: Vector) -> None:
    rotation = Matrix(
        (
            (right.x, up.x, -forward.x),
            (right.y, up.y, -forward.y),
            (right.z, up.z, -forward.z),
        )
    )
    camera.rotation_euler = rotation.to_euler()


def camera_axes_from_angles(azimuth: float, elevation: float, roll: float) -> tuple[Vector, Vector, Vector]:
    view_direction = Vector(
        (
            math.cos(elevation) * math.sin(azimuth),
            -math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        )
    ).normalized()
    forward = (-view_direction).normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(world_up)) > 0.985:
        world_up = Vector((0.0, 1.0, 0.0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()
    if abs(float(roll)) > 1e-8:
        roll_matrix = Matrix.Rotation(float(roll), 4, forward)
        right = (roll_matrix @ right).normalized()
        up = (roll_matrix @ up).normalized()
    return right, up, forward


def canonical_mapping(config: dict) -> str:
    return str(config.get("canonical", {}).get("mapping", "canonical_rig")).lower()


def uses_canonical_camera_rig(config: dict) -> bool:
    mode = str(config.get("camera", {}).get("mode", canonical_mapping(config))).lower()
    return mode in {"canonical_rig", "similarity", "tokenlight"}


def canonical_camera_position(config: dict, rng: random.Random) -> Vector:
    cam_cfg = config["camera"]
    position = cam_cfg.get("canonical_position")
    if position is not None:
        return Vector((float(position[0]), float(position[1]), float(position[2])))
    if cam_cfg.get("canonical_distance") is not None:
        distance = float(cam_cfg["canonical_distance"])
    else:
        lo, hi = cam_cfg.get("canonical_distance_range", [4.5, 4.5])
        distance = rng.uniform(float(lo), float(hi))
    return Vector((0.0, -abs(distance), 0.0))


def similarity_rotation_matrix_meta(right: Vector, forward: Vector, up: Vector) -> list[list[float]]:
    return [
        [float(right.x), float(forward.x), float(up.x)],
        [float(right.y), float(forward.y), float(up.y)],
        [float(right.z), float(forward.z), float(up.z)],
    ]


def similarity_axes_from_meta(meta: dict) -> tuple[Vector, Vector, Vector]:
    axes = meta.get("axes", {})
    right = Vector(axes.get("x_axis_world", [1.0, 0.0, 0.0]))
    forward = Vector(axes.get("y_axis_world", [0.0, 1.0, 0.0]))
    up = Vector(axes.get("z_axis_world", [0.0, 0.0, 1.0]))
    return right.normalized(), forward.normalized(), up.normalized()


def canonical_axis_scale(config: dict) -> Vector:
    value = config.get("canonical", {}).get("axis_scale", [1.0, 1.0, 1.0])
    if isinstance(value, dict):
        return Vector((float(value.get("x", 1.0)), float(value.get("y", 1.0)), float(value.get("z", 1.0))))
    if isinstance(value, (int, float)):
        scale = float(value)
        return Vector((scale, scale, scale))
    return Vector((
        float(value[0]) if len(value) > 0 else 1.0,
        float(value[1]) if len(value) > 1 else 1.0,
        float(value[2]) if len(value) > 2 else 1.0,
    ))


def canonical_axis_scaled_world_scale(config: dict) -> float:
    axis_scale = canonical_axis_scale(config)
    volume_scale = abs(float(axis_scale.x * axis_scale.y * axis_scale.z)) ** (1.0 / 3.0)
    return canonical_world_scale(config) * max(volume_scale, 1e-6)


def transformed_canonical_point(
    config: dict,
    p: Vector | list[float],
    target_center: Vector | None = None,
    apply_axis_scale: bool = True,
) -> Vector:
    runtime = config.get("_runtime", {})
    transform = runtime.get("similarity_transform")
    if not transform:
        raise RuntimeError("No similarity transform has been initialized.")
    right, forward, up = similarity_axes_from_meta(transform)
    center = target_center or Vector(transform["target_center"])
    scale = float(transform["scale"])
    canonical = Vector((float(p[0]), float(p[1]), float(p[2]))) if not isinstance(p, Vector) else p
    offset = canonical - Vector(transform["canonical_center"])
    if apply_axis_scale:
        axis_scale = canonical_axis_scale(config)
        offset = Vector((offset.x * axis_scale.x, offset.y * axis_scale.y, offset.z * axis_scale.z))
    return center + right * (offset.x * scale) + forward * (offset.y * scale) + up * (offset.z * scale)


def apply_scale_multiplier(config: dict, rng: random.Random) -> float:
    runtime = config.setdefault("_runtime", {})
    base_scale = float(runtime.get("canonical_scale", canonical_world_scale(config)))
    lo, hi = config.get("canonical", {}).get("scale_multiplier_range", [0.9, 1.1])
    multiplier = rng.uniform(float(lo), float(hi))
    runtime["canonical_scale_base"] = base_scale
    runtime["canonical_scale_multiplier"] = multiplier
    runtime["canonical_scale"] = max(base_scale * multiplier, 1e-6)
    return float(runtime["canonical_scale"])


def sample_target_center(
    config: dict,
    rng: random.Random,
    bbox_center: Vector,
    right: Vector,
    forward: Vector,
    up: Vector,
    scale: float,
) -> Vector:
    jitter = config.get("canonical", {}).get("target_center_jitter", [0.15, 0.15, 0.1])
    if not isinstance(jitter, (list, tuple)):
        jitter = [float(jitter)] * 3
    values = [float(jitter[i]) if i < len(jitter) else 0.0 for i in range(3)]
    offset = (
        right * rng.uniform(-values[0], values[0])
        + forward * rng.uniform(-values[1], values[1])
        + up * rng.uniform(-values[2], values[2])
    )
    return bbox_center + offset * scale


def initialize_similarity_transform(
    config: dict,
    rng: random.Random,
    bbox_center: Vector,
    azimuth: float,
    elevation: float,
    roll: float,
) -> dict:
    right, up, forward = camera_axes_from_angles(azimuth, elevation, roll)
    scale = apply_scale_multiplier(config, rng)
    target_center = sample_target_center(config, rng, bbox_center, right, forward, up, scale)
    transform = {
        "canonical_center": vec_to_list(canonical_center(config)),
        "target_center": vec_to_list(target_center),
        "bbox_center": vec_to_list(bbox_center),
        "scale": float(scale),
        "scale_base": float(config.get("_runtime", {}).get("canonical_scale_base", scale)),
        "scale_multiplier": float(config.get("_runtime", {}).get("canonical_scale_multiplier", 1.0)),
        "rotation_degrees": {
            "azimuth": math.degrees(azimuth),
            "elevation": math.degrees(elevation),
            "roll": math.degrees(roll),
        },
        "rotation_matrix": similarity_rotation_matrix_meta(right, forward, up),
        "axes": {
            "x_axis_world": vec_to_list(right),
            "y_axis_world": vec_to_list(forward),
            "z_axis_world": vec_to_list(up),
        },
        "formula": {
            "position": "p_world = C_t + s * R * (p_canonical - C)",
            "energy": "E_world = s^2 * E_canonical",
            "radius": "d_world = s * d_canonical",
        },
    }
    config.setdefault("_runtime", {})["similarity_transform"] = transform
    return transform


def create_camera(config: dict, rng: random.Random, center: Vector) -> tuple[bpy.types.Object, dict]:
    cam_cfg = config["camera"]
    fov = math.radians(float(cam_cfg.get("fov_degrees", 39.6)))
    az = math.radians(rng.uniform(*cam_cfg.get("azimuth_degrees_range", [-35.0, 35.0])))
    el = math.radians(rng.uniform(*cam_cfg.get("elevation_degrees_range", [4.0, 24.0])))
    roll_lo, roll_hi = cam_cfg.get("roll_degrees_range", [-180.0, 180.0])
    roll = math.radians(rng.uniform(float(roll_lo), float(roll_hi)))

    canonical_distance = None
    camera_position_can = None
    rig_scale = canonical_world_scale(config)
    if uses_canonical_camera_rig(config):
        camera_position_can = canonical_camera_position(config, rng)
        transform = initialize_similarity_transform(config, rng, center, az, el, roll)
        rig_scale = float(transform["scale"])
        location = transformed_canonical_point(config, camera_position_can, apply_axis_scale=False)
        look_target = Vector(transform["target_center"])
        canonical_distance = float((camera_position_can - canonical_center(config)).length)
        distance = float((location - look_target).length)
        right, forward, up = similarity_axes_from_meta(transform)
    else:
        distance = rng.uniform(*cam_cfg.get("distance_range", [2.8, 3.6]))
        jitter = float(cam_cfg.get("look_at_jitter", 0.06))
        look_target = center + Vector((rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)))
        view_direction = Vector(
            (
                math.cos(el) * math.sin(az),
                -math.cos(el) * math.cos(az),
                math.sin(el),
            )
        ).normalized()
        location = center + view_direction * distance
        right, up, forward = camera_axes_from_angles(az, el, roll)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam_data.angle = fov
    if uses_canonical_camera_rig(config):
        set_camera_axes(cam, right, up, forward)
    else:
        look_at(cam, look_target)
    bpy.context.scene.camera = cam

    meta = {
        "location": vec_to_list(location),
        "look_at": vec_to_list(look_target),
        "fov_degrees": float(cam_cfg.get("fov_degrees", 39.6)),
        "distance": distance,
        "azimuth_degrees": math.degrees(az),
        "elevation_degrees": math.degrees(el),
        "roll_degrees": math.degrees(roll),
        "mode": str(cam_cfg.get("mode", canonical_mapping(config))),
        "canonical_distance": canonical_distance,
        "canonical_position": vec_to_list(camera_position_can) if camera_position_can is not None else None,
        "canonical_scale": rig_scale,
        "distance_over_scale": float(distance / max(rig_scale, 1e-6)),
        "similarity_transform": config.get("_runtime", {}).get("similarity_transform"),
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


def canonical_center(config: dict) -> Vector:
    value = config["canonical"].get("center", [0.0, 0.0, 0.0])
    return Vector((float(value[0]), float(value[1]), float(value[2])))


def canonical_range_size(config: dict) -> float:
    pr = config["canonical"].get("position_range", {})
    sizes = []
    for axis in ("x", "y", "z"):
        lo, hi = pr.get(axis, [-1.0, 1.0])
        sizes.append(abs(float(hi) - float(lo)))
    return max(max(sizes), 1e-6)


def canonical_axis_half_range(config: dict, axis: str) -> float:
    pr = config["canonical"].get("position_range", {})
    center = canonical_center(config)
    center_value = {"x": center.x, "y": center.y, "z": center.z}[axis]
    lo, hi = pr.get(axis, [-1.0, 1.0])
    return max(abs(float(lo) - center_value), abs(float(hi) - center_value), 1e-6)


def compute_canonical_scale(config: dict, bbox_min: Vector, bbox_max: Vector) -> float:
    canonical = config["canonical"]
    if canonical.get("world_scale") is not None:
        scale = float(canonical["world_scale"])
    else:
        size = bbox_max - bbox_min
        max_extent = max(float(size.x), float(size.y), float(size.z), 1e-6)
        margin = float(canonical.get("scale_margin", 1.25))
        scale = margin * max_extent / canonical_range_size(config)
    if canonical.get("min_world_scale") is not None:
        scale = max(scale, float(canonical["min_world_scale"]))
    if canonical.get("max_world_scale") is not None:
        scale = min(scale, float(canonical["max_world_scale"]))
    return max(scale, 1e-6)


def set_canonical_runtime_transform(config: dict, bbox_min: Vector, bbox_max: Vector) -> None:
    config.setdefault("_runtime", {})
    config["_runtime"]["canonical_scale"] = compute_canonical_scale(config, bbox_min, bbox_max)


def canonical_world_scale(config: dict) -> float:
    runtime = config.get("_runtime", {})
    if runtime.get("canonical_scale") is not None:
        return float(runtime["canonical_scale"])
    reference_size = float(config["canonical"].get("reference_object_size", 1.0))
    return max(reference_size / canonical_range_size(config), 1e-6)


def canonical_to_world(p: list[float], camera: bpy.types.Object, config: dict, center: Vector) -> Vector:
    if config.get("_runtime", {}).get("similarity_transform") and canonical_mapping(config) != "camera_frustum":
        return transformed_canonical_point(config, p, apply_axis_scale=True)

    right, up, forward = camera_basis(camera)
    offset = Vector((float(p[0]), float(p[1]), float(p[2]))) - canonical_center(config)
    mapping = canonical_mapping(config)
    if mapping == "camera_frustum":
        camera_location = Vector(camera.location)
        center_depth = max((center - camera_location).dot(forward), 0.1)
        y_norm = offset.y / canonical_axis_half_range(config, "y")
        x_norm = offset.x / canonical_axis_half_range(config, "x")
        z_norm = offset.z / canonical_axis_half_range(config, "z")
        depth_fraction = float(config["canonical"].get("depth_fraction", 0.28))
        depth = max(0.1, center_depth + y_norm * center_depth * depth_fraction)
        image_fraction_x = float(config["canonical"].get("image_plane_fraction_x", config["canonical"].get("image_plane_fraction", 0.68)))
        image_fraction_z = float(config["canonical"].get("image_plane_fraction_z", config["canonical"].get("image_plane_fraction", 0.68)))
        half_width = depth * math.tan(camera.data.angle_x * 0.5) * image_fraction_x
        half_height = depth * math.tan(camera.data.angle_y * 0.5) * image_fraction_z
        return camera_location + forward * depth + right * (x_norm * half_width) + up * (z_norm * half_height)

    scale = canonical_world_scale(config)
    axis_scale = canonical_axis_scale(config)
    return (
        center
        + right * (offset.x * scale * axis_scale.x)
        + forward * (offset.y * scale * axis_scale.y)
        + up * (offset.z * scale * axis_scale.z)
    )


def canonical_transform_meta(config: dict, camera: bpy.types.Object, center: Vector) -> dict:
    similarity = config.get("_runtime", {}).get("similarity_transform")
    if similarity:
        right, forward, up = similarity_axes_from_meta(similarity)
    else:
        right, up, forward = camera_basis(camera)
    scale = float(canonical_world_scale(config))
    depth_center = Vector(similarity["target_center"]) if similarity else Vector(center)
    camera_depth = max((depth_center - Vector(camera.location)).dot(forward), 0.0)
    axis_scale = canonical_axis_scale(config)
    axis_world_scale = Vector((scale * axis_scale.x, scale * axis_scale.y, scale * axis_scale.z))
    meta = {
        "target_center": vec_to_list(center),
        "canonical_center": vec_to_list(canonical_center(config)),
        "scale": scale,
        "axis_scale": vec_to_list(axis_scale),
        "axis_world_scale": vec_to_list(axis_world_scale),
        "light_world_scale": canonical_axis_scaled_world_scale(config),
        "scale_base": config.get("_runtime", {}).get("canonical_scale_base"),
        "scale_multiplier": config.get("_runtime", {}).get("canonical_scale_multiplier"),
        "scale_rule": config["canonical"].get("scale_rule", "bbox_max_extent"),
        "mapping": config["canonical"].get("mapping", "canonical_rig"),
        "image_plane_fraction": config["canonical"].get("image_plane_fraction"),
        "depth_fraction": config["canonical"].get("depth_fraction"),
        "camera_distance": camera_depth,
        "camera_distance_over_scale": float(camera_depth / max(scale, 1e-6)),
        "camera_distance_over_light_scale": float(camera_depth / max(canonical_axis_scaled_world_scale(config), 1e-6)),
        "axis_order": "x=camera_right,y=camera_forward_depth,z=camera_up",
        "x_axis_world": vec_to_list(right),
        "y_axis_world": vec_to_list(forward),
        "z_axis_world": vec_to_list(up),
    }
    if similarity:
        meta["similarity_transform"] = similarity
        meta["target_center"] = similarity["target_center"]
        meta["rotation_matrix"] = similarity["rotation_matrix"]
        meta["rotation_degrees"] = similarity["rotation_degrees"]
    return meta


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


def scene_ambient_source_meta() -> dict:
    lights = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        lights.append(
            {
                "name": obj.name,
                "type": data.type,
                "location": vec_to_list(Vector(obj.location)),
                "energy": float(getattr(data, "energy", 0.0)),
                "color": [float(channel) for channel in getattr(data, "color", (1.0, 1.0, 1.0))],
                "hide_render": bool(obj.hide_render),
            }
        )
    return {
        "type": "scene",
        "world_preserved": True,
        "light_count": len(lights),
        "renderable_light_count": sum(1 for light in lights if not light["hide_render"]),
        "lights": lights,
    }


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


def sample_spatial_light_settings(spatial: dict, rng: random.Random, world_scale: float) -> dict:
    energy_lo, energy_hi = random_float_range(rng, spatial.get("energy_range"), float(spatial.get("base_energy", 500.0)))
    base_energy = rng.uniform(energy_lo, energy_hi)
    radius_lo, radius_hi = random_float_range(rng, spatial.get("radius_range"), float(spatial.get("fixed_radius", 0.06)))
    canonical_radius = rng.uniform(radius_lo, radius_hi)
    component_color = sample_spatial_light_color(spatial, rng)
    return {
        "component_color": [float(component_color[0]), float(component_color[1]), float(component_color[2])],
        "canonical_energy": base_energy,
        "world_energy": base_energy * world_scale * world_scale,
        "canonical_radius": canonical_radius,
        "world_radius": canonical_radius * world_scale,
    }


def control_values(config: dict, default: list[float] | None = None) -> list[float]:
    values = config.get("values", config.get("levels"))
    if values is None:
        count = int(config.get("count", 6))
        if default is not None and count == len(default):
            values = default
        else:
            values = [0.0] if count <= 1 else [i / float(count - 1) for i in range(count)]
    return [float(value) for value in values]


def normalize_control_value(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lerp_range(value: float, range_value: list[float] | tuple[float, ...]) -> float:
    lo = float(range_value[0])
    hi = float(range_value[1])
    return lo + normalize_control_value(value) * (hi - lo)


def per_light_diffuse_config(spatial: dict) -> dict:
    raw = spatial.get("per_light_diffuse", {})
    config = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}
    config.setdefault("enabled", bool(spatial.get("render_diffuse_variants", False)))
    config.setdefault("values", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    config.setdefault("radius_range", spatial.get("per_light_diffuse_radius_range", spatial.get("radius_range", [0.02, 0.25])))
    config.setdefault("radius_mapping", "linear")
    return config


def global_diffuse_config(config: dict) -> dict:
    raw = config.get("global_diffuse", {})
    result = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}
    result.setdefault("enabled", False)
    result.setdefault("values", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    result.setdefault("implementation", "dominant_area_light_spread")
    result.setdefault("constant_ambient_color", [0.72, 0.72, 0.72])
    result.setdefault("constant_ambient_strength", 0.35)
    result.setdefault("area_light_canonical_position", [-0.45, -0.65, 1.15])
    result.setdefault("area_light_energy", 650.0)
    result.setdefault("spread_degrees_range", [6.0, 70.0])
    result.setdefault("energy_scale_with_scene", True)
    result.setdefault("complete_target_variants", False)
    return result


def radius_for_diffuse_level(value: float, config: dict) -> float:
    t = normalize_control_value(value)
    if str(config.get("radius_mapping", "linear")).lower() in {"quadratic", "squared"}:
        t = t * t
    radius_range = config.get("radius_range", [0.02, 0.25])
    return float(radius_range[0]) + t * (float(radius_range[1]) - float(radius_range[0]))


def spread_degrees_for_diffuse_level(value: float, config: dict) -> float:
    return lerp_range(value, config.get("spread_degrees_range", [6.0, 70.0]))


def area_size_for_spread(location: Vector, target: Vector, spread_degrees: float) -> tuple[float, float]:
    distance = max(float((location - target).length), 0.1)
    size = 2.0 * distance * math.tan(math.radians(float(spread_degrees)) * 0.5)
    return max(size, 0.001), distance


def set_ambient_source_from_meta(source: dict, config: dict) -> dict:
    fallback_color = config.get("ambient", {}).get("fallback_color", [0.78, 0.78, 0.78])
    source_type = str(source.get("type", "")).lower()
    if source_type == "hdri":
        return set_hdri_world(source.get("path"), float(source.get("strength", 1.0)), float(source.get("rotation_z", 0.0)), fallback_color)
    if source_type == "constant":
        set_constant_world(tuple(source.get("color", fallback_color)), float(source.get("strength", 1.0)))
    return source


def round_even(value: float, minimum: int) -> int:
    rounded = max(int(round(value)), int(minimum))
    return rounded + (rounded % 2)


def hdri_blur_size(width: int, height: int, dg: float, diffuse: dict) -> tuple[int, int]:
    min_width = max(int(diffuse.get("hdri_blur_min_width", 16)), 2)
    min_height = max(int(diffuse.get("hdri_blur_min_height", 8)), 2)
    t = normalize_control_value(dg)
    target_width = int(round(float(width) * ((float(min_width) / max(float(width), 1.0)) ** t)))
    target_height = int(round(float(height) * ((float(min_height) / max(float(height), 1.0)) ** t)))
    return min(width, round_even(target_width, min_width)), min(height, round_even(target_height, min_height))


def hdri_blur_cache_root(scene_dir: Path) -> Path:
    output_root = scene_dir.parents[1] if len(scene_dir.parents) > 1 else scene_dir.parent
    return output_root / "hdri_blur_cache"


def average_hdri_to_size(src_pixels: array, width: int, height: int, channels: int, target_width: int, target_height: int) -> array:
    dst_pixels = array("f", [0.0]) * (target_width * target_height * 4)
    channels = max(int(channels), 1)
    for ty in range(target_height):
        y0 = int(ty * height / target_height)
        y1 = max(int((ty + 1) * height / target_height), y0 + 1)
        y1 = min(y1, height)
        for tx in range(target_width):
            x0 = int(tx * width / target_width)
            x1 = max(int((tx + 1) * width / target_width), x0 + 1)
            x1 = min(x1, width)
            sums = [0.0, 0.0, 0.0]
            count = 0
            for sy in range(y0, y1):
                row_offset = sy * width * channels
                for sx in range(x0, x1):
                    offset = row_offset + sx * channels
                    sums[0] += float(src_pixels[offset])
                    sums[1] += float(src_pixels[offset + min(1, channels - 1)])
                    sums[2] += float(src_pixels[offset + min(2, channels - 1)])
                    count += 1
            inv_count = 1.0 / max(float(count), 1.0)
            dst_offset = (ty * target_width + tx) * 4
            dst_pixels[dst_offset] = sums[0] * inv_count
            dst_pixels[dst_offset + 1] = sums[1] * inv_count
            dst_pixels[dst_offset + 2] = sums[2] * inv_count
            dst_pixels[dst_offset + 3] = 1.0
    return dst_pixels


def make_blurred_hdri_variant(scene_dir: Path, hdri_path: str, dg: float, diffuse: dict) -> dict:
    source_path = Path(hdri_path).resolve()
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    src = bpy.data.images.load(str(source_path), check_existing=True)
    width, height = int(src.size[0]), int(src.size[1])
    channels = int(src.channels)
    target_width, target_height = hdri_blur_size(width, height, dg, diffuse)
    cache_dir = hdri_blur_cache_root(scene_dir) / f"{source_path.stem}_{source_hash}"
    cache_path = cache_dir / f"dg_{float(dg):.3f}_{target_width}x{target_height}.exr"
    if cache_path.exists():
        return {
            "path": str(cache_path),
            "source_path": str(source_path),
            "source_resolution": [width, height],
            "resolution": [target_width, target_height],
            "blur_method": "latlong_downsample_average",
            "cached": True,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    src_pixels = array("f", [0.0]) * (width * height * channels)
    src.pixels.foreach_get(src_pixels)
    dst_pixels = average_hdri_to_size(src_pixels, width, height, channels, target_width, target_height)
    dst = bpy.data.images.new(
        f"TL_hdri_dg_{float(dg):.3f}_{target_width}x{target_height}",
        width=target_width,
        height=target_height,
        alpha=True,
        float_buffer=True,
    )
    try:
        dst.pixels.foreach_set(dst_pixels)
        dst.filepath_raw = str(cache_path)
        dst.file_format = "OPEN_EXR"
        dst.save()
    finally:
        bpy.data.images.remove(dst)
    return {
        "path": str(cache_path),
        "source_path": str(source_path),
        "source_resolution": [width, height],
        "resolution": [target_width, target_height],
        "blur_method": "latlong_downsample_average",
        "cached": False,
    }


def render_exr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = str(scene.get("tl_exr_color_depth", "16"))
    run_blender_render(path, write_still=True)


def render_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    run_blender_render(path, write_still=True)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"


def scene_render_log_path(output_path: Path) -> Path:
    output_path = output_path.resolve()
    for parent in [output_path.parent, *output_path.parents]:
        if parent.parent.name == "scenes":
            return parent / "_blender_render.log"
    return output_path.parent / "_blender_render.log"


@contextmanager
def redirect_render_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with log_path.open("ab", buffering=0) as log:
        header = f"\n--- Blender render {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        os.write(log.fileno(), header.encode("utf-8", errors="replace"))
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def run_blender_render(output_path: Path, write_still: bool) -> None:
    log_path = scene_render_log_path(output_path)
    try:
        with redirect_render_output(log_path):
            bpy.ops.render.render(write_still=write_still)
    except Exception:
        progress_write(f"[Relighting] Blender render failed; internal log: {log_path}")
        raise


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


def remove_component_files(scene_dir: Path, rel_base: str, config: dict) -> None:
    for fmt in component_formats(config):
        (scene_dir / f"{rel_base}.{fmt}").unlink(missing_ok=True)


def component_output_exr_path(scene_dir: Path, output: dict | None) -> Path | None:
    if not output:
        return None
    rel = output.get("exr") or output.get("render_exr")
    if rel is None:
        primary = output.get("primary") or output.get("render")
        if primary and str(primary).lower().endswith(".exr"):
            rel = primary
    return scene_dir / str(rel) if rel else None


def exr_luminance_stats(exr_path: Path, nonzero_threshold: float) -> dict:
    if not exr_path.exists():
        return {"error": f"missing_exr:{exr_path}", "valid_stats": False}
    image = bpy.data.images.load(str(exr_path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        channels = max(int(image.channels), 1)
        pixels = array("f", [0.0]) * (width * height * channels)
        image.pixels.foreach_get(pixels)
        values: list[float] = []
        nonzero = 0
        for pixel_index in range(width * height):
            offset = pixel_index * channels
            r = float(pixels[offset])
            g = float(pixels[offset + min(1, channels - 1)])
            b = float(pixels[offset + min(2, channels - 1)])
            lum = max((r + g + b) / 3.0, 0.0)
            if math.isfinite(lum):
                values.append(lum)
                if lum > nonzero_threshold:
                    nonzero += 1
        if not values:
            return {
                "width": width,
                "height": height,
                "valid_stats": False,
                "p99_lum": 0.0,
                "mean_lum": 0.0,
                "max_lum": 0.0,
                "nonzero_pixel_ratio": 0.0,
            }
        values.sort()
        p99_index = min(max(int((len(values) - 1) * 0.99), 0), len(values) - 1)
        mean_lum = sum(values) / float(len(values))
        return {
            "width": width,
            "height": height,
            "valid_stats": True,
            "p99_lum": float(values[p99_index]),
            "mean_lum": float(mean_lum),
            "max_lum": float(values[-1]),
            "nonzero_pixel_ratio": float(nonzero) / float(max(len(values), 1)),
            "nonzero_luminance_threshold": float(nonzero_threshold),
        }
    finally:
        bpy.data.images.remove(image)


def point_light_valid_filter_config(spatial: dict) -> dict:
    raw = spatial.get("valid_filter", {})
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    raw = dict(raw) if isinstance(raw, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "grid_resolution": int(raw.get("grid_resolution", spatial.get("grid_resolution", 4))),
        "p99_luminance_threshold": float(raw.get("p99_luminance_threshold", 0.01)),
        "nonzero_pixel_ratio_threshold": float(raw.get("nonzero_pixel_ratio_threshold", 0.001)),
        "nonzero_luminance_threshold": float(raw.get("nonzero_luminance_threshold", 1.0e-4)),
        "max_refill_attempts": int(raw.get("max_refill_attempts", 256)),
        "keep_attempt_renders": bool(raw.get("keep_attempt_renders", False)),
        "require_target_count": bool(raw.get("require_target_count", True)),
    }


def validate_point_light_component(scene_dir: Path, output: dict | None, valid_filter: dict) -> tuple[bool, dict]:
    exr_path = component_output_exr_path(scene_dir, output)
    if exr_path is None:
        stats = {
            "valid": False,
            "skip_reason": "missing_exr_for_validation",
            "thresholds": {
                "p99_luminance_threshold": float(valid_filter["p99_luminance_threshold"]),
                "nonzero_pixel_ratio_threshold": float(valid_filter["nonzero_pixel_ratio_threshold"]),
            },
        }
        return False, stats
    stats = exr_luminance_stats(exr_path, float(valid_filter["nonzero_luminance_threshold"]))
    p99_ok = float(stats.get("p99_lum", 0.0)) >= float(valid_filter["p99_luminance_threshold"])
    ratio_ok = float(stats.get("nonzero_pixel_ratio", 0.0)) >= float(valid_filter["nonzero_pixel_ratio_threshold"])
    valid = bool(stats.get("valid_stats", False)) and p99_ok and ratio_ok
    stats.update(
        {
            "valid": valid,
            "p99_ok": p99_ok,
            "nonzero_ratio_ok": ratio_ok,
            "thresholds": {
                "p99_luminance_threshold": float(valid_filter["p99_luminance_threshold"]),
                "nonzero_pixel_ratio_threshold": float(valid_filter["nonzero_pixel_ratio_threshold"]),
            },
        }
    )
    if not valid:
        stats["skip_reason"] = "black_or_too_dark_point_light"
    return valid, stats


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


def scene_world_snapshot():
    world = bpy.context.scene.world
    return world.copy() if world is not None else None


def restore_scene_world(snapshot) -> None:
    bpy.context.scene.world = snapshot


def cycles_bounce_snapshot() -> dict[str, int] | None:
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return None
    names = [
        "max_bounces",
        "diffuse_bounces",
        "glossy_bounces",
        "transmission_bounces",
        "transparent_max_bounces",
        "volume_bounces",
    ]
    return {name: int(getattr(scene.cycles, name)) for name in names if hasattr(scene.cycles, name)}


def restore_cycles_bounces(snapshot: dict[str, int] | None) -> None:
    if snapshot is None:
        return
    cycles = bpy.context.scene.cycles
    for name, value in snapshot.items():
        if hasattr(cycles, name):
            setattr(cycles, name, value)


def set_direct_only_cycles_bounces() -> None:
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return
    for name in ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces", "volume_bounces"):
        if hasattr(scene.cycles, name):
            setattr(scene.cycles, name, 0)


def set_optical_cycles_bounces(config: dict) -> None:
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return
    bounces = config.get("bounces", {})
    defaults = {
        "max_bounces": 8,
        "diffuse_bounces": 1,
        "glossy_bounces": 4,
        "transmission_bounces": 8,
        "transparent_max_bounces": 8,
        "volume_bounces": 0,
    }
    for name, default in defaults.items():
        if hasattr(scene.cycles, name):
            value = int(bounces.get(name, default)) if isinstance(bounces, dict) else default
            setattr(scene.cycles, name, max(int(getattr(scene.cycles, name)), value))
    if bool(config.get("caustics", True)):
        for name in ("caustics_reflective", "caustics_refractive"):
            if hasattr(scene.cycles, name):
                setattr(scene.cycles, name, True)


def pbr_white_shading_config(config: dict) -> dict:
    render_cfg = config.get("render", {})
    raw = render_cfg.get(
        "pbr_white_shading",
        render_cfg.get("white_shading_point_lights", config.get("pbr_white_shading", False)),
    )
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        mode = str(raw.get("mode", "optical")).lower()
        direct_only = bool(raw.get("direct_only", mode != "optical"))
        output_root = str(raw.get("output_root", "pbr/white_shading_optical" if mode == "optical" else "pbr/white_shading"))
        caustics = bool(raw.get("caustics", True))
        bounces = raw.get("bounces", {})
        force_white_light = bool(raw.get("force_white_light", True))
    else:
        enabled = bool(raw)
        mode = "optical"
        direct_only = False
        output_root = "pbr/white_shading_optical"
        caustics = True
        bounces = {}
        force_white_light = True
    return {
        "enabled": enabled,
        "mode": mode,
        "direct_only": direct_only,
        "caustics": caustics,
        "bounces": bounces if isinstance(bounces, dict) else {},
        "force_white_light": force_white_light,
        "material": "white_diffuse_optical" if mode == "optical" else "white_diffuse",
        "output_root": output_root,
    }


def white_shading_light_base(light_index: int, white_config: dict) -> str:
    output_root = str(white_config.get("output_root", "pbr/white_shading_optical")).strip("/")
    return f"{output_root}/point_light_{light_index:03d}"


def white_shading_meta_key(white_config: dict) -> str:
    mode = str(white_config.get("mode", "optical")).lower()
    return "white_shading_optical" if mode == "optical" else "white_shading"


def make_white_diffuse_override_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("TL_pbr_white_diffuse_override")
    if mat is None:
        mat = bpy.data.materials.new("TL_pbr_white_diffuse_override")
    mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    set_input(diffuse, "Color", (1.0, 1.0, 1.0, 1.0))
    set_input(diffuse, "Roughness", 0.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return mat


def socket_float(node, names: list[str], fallback: float) -> float:
    for name in names:
        if name in node.inputs:
            value = node.inputs[name].default_value
            try:
                return float(value)
            except TypeError:
                return fallback
    return fallback


def material_has_optical_name(mat: bpy.types.Material | None) -> bool:
    if mat is None:
        return False
    name = mat.name.lower()
    return any(token in name for token in ("glass", "window", "pane", "transparent", "acrylic", "lens", "water", "crystal"))


def material_has_alpha_cutout_name(mat: bpy.types.Material | None) -> bool:
    if mat is None:
        return False
    name = mat.name.lower()
    return any(token in name for token in ("cutout", "alpha_cutout", "decal", "label", "sticker", "leaf", "foliage"))


def material_is_optical(mat: bpy.types.Material | None) -> bool:
    if mat is None:
        return False
    if material_has_optical_name(mat):
        return True
    diffuse_alpha = float(getattr(mat, "diffuse_color", (1.0, 1.0, 1.0, 1.0))[3])
    if diffuse_alpha < 0.98:
        return True
    if not mat.use_nodes or mat.node_tree is None:
        return False
    optical_node_types = {"BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_TRANSLUCENT", "BSDF_REFRACTION"}
    for node in mat.node_tree.nodes:
        if node.type in optical_node_types:
            return True
        if node.type == "BSDF_PRINCIPLED":
            alpha = socket_float(node, ["Alpha"], 1.0)
            transmission = socket_float(node, ["Transmission Weight", "Transmission"], 0.0)
            if alpha < 0.98 or transmission > 0.02:
                return True
    return False


def remove_input_links(tree, socket) -> None:
    for link in list(socket.links):
        tree.links.remove(link)


def set_color_socket_white(tree, node, names: list[str]) -> None:
    for name in names:
        if name not in node.inputs:
            continue
        socket = node.inputs[name]
        remove_input_links(tree, socket)
        value = socket.default_value
        if hasattr(value, "__len__"):
            alpha = float(value[3]) if len(value) > 3 else 1.0
            socket.default_value = (1.0, 1.0, 1.0, alpha)
        else:
            socket.default_value = 1.0


def set_color_socket_value(tree, node, names: list[str], color: tuple[float, float, float, float]) -> None:
    for name in names:
        if name not in node.inputs:
            continue
        socket = node.inputs[name]
        remove_input_links(tree, socket)
        value = socket.default_value
        if hasattr(value, "__len__"):
            alpha = float(color[3]) if len(value) > 3 else 1.0
            socket.default_value = (float(color[0]), float(color[1]), float(color[2]), alpha)
        else:
            socket.default_value = float(color[0])


def set_scalar_socket_value(tree, node, names: list[str], value: float) -> None:
    for name in names:
        if name in node.inputs:
            socket = node.inputs[name]
            remove_input_links(tree, socket)
            socket.default_value = float(value)


def set_scalar_input(node, names: list[str], value: float) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value


def neutralize_material_color_channels(mat: bpy.types.Material) -> None:
    if not mat.use_nodes or mat.node_tree is None:
        return
    tree = mat.node_tree
    color_shader_nodes = {
        "BSDF_GLASS",
        "BSDF_GLOSSY",
        "BSDF_REFRACTION",
        "BSDF_TRANSLUCENT",
        "BSDF_TRANSPARENT",
        "BSDF_SHEEN",
    }
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            set_color_socket_white(tree, node, ["Base Color", "Specular Tint", "Coat Tint", "Sheen Tint", "Subsurface Color"])
            set_color_socket_value(tree, node, ["Emission Color"], (0.0, 0.0, 0.0, 1.0))
            set_scalar_socket_value(tree, node, ["Emission Strength"], 0.0)
        elif node.type in color_shader_nodes:
            set_color_socket_white(tree, node, ["Color"])
        elif node.type == "EMISSION":
            set_color_socket_value(tree, node, ["Color"], (0.0, 0.0, 0.0, 1.0))
            set_scalar_socket_value(tree, node, ["Strength"], 0.0)
        elif node.type in {"VOLUME_ABSORPTION", "VOLUME_SCATTER"}:
            set_color_socket_white(tree, node, ["Color"])


def make_colorless_optical_material(source: bpy.types.Material) -> bpy.types.Material:
    safe_name = source.name.replace("/", "_")
    mat = source.copy()
    mat.name = f"TL_pbr_white_shading_optical_{safe_name}"
    diffuse = getattr(mat, "diffuse_color", (1.0, 1.0, 1.0, 1.0))
    alpha = float(diffuse[3]) if len(diffuse) > 3 else 1.0
    mat.diffuse_color = (1.0, 1.0, 1.0, alpha)
    neutralize_material_color_channels(mat)
    return mat


def optical_override_material(
    source: bpy.types.Material | None,
    white_material: bpy.types.Material,
    cache: dict[str, bpy.types.Material],
    preserve_optical: bool,
) -> bpy.types.Material:
    if preserve_optical and material_is_optical(source):
        key = source.name if source else "__none__"
        if key not in cache and source is not None:
            cache[key] = make_colorless_optical_material(source)
        return cache.get(key, white_material)
    return white_material


def apply_white_diffuse_material_override(
    material: bpy.types.Material,
    preserve_optical: bool = False,
) -> tuple[dict[str, list[bpy.types.Material]], dict[str, bpy.types.Material]]:
    snapshot = object_material_snapshot()
    optical_cache: dict[str, bpy.types.Material] = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        original = snapshot.get(obj.name, [])
        source_materials = original or [None]
        obj.data.materials.clear()
        for source in source_materials:
            obj.data.materials.append(optical_override_material(source, material, optical_cache, preserve_optical))
    return snapshot, optical_cache


def light_color_snapshot() -> dict[str, tuple[float, float, float]]:
    return {
        obj.name: tuple(float(channel) for channel in obj.data.color)
        for obj in bpy.data.objects
        if obj.type == "LIGHT" and hasattr(obj.data, "color")
    }


def set_all_light_colors(color: tuple[float, float, float]) -> None:
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and hasattr(obj.data, "color"):
            obj.data.color = color


def restore_light_colors(snapshot: dict[str, tuple[float, float, float]]) -> None:
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.name in snapshot and hasattr(obj.data, "color"):
            obj.data.color = snapshot[obj.name]


def render_white_diffuse_component(scene_dir: Path, rel_base: str, config: dict, white_config: dict) -> dict:
    material = make_white_diffuse_override_material()
    preserve_optical = str(white_config.get("mode", "optical")).lower() == "optical"
    material_snapshot, optical_cache = apply_white_diffuse_material_override(material, preserve_optical=preserve_optical)
    bounce_snapshot = cycles_bounce_snapshot()
    light_snapshot = light_color_snapshot()
    try:
        if bool(white_config.get("force_white_light", True)):
            set_all_light_colors((1.0, 1.0, 1.0))
        if preserve_optical:
            set_optical_cycles_bounces(white_config)
        elif bool(white_config.get("direct_only", True)):
            set_direct_only_cycles_bounces()
        return render_component(scene_dir, rel_base, config)
    finally:
        restore_light_colors(light_snapshot)
        restore_cycles_bounces(bounce_snapshot)
        restore_object_materials(material_snapshot)
        for mat in optical_cache.values():
            if mat.users == 0:
                bpy.data.materials.remove(mat)


def render_material_property_map(scene_dir: Path, rel_path: str, property_name: str, fallback: float) -> str:
    snapshot = object_material_snapshot()
    world_snapshot = scene_world_snapshot()
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
        restore_scene_world(world_snapshot)
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
        run_blender_render(scene_dir / "pbr" / "passes.exr", write_still=False)
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


def canonical_position_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(av) - float(bv)) ** 2 for av, bv in zip(a, b)))


def respects_min_position_distance(candidate: list[float], positions: list[list[float]], min_distance: float) -> bool:
    if min_distance <= 0.0:
        return True
    return all(canonical_position_distance(candidate, position) >= min_distance for position in positions)


def jittered_cell_coordinate(rng: random.Random, axis_range: list[float], cell_index: int, side: int, jitter: float) -> float:
    lo, hi = float(axis_range[0]), float(axis_range[1])
    cell_size = (hi - lo) / float(side)
    center = lo + cell_size * (float(cell_index) + 0.5)
    half_jitter = abs(cell_size) * 0.5 * jitter
    if half_jitter <= 0.0:
        return center
    value = rng.uniform(center - half_jitter, center + half_jitter)
    return min(max(value, min(lo, hi)), max(lo, hi))


def sample_jittered_grid_positions(
    pr: dict,
    count: int,
    rng: random.Random,
    spatial: dict,
    existing_positions: list[list[float]],
) -> list[list[float]]:
    side = int(spatial.get("grid_resolution") or math.ceil(count ** (1.0 / 3.0)))
    side = max(1, side)
    while side**3 < count:
        side += 1
    jitter = min(max(float(spatial.get("jitter", 0.8)), 0.0), 1.0)
    min_distance = max(float(spatial.get("min_position_distance", 0.0)), 0.0)
    max_attempts = max(int(spatial.get("jitter_attempts", 32)), 1)
    ranges = [pr["x"], pr["y"], pr["z"]]
    cells = [(ix, iy, iz) for iz in range(side) for iy in range(side) for ix in range(side)]
    rng.shuffle(cells)

    sampled: list[list[float]] = []
    for cell in cells:
        if len(sampled) >= count:
            break
        candidate: list[float] | None = None
        fallback: list[float] | None = None
        for _attempt in range(max_attempts):
            coords = [
                jittered_cell_coordinate(rng, axis_range, cell_index, side, jitter)
                for cell_index, axis_range in zip(cell, ranges)
            ]
            fallback = coords
            if respects_min_position_distance(coords, existing_positions + sampled, min_distance):
                candidate = coords
                break
        sampled.append(candidate or fallback or [0.0, 0.0, 0.0])
    return sampled


def sample_uniform_random_positions(
    pr: dict,
    count: int,
    rng: random.Random,
    spatial: dict,
    existing_positions: list[list[float]],
) -> list[list[float]]:
    min_distance = max(float(spatial.get("min_position_distance", 0.0)), 0.0)
    max_attempts = max(int(spatial.get("random_attempts", 64)), 1)
    sampled: list[list[float]] = []
    for _ in range(count):
        candidate: list[float] | None = None
        fallback: list[float] | None = None
        for _attempt in range(max_attempts):
            coords = [rng.uniform(*pr["x"]), rng.uniform(*pr["y"]), rng.uniform(*pr["z"])]
            fallback = coords
            if respects_min_position_distance(coords, existing_positions + sampled, min_distance):
                candidate = coords
                break
        sampled.append(candidate or fallback or [0.0, 0.0, 0.0])
    return sampled


def sample_jittered_grid_candidates(
    pr: dict,
    count: int,
    rng: random.Random,
    spatial: dict,
    existing_positions: list[list[float]],
    source: str,
) -> list[dict]:
    side = int(spatial.get("grid_resolution") or math.ceil(count ** (1.0 / 3.0)))
    side = max(1, side)
    while side**3 < count:
        side += 1
    jitter = min(max(float(spatial.get("jitter", 0.8)), 0.0), 1.0)
    min_distance = max(float(spatial.get("min_position_distance", 0.0)), 0.0)
    max_attempts = max(int(spatial.get("jitter_attempts", 32)), 1)
    ranges = [pr["x"], pr["y"], pr["z"]]
    cells = [(ix, iy, iz) for iz in range(side) for iy in range(side) for ix in range(side)]
    rng.shuffle(cells)

    candidates: list[dict] = []
    for cell in cells:
        if len(candidates) >= count:
            break
        candidate: list[float] | None = None
        fallback: list[float] | None = None
        for attempt_in_cell in range(max_attempts):
            coords = [
                jittered_cell_coordinate(rng, axis_range, cell_index, side, jitter)
                for cell_index, axis_range in zip(cell, ranges)
            ]
            fallback = coords
            if respects_min_position_distance(coords, existing_positions + [row["canonical_position"] for row in candidates], min_distance):
                candidate = coords
                break
        candidates.append(
            {
                "canonical_position": candidate or fallback or [0.0, 0.0, 0.0],
                "candidate_source": source,
                "grid_cell": [int(cell[0]), int(cell[1]), int(cell[2])],
                "grid_resolution": side,
            }
        )
    return candidates


def sample_random_point_candidate(pr: dict, rng: random.Random, spatial: dict, existing_positions: list[list[float]], source: str) -> dict:
    position = sample_uniform_random_positions(pr, 1, rng, spatial, existing_positions)[0]
    return {
        "canonical_position": position,
        "candidate_source": source,
        "grid_cell": None,
        "grid_resolution": None,
    }


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
    if remaining > 0 and sampling == "grid_plus_random":
        side = max(1, int(spatial.get("grid_resolution") or 4))
        grid_count = min(remaining, int(spatial.get("grid_sample_count", side**3)))
        grid_spatial = dict(spatial)
        grid_spatial["grid_resolution"] = side
        positions.extend(sample_jittered_grid_positions(pr, grid_count, rng, grid_spatial, positions))
        remaining = count - len(positions)
        if remaining > 0:
            positions.extend(sample_uniform_random_positions(pr, remaining, rng, spatial, positions))
    if remaining > 0 and sampling == "jittered_grid":
        positions.extend(sample_jittered_grid_positions(pr, remaining, rng, spatial, positions))
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
        positions.extend(sample_uniform_random_positions(pr, count - len(positions), rng, spatial, positions))
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
    edges: list[tuple[tuple[int, int, int], tuple[int, int, int], str]] = []
    for ix in range(2):
        for iy in range(2):
            edges.append(((ix, iy, 0), (ix, iy, 1), "z"))
    for ix in range(2):
        for iz in range(2):
            edges.append(((ix, 0, iz), (ix, 1, iz), "y"))
    for iy in range(2):
        for iz in range(2):
            edges.append(((0, iy, iz), (1, iy, iz), "x"))

    canonical = config["canonical"]
    default_colors = {
        "x": (1.0, 0.35, 0.08),
        "y": (0.05, 0.8, 1.0),
        "z": (0.35, 1.0, 0.25),
    }
    color_cfg = canonical.get("debug_cube_edge_colors", {})

    def edge_color(axis: str) -> tuple[float, float, float]:
        value = color_cfg.get(axis, default_colors[axis]) if isinstance(color_cfg, dict) else default_colors[axis]
        return (float(value[0]), float(value[1]), float(value[2]))

    strength = float(canonical.get("debug_cube_edge_strength", 2.5))
    materials = {
        axis: make_emission_mat(f"TL_debug_light_volume_{axis}_mat", edge_color(axis), strength=strength)
        for axis in ("x", "y", "z")
    }
    world_scale = canonical_axis_scaled_world_scale(config)
    line_scale = float(canonical.get("debug_cube_line_scale", 2.0))
    bevel_depth = float(canonical.get("debug_line_bevel", 0.006)) * world_scale * line_scale
    debug_objects = [
        create_debug_curve_line(f"TL_Debug_LightVolume_{i:02d}", corners[a], corners[b], materials[axis], bevel_depth=bevel_depth)
        for i, (a, b, axis) in enumerate(edges)
    ]
    if bool(canonical.get("debug_cube_corner_markers", True)):
        corner_radius = float(canonical.get("debug_cube_corner_radius", canonical.get("debug_marker_radius", 0.02))) * world_scale
        corner_material = make_emission_mat("TL_debug_light_volume_corner_mat", (1.0, 1.0, 1.0), strength=strength)
        for i, key in enumerate(sorted(corners)):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=corner_radius, location=corners[key])
            marker = bpy.context.object
            marker.name = f"TL_Debug_LightVolumeCorner_{i:02d}"
            marker.data.materials.append(corner_material)
            debug_objects.append(marker)
    return debug_objects


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
    debug_objects = add_debug_light_volume_bounds(config, camera, center)
    z_layers = debug_light_z_layers(positions)
    marker_radius = float(config["canonical"].get("debug_marker_radius", 0.02)) * canonical_axis_scaled_world_scale(config)
    for i, p_can in enumerate(positions):
        p_world = canonical_to_world(p_can, camera, config, center)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=marker_radius, location=p_world)
        marker = bpy.context.object
        marker.name = f"TL_Debug_LightPos_{i:03d}"
        marker.data.materials.append(debug_light_material(z_layers.get(i, 0)))
        debug_objects.append(marker)
    rel_path = f"../preview/{scene_dir.name}_light_positions.png"
    try:
        render_png(scene_dir / rel_path)
    finally:
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


def render_global_diffuse_components(
    scene_dir: Path,
    config: dict,
    camera: bpy.types.Object,
    center: Vector,
    ambient_source_meta: dict,
    ambient_output: dict,
) -> dict | None:
    diffuse = global_diffuse_config(config)
    if not bool(diffuse.get("enabled", False)):
        return None

    values = control_values(diffuse, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ambient_color = tuple(float(c) for c in diffuse.get("constant_ambient_color", [0.72, 0.72, 0.72])[:3])
    ambient_strength = float(diffuse.get("constant_ambient_strength", 0.35))
    p_can = diffuse.get("area_light_canonical_position", [-0.45, -0.65, 1.15])
    p_world = canonical_to_world(p_can, camera, config, center)
    canonical_energy = float(diffuse.get("area_light_energy", 650.0))
    world_scale = canonical_world_scale(config)
    world_energy = canonical_energy * world_scale * world_scale if bool(diffuse.get("energy_scale_with_scene", True)) else canonical_energy
    variants = []

    remove_all_lights()
    set_constant_world(ambient_color, ambient_strength)
    diffuse_ambient_output = render_component(scene_dir, "global_diffuse/ambient_constant", config)

    remove_all_lights()
    set_black_world()
    with progress_bar(values, total=len(values), desc=f"{scene_dir.name} global diffuse", unit="dg") as pbar:
        for i, dg in enumerate(pbar):
            pbar.set_postfix(dg=f"{float(dg):.2f}")
            spread_degrees = spread_degrees_for_diffuse_level(float(dg), diffuse)
            area_size, distance = area_size_for_spread(p_world, center, spread_degrees)
            light = create_area_light(
                f"TL_GlobalDiffuse_Area_{i:03d}",
                p_world,
                center,
                world_energy,
                area_size,
            )
            output = render_component(scene_dir, f"global_diffuse/spread_{i:03d}", config)
            bpy.data.objects.remove(light, do_unlink=True)
            variant = {
                "id": i,
                "dg": float(dg),
                "normalized_diffuse": float(dg),
                "spread_degrees": float(spread_degrees),
                "area_size": float(area_size),
                "distance": float(distance),
                "complete_target": False,
            }
            variant.update(component_meta(output))
            variants.append(variant)

    set_ambient_source_from_meta(ambient_source_meta, config)
    remove_all_lights()
    return {
        "ambient_render": diffuse_ambient_output["primary"],
        "ambient_output": component_meta(diffuse_ambient_output),
        "ambient_source": {
            "type": "constant",
            "color": list(ambient_color),
            "strength": ambient_strength,
        },
        "base_variant_id": 0,
        "implementation": str(diffuse.get("implementation", "dominant_area_light_spread")),
        "ambient_source_type": "constant",
        "complete_target_variants": False,
        "light": {
            "type": "area",
            "canonical_position": [float(p_can[0]), float(p_can[1]), float(p_can[2])],
            "world_position": vec_to_list(p_world),
            "target": vec_to_list(center),
            "canonical_energy": canonical_energy,
            "world_energy": float(world_energy),
            "color": [1.0, 1.0, 1.0],
        },
        "spread_degrees_range": [float(v) for v in diffuse.get("spread_degrees_range", [6.0, 70.0])],
        "intensity_range": diffuse.get("intensity_range", [0.85, 1.15]),
        "ambient_scale_range": diffuse.get("ambient_scale_range", [0.85, 1.15]),
        "values": [float(value) for value in values],
        "variants": variants,
    }


def render_spatial_components(scene_dir: Path, config: dict, rng: random.Random, camera: bpy.types.Object, center: Vector) -> dict:
    white_only = bool(config.get("_pbr_white_shading_only", False))
    ambient_source = str(config.get("_ambient_source", "hdri")).lower()
    point_light_mode = str(config.get("_point_light_mode", "component")).lower()
    render_point_light_targets = point_light_mode in {"target", "additive", "add-to-scene", "scene-plus-light"}
    if ambient_source == "scene":
        source = scene_ambient_source_meta()
    else:
        ambient = config["ambient"]
        hdri_path, hdri_mode = choose_hdri_path(config, rng)
        hdri_strength = rng.uniform(*ambient.get("hdri_strength_range", [0.8, 1.2]))
        hdri_rotation = rng.random() * 2.0 * math.pi if ambient.get("hdri_rotation_random", True) else 0.0
        source = set_hdri_world(hdri_path, hdri_strength, hdri_rotation, ambient.get("fallback_color", [0.78, 0.78, 0.78]))
        source["hdri_mode"] = hdri_mode
        remove_all_lights()
    ambient_output = None
    ambient_render = None
    ambient_png = None
    global_diffuse_meta = None
    if not white_only:
        ambient_output = render_component(scene_dir, "spatial/ambient", config)
        ambient_render = ambient_output["primary"]
        ambient_png = f"../preview/{scene_dir.name}_ambient.png"
        write_component_preview_png(scene_dir, ambient_output, ambient_png, config)
        global_diffuse_meta = render_global_diffuse_components(scene_dir, config, camera, center, source, ambient_output)
    spatial = config["spatial"]
    valid_filter = point_light_valid_filter_config(spatial)
    target_point_count = int(spatial.get("positions_per_scene", 64))
    candidate_attempts: list[dict] = []
    final_positions: list[list[float]] = []
    initial_candidates: list[dict] = []
    if valid_filter["enabled"]:
        pr = config["canonical"]["position_range"]
        initial_spatial = dict(spatial)
        initial_spatial["grid_resolution"] = int(valid_filter["grid_resolution"])
        initial_candidates = sample_jittered_grid_candidates(
            pr,
            min(target_point_count, int(valid_filter["grid_resolution"]) ** 3),
            rng,
            initial_spatial,
            [],
            "grid_initial",
        )
        positions = [row["canonical_position"] for row in initial_candidates]
    else:
        positions = sample_spatial_positions(config, rng)
    light_position_preview = None
    pbr_maps = render_pbr_maps(scene_dir, config) if config.get("_render_pbr", False) and not white_only else None
    white_shading_config = config.get("_pbr_white_shading", pbr_white_shading_config(config))
    white_shading_enabled = bool(config.get("_render_pbr_white_shading", False))
    white_meta_key = white_shading_meta_key(white_shading_config)

    lights_meta = []
    include_hdri_in_point_lights = bool(spatial.get("include_hdri_in_point_lights", False))
    include_ambient_in_point_lights = False if white_only else render_point_light_targets or include_hdri_in_point_lights
    if not render_point_light_targets:
        if ambient_source == "scene" and not include_ambient_in_point_lights:
            remove_all_lights()
        if not include_hdri_in_point_lights:
            set_black_world()
    if white_only:
        remove_all_lights()
        set_black_world()
    receiver_bounds = config["_runtime"].get("receiver_bounds")
    receiver_materials = config["_runtime"].get("receiver_materials", [])
    invalid_reference_source: str | None = "spatial/ambient" if include_ambient_in_point_lights else None
    invalid_white_reference_source: str | None = None
    transform_meta = canonical_transform_meta(config, camera, center)
    world_scale = float(transform_meta.get("light_world_scale", transform_meta["scale"]))
    filter_receiver_bounds = bool(spatial.get("receiver_bounds_filter", False))
    per_light_diffuse = per_light_diffuse_config(spatial)
    per_light_diffuse_enabled = bool(per_light_diffuse.get("enabled", False)) and not white_only
    per_light_diffuse_values = control_values(per_light_diffuse, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    render_light_color = [1.0, 1.0, 1.0]

    def render_diffuse_variants_for_light(
        light_index: int,
        p_world: Vector,
        light_settings: dict,
        rel_base: str,
        valid: bool,
        invalid_source: str | None,
    ) -> list[dict]:
        diffuse_variants = []
        if not per_light_diffuse_enabled:
            return diffuse_variants
        for variant_index, d_value in enumerate(per_light_diffuse_values):
            variant_radius = radius_for_diffuse_level(d_value, per_light_diffuse)
            variant_world_radius = variant_radius * world_scale
            variant_base = f"{rel_base}/d_{variant_index:03d}"
            variant_copied_from = None
            if valid:
                light = create_point_light(
                    f"TL_Point_{light_index:03d}_D_{variant_index:03d}",
                    p_world,
                    float(light_settings["world_energy"]),
                    variant_world_radius,
                    render_light_color,
                )
                try:
                    variant_output = render_component(scene_dir, variant_base, config)
                finally:
                    bpy.data.objects.remove(light, do_unlink=True)
            else:
                variant_output = copy_component(scene_dir, invalid_source or rel_base, variant_base, config)
                variant_copied_from = f"{invalid_source or rel_base}.{primary_component_format(config)}"
            variant_meta = {
                "id": variant_index,
                "d": float(d_value),
                "normalized_diffuse": float(d_value),
                "canonical_radius": float(variant_radius),
                "world_radius": float(variant_world_radius),
                "copied_from": variant_copied_from,
            }
            variant_meta.update(component_meta(variant_output))
            diffuse_variants.append(variant_meta)
        return diffuse_variants

    if valid_filter["enabled"] and white_only:
        progress_write(f"[Relighting] point-light valid_filter is disabled for {scene_dir.name}: white-only legacy mode")
        valid_filter["enabled"] = False

    if valid_filter["enabled"]:
        pr = config["canonical"]["position_range"]
        max_attempts = max(target_point_count, len(initial_candidates)) + max(int(valid_filter["max_refill_attempts"]), 0)
        keep_attempt_renders = bool(valid_filter["keep_attempt_renders"])
        with progress_bar(range(max_attempts), total=max_attempts, desc=f"{scene_dir.name} point candidates", unit="attempt") as pbar:
            for attempt_index in pbar:
                if len(lights_meta) >= target_point_count:
                    break
                pbar.set_postfix(valid=f"{len(lights_meta)}/{target_point_count}", attempt=f"{attempt_index:04d}")
                if attempt_index < len(initial_candidates):
                    candidate = initial_candidates[attempt_index]
                else:
                    candidate = sample_random_point_candidate(pr, rng, spatial, final_positions, "random_refill")
                p_can = [float(value) for value in candidate["canonical_position"]]
                p_world = canonical_to_world(p_can, camera, config, center)
                attempt_base = f"spatial/point_light_attempts/attempt_{attempt_index:04d}"
                light_settings = sample_spatial_light_settings(spatial, rng, world_scale)
                canonical_radius = float(light_settings["canonical_radius"])
                world_radius = float(light_settings["world_radius"])
                if filter_receiver_bounds:
                    geometry_valid, skip_reason = point_inside_receiver_bounds(p_world, receiver_bounds, world_radius)
                else:
                    geometry_valid, skip_reason = True, None

                light_output = None
                validation_stats = {"valid": False, "skip_reason": skip_reason or "geometry_rejected"}
                if geometry_valid:
                    light = create_point_light(
                        f"TL_Point_Attempt_{attempt_index:04d}",
                        p_world,
                        float(light_settings["world_energy"]),
                        world_radius,
                        render_light_color,
                    )
                    try:
                        light_output = render_component(scene_dir, attempt_base, config)
                    finally:
                        bpy.data.objects.remove(light, do_unlink=True)
                    accepted, validation_stats = validate_point_light_component(scene_dir, light_output, valid_filter)
                    skip_reason = None if accepted else str(validation_stats.get("skip_reason", "invalid_point_light"))
                else:
                    accepted = False

                attempt_meta = {
                    "attempt_index": attempt_index,
                    "candidate_source": candidate.get("candidate_source"),
                    "grid_cell": candidate.get("grid_cell"),
                    "grid_resolution": candidate.get("grid_resolution"),
                    "canonical_position": p_can,
                    "world_position": vec_to_list(p_world),
                    "geometry_valid": geometry_valid,
                    "valid": accepted,
                    "skip_reason": skip_reason,
                    "validation": validation_stats,
                    "accepted_light_id": None,
                }
                if keep_attempt_renders and light_output:
                    attempt_meta["attempt_output"] = component_meta(light_output)

                if accepted:
                    light_index = len(lights_meta)
                    rel_base = f"spatial/point_lights/light_{light_index:03d}"
                    final_output = copy_component(scene_dir, attempt_base, rel_base, config)
                    if not keep_attempt_renders:
                        remove_component_files(scene_dir, attempt_base, config)
                    diffuse_variants = render_diffuse_variants_for_light(light_index, p_world, light_settings, rel_base, True, None)
                    light_meta = {
                        "id": light_index,
                        "attempt_index": attempt_index,
                        "candidate_source": candidate.get("candidate_source"),
                        "grid_cell": candidate.get("grid_cell"),
                        "grid_resolution": candidate.get("grid_resolution"),
                        "canonical_position": p_can,
                        "world_position": vec_to_list(p_world),
                        "valid": True,
                        "skip_reason": None,
                        "copied_from": f"{attempt_base}.{primary_component_format(config)}",
                        "canonical_energy": float(light_settings["canonical_energy"]),
                        "world_energy": float(light_settings["world_energy"]),
                        "energy": float(light_settings["world_energy"]),
                        "component_color": [float(c) for c in render_light_color],
                        "render_color": [float(c) for c in render_light_color],
                        "component_color_semantics": "basis_render_color",
                        "canonical_radius": canonical_radius,
                        "world_radius": world_radius,
                        "validation": validation_stats,
                    }
                    light_meta.update(component_meta(final_output))
                    if diffuse_variants:
                        light_meta["diffuse_variants"] = diffuse_variants
                    attempt_meta["accepted_light_id"] = light_index
                    lights_meta.append(light_meta)
                    final_positions.append(p_can)
                elif light_output and not keep_attempt_renders:
                    remove_component_files(scene_dir, attempt_base, config)
                candidate_attempts.append(attempt_meta)

        if len(lights_meta) < target_point_count:
            message = (
                f"{scene_dir.name}: valid point-light filter accepted {len(lights_meta)}/"
                f"{target_point_count} after {len(candidate_attempts)} attempts"
            )
            if bool(valid_filter["require_target_count"]):
                raise RuntimeError(message)
            progress_write(f"[Relighting] WARNING {message}")
    else:
        with progress_bar(positions, total=len(positions), desc=f"{scene_dir.name} point lights", unit="light") as pbar:
            for light_index, p_can in enumerate(pbar):
                pbar.set_postfix(light=f"{light_index:03d}")
                p_world = canonical_to_world(p_can, camera, config, center)
                rel_base = f"spatial/point_lights/light_{light_index:03d}"
                light_settings = sample_spatial_light_settings(spatial, rng, world_scale)
                canonical_radius = float(light_settings["canonical_radius"])
                world_radius = float(light_settings["world_radius"])
                if filter_receiver_bounds:
                    valid, skip_reason = point_inside_receiver_bounds(p_world, receiver_bounds, world_radius)
                else:
                    valid, skip_reason = True, None
                copied_from = None
                white_output = None
                white_copied_from = None
                if valid:
                    light_output = None
                    light = create_point_light(
                        f"TL_Point_{light_index:03d}",
                        p_world,
                        float(light_settings["world_energy"]),
                        world_radius,
                        render_light_color,
                    )
                    try:
                        if not white_only:
                            light_output = render_component(scene_dir, rel_base, config)
                        if white_shading_enabled:
                            white_base = white_shading_light_base(light_index, white_shading_config)
                            white_output = render_white_diffuse_component(scene_dir, white_base, config, white_shading_config)
                    finally:
                        bpy.data.objects.remove(light, do_unlink=True)
                else:
                    light_output = None
                    if not white_only:
                        if invalid_reference_source is None:
                            light_output = render_component(scene_dir, rel_base, config)
                            invalid_reference_source = rel_base
                        else:
                            light_output = copy_component(scene_dir, invalid_reference_source, rel_base, config)
                            copied_from = f"{invalid_reference_source}.{primary_component_format(config)}"
                    if white_shading_enabled:
                        white_base = white_shading_light_base(light_index, white_shading_config)
                        if invalid_white_reference_source is None:
                            white_output = render_white_diffuse_component(scene_dir, white_base, config, white_shading_config)
                            invalid_white_reference_source = white_base
                        else:
                            white_output = copy_component(scene_dir, invalid_white_reference_source, white_base, config)
                            white_copied_from = f"{invalid_white_reference_source}.{primary_component_format(config)}"
                diffuse_variants = render_diffuse_variants_for_light(
                    light_index,
                    p_world,
                    light_settings,
                    rel_base,
                    valid,
                    invalid_reference_source,
                )
                light_meta = {
                    "id": light_index,
                    "canonical_position": p_can,
                    "world_position": vec_to_list(p_world),
                    "valid": valid,
                    "skip_reason": skip_reason,
                    "copied_from": copied_from,
                    "canonical_energy": float(light_settings["canonical_energy"]),
                    "world_energy": float(light_settings["world_energy"]),
                    "energy": float(light_settings["world_energy"]),
                    "component_color": [float(c) for c in render_light_color],
                    "render_color": [float(c) for c in render_light_color],
                    "component_color_semantics": "basis_render_color",
                    "canonical_radius": canonical_radius,
                    "world_radius": world_radius,
                }
                if light_output:
                    light_meta.update(component_meta(light_output))
                if white_output:
                    light_meta[white_meta_key] = component_meta(white_output)
                    if white_only:
                        light_meta["render"] = white_output["primary"]
                    if white_copied_from:
                        light_meta[white_meta_key]["copied_from"] = white_copied_from
                if diffuse_variants:
                    light_meta["diffuse_variants"] = diffuse_variants
                lights_meta.append(light_meta)
                final_positions.append(p_can)

    if config.get("_light_preview", False):
        light_position_preview = render_light_position_preview(
            scene_dir,
            final_positions or positions,
            config,
            camera,
            center,
        )
    return {
        "ambient_render": ambient_render,
        "ambient_output": component_meta(ambient_output) if ambient_output else None,
        "ambient_png": ambient_png,
        "light_position_preview": light_position_preview,
        "ambient_source": source,
        "global_diffuse": global_diffuse_meta,
        "pbr_maps": pbr_maps,
        "pbr_white_shading": white_shading_config if white_shading_enabled else None,
        "pbr_white_shading_only": white_only,
        "light_volume_center": vec_to_list(center),
        "light_volume_center_source": config.get("_runtime", {}).get("light_volume_center_source", "bbox_center"),
        "light_volume_adjustment": config.get("_runtime", {}).get("light_volume_adjustment"),
        "canonical_transform": transform_meta,
        "receiver_bounds": receiver_bounds_to_meta(receiver_bounds) if receiver_bounds else None,
        "receiver_bounds_filter": filter_receiver_bounds,
        "receiver_materials": receiver_materials,
        "positions_per_scene": int(spatial.get("positions_per_scene", 64)),
        "position_sampling": spatial.get("sampling", "stratified_random"),
        "grid_resolution": spatial.get("grid_resolution"),
        "grid_sample_count": spatial.get("grid_sample_count"),
        "random_extra_count": spatial.get("random_extra_count"),
        "jitter": spatial.get("jitter"),
        "min_position_distance": spatial.get("min_position_distance"),
        "valid_point_light_count": sum(1 for light in lights_meta if light["valid"]),
        "invalid_point_light_count": sum(1 for light in lights_meta if not light["valid"]),
        "valid_filter": {
            "enabled": bool(valid_filter.get("enabled", False)),
            "grid_resolution": int(valid_filter.get("grid_resolution", spatial.get("grid_resolution", 4))),
            "p99_luminance_threshold": float(valid_filter.get("p99_luminance_threshold", 0.01)),
            "nonzero_pixel_ratio_threshold": float(valid_filter.get("nonzero_pixel_ratio_threshold", 0.001)),
            "nonzero_luminance_threshold": float(valid_filter.get("nonzero_luminance_threshold", 1.0e-4)),
            "max_refill_attempts": int(valid_filter.get("max_refill_attempts", 256)),
            "require_target_count": bool(valid_filter.get("require_target_count", True)),
            "attempt_count": len(candidate_attempts) if candidate_attempts else len(lights_meta),
            "initial_grid_candidate_count": len(initial_candidates),
            "random_refill_attempt_count": sum(1 for row in candidate_attempts if row.get("candidate_source") == "random_refill"),
            "accepted_count": len(lights_meta),
        },
        "candidate_attempts": candidate_attempts if candidate_attempts else None,
        "point_light_mode": point_light_mode,
        "point_light_basis_color": [1.0, 1.0, 1.0],
        "point_light_output_semantics": (
            "ambient_plus_white_point_light_target" if render_point_light_targets else "isolated_white_point_light_component"
        ),
        "include_hdri_in_point_lights": include_hdri_in_point_lights,
        "include_ambient_source_in_point_lights": include_ambient_in_point_lights,
        "color_range": spatial.get("color_range"),
        "intensity_range": spatial.get("intensity_range"),
        "radius_range": spatial.get("radius_range"),
        "per_light_diffuse": {
            "enabled": per_light_diffuse_enabled,
            "values": [float(value) for value in per_light_diffuse_values],
            "radius_range": per_light_diffuse.get("radius_range"),
            "radius_mapping": per_light_diffuse.get("radius_mapping", "linear"),
        },
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
            area_size, distance = area_size_for_spread(p_world, center, spread_deg)
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
    white_only = bool(config.get("_pbr_white_shading_only", False))
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
    set_canonical_runtime_transform(config, bbox_min, bbox_max)
    center = (bbox_min + bbox_max) * 0.5
    camera, camera_meta = create_camera(config, rng, center)
    look_target = Vector(camera_meta["look_at"])
    fit_default = not uses_canonical_camera_rig(config)
    if bool(config["camera"].get("fit_to_object", fit_default)):
        framing_meta = fit_camera_to_objects(camera, subject_objects, look_target)
        camera_meta["location"] = framing_meta["location"]
        camera_meta["distance"] = framing_meta["distance"]
        camera_meta["distance_over_scale"] = float(camera_meta["distance"] / max(canonical_world_scale(config), 1e-6))
    else:
        framing_meta = {
            "adjusted": False,
            "fit_to_object": False,
            "location": vec_to_list(Vector(camera.location)),
            "distance": float((Vector(camera.location) - look_target).length),
            "margin": None,
        }
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
        ambient_preview = f"../preview/{scene_dir.name}_ambient.png"
        render_png(scene_dir / ambient_preview)
        light_position_preview = None
        if config.get("_light_preview", False):
            positions = sample_spatial_positions(config, rng)
            light_position_preview = render_light_position_preview(scene_dir, positions, config, camera, center)
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
                "orientation": config.get("_runtime", {}).get("object_orientation"),
                "import_adjustments": config.get("_runtime", {}).get("object_import_adjustments"),
            },
            "camera": camera_meta,
            "render": {
                "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
                "samples": int(config["render"].get("samples", 128)),
                "engine": bpy.context.scene.render.engine,
                "linear_rgb": True,
                "tone_mapping_applied": False,
                "light_transport": light_transport_meta(config),
            },
            "canonical": config["canonical"],
            "debug": {
                "preview_only": True,
                "ambient_source": source,
                "ambient_preview": ambient_preview,
                "light_position_preview": light_position_preview,
                "light_volume_center": vec_to_list(center),
                "light_volume_center_source": "bbox_center",
                "canonical_transform": canonical_transform_meta(config, camera, center),
                "receiver_materials": config["_runtime"].get("receiver_materials", []),
            },
        }
        write_json(scene_dir / "meta.json", meta)
        return meta

    mask_path = None if white_only else render_object_mask(scene_dir, subject_objects)

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
            "orientation": config.get("_runtime", {}).get("object_orientation"),
            "import_adjustments": config.get("_runtime", {}).get("object_import_adjustments"),
        },
        "camera": camera_meta,
        "render": {
            "resolution": [bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y],
            "samples": int(config["render"].get("samples", 128)),
            "engine": bpy.context.scene.render.engine,
            "linear_rgb": True,
            "tone_mapping_applied": False,
            "light_transport": light_transport_meta(config),
        },
        "canonical": config["canonical"],
    }
    if mask_path:
        meta["masks"] = {"object": mask_path}

    if (white_only or only in ("all", "spatial")) and config["spatial"].get("enabled", True):
        meta["spatial"] = render_spatial_components(scene_dir, config, rng, camera, center)
    if not white_only and only in ("all", "diffuse") and config["diffuse"].get("enabled", True):
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


def write_failed_object_scene(output_root: Path, scene_index: int, render_only: str, exc: Exception) -> dict:
    scene_id = f"scene_{scene_index:06d}"
    record = {
        "schema": "tokenlight_failed_object_scene_v1",
        "scene_id": scene_id,
        "scene_type": "object_centric",
        "scene_index": int(scene_index),
        "render_only": render_only,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_root / "failed_scenes" / f"{scene_id}_error.json", record)
    return record


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
    if args.positions_per_scene is not None:
        config["spatial"]["positions_per_scene"] = args.positions_per_scene
    if args.global_diffuse is not None:
        config.setdefault("global_diffuse", {})["enabled"] = bool(args.global_diffuse)
    if args.per_light_diffuse is not None:
        config.setdefault("spatial", {}).setdefault("per_light_diffuse", {})["enabled"] = bool(args.per_light_diffuse)
    if args.pbr_white_shading is not None:
        raw_white = config.setdefault("render", {}).get("pbr_white_shading")
        if isinstance(raw_white, dict):
            raw_white["enabled"] = bool(args.pbr_white_shading)
        else:
            config["render"]["pbr_white_shading"] = bool(args.pbr_white_shading)
    if args.pbr_white_shading_only:
        raw_white = config.setdefault("render", {}).get("pbr_white_shading")
        if isinstance(raw_white, dict):
            raw_white["enabled"] = True
            raw_white["mode"] = "optical"
            raw_white["direct_only"] = False
            raw_white.setdefault("output_root", "pbr/white_shading_optical")
        else:
            config["render"]["pbr_white_shading"] = {
                "enabled": True,
                "mode": "optical",
                "direct_only": False,
                "output_root": "pbr/white_shading_optical",
            }
    config["_component_format"] = str(config["render"].get("component_format", "exr")).lower()
    config["_ambient_source"] = (
        args.ambient_source
        or str(config.get("spatial", {}).get("ambient_source", config.get("ambient_source", "hdri"))).lower()
    )
    config["_point_light_mode"] = (
        args.point_light_mode
        or str(config.get("spatial", {}).get("point_light_mode", config.get("point_light_mode", "component"))).lower()
    )
    config["_hdri_mode"] = args.hdri_mode or str(config.get("ambient", {}).get("hdri_mode", "on")).lower()
    config["_pbr_white_shading_only"] = bool(args.pbr_white_shading_only)
    config["_debug_preview_only"] = bool(args.debug and not args.pbr_white_shading_only)
    config["_light_preview"] = bool(args.light_preview or args.debug_light_preview or args.debug)
    config["_render_pbr"] = bool(args.pbr and not args.pbr_white_shading_only)
    config["_soft_light_transport"] = bool(args.soft_light_transport)
    config["_pbr_white_shading"] = pbr_white_shading_config(config)
    config["_render_pbr_white_shading"] = bool(
        (config["_render_pbr"] or config["_pbr_white_shading_only"]) and config["_pbr_white_shading"]["enabled"]
    )

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
    failed_metas = []
    render_only = "spatial" if args.pbr_white_shading_only else args.only
    if render_only in ("all", "spatial", "diffuse"):
        scene_indices = range(args.start_index, args.start_index + scene_count)
        with progress_bar(scene_indices, total=scene_count, desc="Object scenes", unit="scene") as pbar:
            for i in pbar:
                pbar.set_postfix(scene=f"{i:06d}")
                progress_write(f"[Relighting] Rendering object scene {i}")
                try:
                    metas.append(render_object_scene(i, config, root, render_only))
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    failed = write_failed_object_scene(output_root, i, render_only, exc)
                    failed_metas.append(failed)
                    progress_write(
                        f"[Relighting] WARNING skipping {failed['scene_id']} after "
                        f"{failed['error_type']}: {failed['error']}"
                    )

    if not args.debug and not args.pbr_white_shading_only and args.only in ("all", "fixtures") and config["fixtures"].get("enabled", True):
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
            "global_diffuse_count": len(control_values(global_diffuse_config(config))) if global_diffuse_config(config).get("enabled") else 0,
            "per_light_diffuse_count": len(control_values(per_light_diffuse_config(config["spatial"]))) if per_light_diffuse_config(config["spatial"]).get("enabled") else 0,
            "fov_degrees": config["camera"].get("fov_degrees", 39.6),
            "linear_rgb_components": True,
            "component_format": config["_component_format"],
            "hdri_mode": config["_hdri_mode"],
        },
        "scene_count_written": len(metas),
        "scene_count_failed": len(failed_metas),
        "scenes": [{"scene_id": m["scene_id"], "scene_type": m["scene_type"], "meta": f"scenes/{m['scene_id']}/meta.json"} for m in metas],
        "failed_scenes": [
            {
                "scene_id": m["scene_id"],
                "scene_type": m["scene_type"],
                "error_type": m["error_type"],
                "error": m["error"],
                "record": f"failed_scenes/{m['scene_id']}_error.json",
            }
            for m in failed_metas
        ],
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    progress_write(f"[Relighting] Wrote manifest: {output_root / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
