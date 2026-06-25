from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("Run this script with Blender Python.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_object_relighting as relight  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render neutral material single-light shading previews.")
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json")
    parser.add_argument(
        "--scenes-root",
        default="outputs/objaverse_ratio3p5_cube1p6_full_scene4000_4039_640/scenes",
        help="Existing scene component root containing ambient/depth/normal/point-light EXRs and meta.json.",
    )
    parser.add_argument(
        "--preview-output-dir",
        default="outputs/neutral_material_shading_previews_ambient_point_neutral_depth_normal",
    )
    parser.add_argument("--scenes", nargs="+", type=int, required=True)
    parser.add_argument("--variants-per-scene", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--neutral-color", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--min-light-mean", type=float, default=0.01)
    parser.add_argument("--min-light-max", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args(argv)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(path: Path, samples: int | None) -> dict:
    config = relight.load_json(path)
    if samples is not None:
        config.setdefault("render", {})["samples"] = int(samples)
    config.setdefault("render", {})["component_format"] = "exr"
    config["_component_format"] = "exr"
    config["_ambient_source"] = "hdri"
    config["_point_light_mode"] = "component"
    config["_hdri_mode"] = str(config.get("ambient", {}).get("hdri_mode", "on")).lower()
    config["_pbr_white_shading_only"] = False
    config["_debug_preview_only"] = False
    config["_light_preview"] = False
    config["_render_pbr"] = False
    config["_pbr_white_shading"] = relight.pbr_white_shading_config(config)
    config["_render_pbr_white_shading"] = False

    object_manifest = relight.resolve_path(REPO_ROOT, config.get("object_manifest"))
    hdri_manifest = relight.resolve_path(REPO_ROOT, config.get("hdri_manifest"))
    receiver_texture_manifest = relight.resolve_path(
        REPO_ROOT,
        config.get("receiver_texture_manifest") or config.get("layout", {}).get("receiver_texture_manifest"),
    )
    fixture_manifest = relight.resolve_path(REPO_ROOT, config.get("fixture_scene_manifest"))
    config["_runtime"] = {
        "objects": relight.load_path_lines(object_manifest, REPO_ROOT) if object_manifest else [],
        "hdris": relight.load_path_lines(hdri_manifest, REPO_ROOT) if hdri_manifest else [],
        "receiver_textures": relight.load_receiver_texture_manifest(receiver_texture_manifest, REPO_ROOT)
        if receiver_texture_manifest
        else [],
        "fixture_scenes": relight.normalize_fixture_rows(relight.load_jsonl(fixture_manifest), REPO_ROOT)
        if fixture_manifest
        else [],
    }
    return config


def socket_summary(node) -> dict:
    result = {}
    for key in [
        "Base Color",
        "Alpha",
        "Roughness",
        "Metallic",
        "Specular IOR Level",
        "Specular Tint",
        "Transmission Weight",
        "IOR",
        "Emission Strength",
    ]:
        if key not in node.inputs:
            continue
        value = node.inputs[key].default_value
        try:
            value = [float(v) for v in value]
        except TypeError:
            value = float(value)
        result[key] = value
    return result


def material_diagnostic(mat: bpy.types.Material | None) -> dict:
    if mat is None:
        return {"name": None}
    row = {
        "name": mat.name,
        "use_nodes": bool(mat.use_nodes),
        "diffuse_color": [float(v) for v in getattr(mat, "diffuse_color", (1.0, 1.0, 1.0, 1.0))],
        "blend_method": str(getattr(mat, "blend_method", "")),
        "is_optical": bool(relight.material_is_optical(mat)),
        "has_optical_name": bool(relight.material_has_optical_name(mat)),
        "has_alpha_cutout_name": bool(relight.material_has_alpha_cutout_name(mat)),
        "node_types": [],
        "principled": [],
        "ambiguous_flags": [],
    }
    if not mat.use_nodes or mat.node_tree is None:
        row["ambiguous_flags"].append("no_node_tree")
        return row

    for node in mat.node_tree.nodes:
        row["node_types"].append(node.type)
        if node.type == "BSDF_PRINCIPLED":
            row["principled"].append(socket_summary(node))
        elif node.type == "GROUP":
            row["ambiguous_flags"].append("node_group")
        elif node.type == "EMISSION":
            row["ambiguous_flags"].append("emission_node")
        elif node.type in {"BSDF_TRANSPARENT", "BSDF_GLASS", "BSDF_REFRACTION", "BSDF_TRANSLUCENT"}:
            row["ambiguous_flags"].append(f"optical_node:{node.type}")

    if str(getattr(mat, "blend_method", "")).upper() not in {"OPAQUE", "HASHED"} and not row["is_optical"]:
        row["ambiguous_flags"].append("transparent_blend_but_not_optical")
    return row


def remove_input_links(tree, socket) -> None:
    for link in list(socket.links):
        tree.links.remove(link)


def set_color_socket(tree, node, names: list[str], color: tuple[float, float, float, float], preserve_alpha: bool = True) -> None:
    for name in names:
        if name not in node.inputs:
            continue
        socket = node.inputs[name]
        remove_input_links(tree, socket)
        value = socket.default_value
        if hasattr(value, "__len__"):
            alpha = float(value[3]) if preserve_alpha and len(value) > 3 else float(color[3])
            socket.default_value = (float(color[0]), float(color[1]), float(color[2]), alpha)
        else:
            socket.default_value = float(color[0])


def set_scalar_socket(tree, node, names: list[str], value: float) -> None:
    for name in names:
        if name not in node.inputs:
            continue
        socket = node.inputs[name]
        remove_input_links(tree, socket)
        socket.default_value = float(value)


def neutralize_original_material(mat: bpy.types.Material, neutral: float) -> bpy.types.Material:
    safe_name = mat.name.replace("/", "_")
    clone = mat.copy()
    clone.name = f"TL_neutral_material_{safe_name}"
    diffuse = getattr(clone, "diffuse_color", (1.0, 1.0, 1.0, 1.0))
    alpha = float(diffuse[3]) if len(diffuse) > 3 else 1.0
    clone.diffuse_color = (neutral, neutral, neutral, alpha)
    if not clone.use_nodes or clone.node_tree is None:
        return clone

    tree = clone.node_tree
    neutral_rgba = (neutral, neutral, neutral, 1.0)
    white_rgba = (1.0, 1.0, 1.0, 1.0)
    black_rgba = (0.0, 0.0, 0.0, 1.0)
    optical_color_nodes = {"BSDF_GLASS", "BSDF_REFRACTION", "BSDF_TRANSPARENT", "BSDF_TRANSLUCENT"}
    neutral_color_nodes = {"BSDF_GLOSSY", "BSDF_SHEEN"}

    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            set_color_socket(tree, node, ["Base Color", "Subsurface Color"], neutral_rgba, preserve_alpha=True)
            set_color_socket(tree, node, ["Specular Tint", "Coat Tint", "Sheen Tint"], white_rgba, preserve_alpha=True)
            set_color_socket(tree, node, ["Emission Color"], black_rgba, preserve_alpha=False)
            set_scalar_socket(tree, node, ["Emission Strength"], 0.0)
        elif node.type in optical_color_nodes:
            set_color_socket(tree, node, ["Color"], white_rgba, preserve_alpha=True)
        elif node.type in neutral_color_nodes:
            set_color_socket(tree, node, ["Color"], white_rgba, preserve_alpha=True)
        elif node.type == "EMISSION":
            set_color_socket(tree, node, ["Color"], black_rgba, preserve_alpha=False)
            set_scalar_socket(tree, node, ["Strength"], 0.0)
        elif node.type in {"VOLUME_ABSORPTION", "VOLUME_SCATTER"}:
            set_color_socket(tree, node, ["Color"], white_rgba, preserve_alpha=True)
    return clone


def make_default_neutral_material(neutral: float) -> bpy.types.Material:
    mat = bpy.data.materials.get("TL_default_neutral_material")
    if mat is None:
        mat = relight.make_principled_mat("TL_default_neutral_material", (neutral, neutral, neutral), roughness=0.75)
    return mat


def apply_neutral_material_override(neutral: float) -> tuple[dict[str, list[bpy.types.Material]], dict[str, bpy.types.Material], list[dict]]:
    snapshot = relight.object_material_snapshot()
    cache: dict[str, bpy.types.Material] = {}
    diagnostics = []
    default_mat = make_default_neutral_material(neutral)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        original = snapshot.get(obj.name, [])
        obj.data.materials.clear()
        if not original:
            obj.data.materials.append(default_mat)
            diagnostics.append({"object": obj.name, "slot": 0, "source": material_diagnostic(None), "override": default_mat.name})
            continue
        for slot_index, source in enumerate(original):
            if source is None:
                obj.data.materials.append(default_mat)
                diagnostics.append({"object": obj.name, "slot": slot_index, "source": material_diagnostic(None), "override": default_mat.name})
                continue
            key = source.name
            if key not in cache:
                cache[key] = neutralize_original_material(source, neutral)
            obj.data.materials.append(cache[key])
            diagnostics.append(
                {
                    "object": obj.name,
                    "slot": slot_index,
                    "source": material_diagnostic(source),
                    "override": cache[key].name,
                }
            )
    return snapshot, cache, diagnostics


def set_neutral_cycles_bounces() -> dict[str, int] | None:
    snapshot = relight.cycles_bounce_snapshot()
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return snapshot
    settings = {
        "max_bounces": 8,
        "diffuse_bounces": 2,
        "glossy_bounces": 4,
        "transmission_bounces": 8,
        "transparent_max_bounces": 8,
        "volume_bounces": 0,
    }
    for name, value in settings.items():
        if hasattr(scene.cycles, name):
            setattr(scene.cycles, name, value)
    for name in ("caustics_reflective", "caustics_refractive"):
        if hasattr(scene.cycles, name):
            setattr(scene.cycles, name, True)
    return snapshot


def existing_meta(scene_dir: Path) -> dict:
    meta_path = scene_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def choose_points(scene_dir: Path, meta: dict, rng: random.Random, count: int, min_mean: float, min_max: float) -> list[dict]:
    point_lights = {int(row["id"]): row for row in meta["spatial"]["point_lights"]}
    candidates = []
    fallback = []
    for point_id, row in point_lights.items():
        point_path = scene_dir / row["render_exr"]
        if not point_path.exists():
            continue
        mean, maximum = point_light_stats(point_path)
        item = {"point_light_id": point_id, "point_light_mean": mean, "point_light_max": maximum}
        fallback.append(item)
        if mean > min_mean and maximum > min_max:
            candidates.append(item)
    pool = candidates if len(candidates) >= count else fallback
    if len(pool) < count:
        raise RuntimeError(f"Not enough point-light candidates in {scene_dir}: {len(pool)}")
    selected = rng.sample(pool, count)
    for variant, row in enumerate(selected):
        row["variant"] = variant
        row["sampled_color_rgb"] = [round(float(c), 6) for c in sample_color(rng)]
    return selected


def point_light_stats(path: Path) -> tuple[float, float]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        channels = int(image.channels)
        pixels = [0.0] * (width * height * channels)
        image.pixels.foreach_get(pixels)
        total = 0.0
        maximum = 0.0
        count = width * height
        for pixel_index in range(count):
            offset = pixel_index * channels
            lum = 0.0
            for channel in range(min(3, channels)):
                value = max(float(pixels[offset + channel]), 0.0)
                lum += value
                maximum = max(maximum, value)
            total += lum / 3.0
        return total / max(count, 1), maximum
    finally:
        bpy.data.images.remove(image)


def sample_color(rng: random.Random) -> list[float]:
    color = [rng.uniform(0.18, 1.0) for _ in range(3)]
    color[rng.randrange(3)] = rng.uniform(0.75, 1.0)
    return color


def recreate_scene(scene_index: int, config: dict, meta: dict) -> tuple[bpy.types.Object, Vector]:
    rng = random.Random(int(config["seed"]) + scene_index)
    relight.clear_scene()
    relight.setup_render_settings(config)
    objects = config["_runtime"]["objects"]
    primitives = config["object"].get("primitive_fallbacks", ["sphere"])
    asset = rng.choice(objects) if objects else None
    primitive = rng.choice(primitives)
    subject_objects = relight.import_asset_or_primitive(asset, primitive, rng, config)
    if meta.get("object", {}).get("path") and str(asset) != str(meta["object"]["path"]):
        print(f"[NeutralMaterial] WARNING scene {scene_index}: asset mismatch {asset} != {meta['object']['path']}")

    bbox_min, bbox_max = relight.mesh_bbox(subject_objects)
    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)
    center = (bbox_min + bbox_max) * 0.5
    camera, camera_meta = relight.create_camera(config, rng, center)
    fit_default = not relight.uses_canonical_camera_rig(config)
    if bool(config["camera"].get("fit_to_object", fit_default)):
        look_target = Vector(camera_meta["look_at"])
        relight.fit_camera_to_objects(camera, subject_objects, look_target)
    relight.create_receivers(config, rng, camera, center)
    relight.remove_all_lights()
    relight.set_black_world()
    return camera, center


def render_neutral_points(
    scene_index: int,
    scene_dir: Path,
    config: dict,
    selected: list[dict],
    neutral: float,
) -> dict:
    meta = existing_meta(scene_dir)
    recreate_scene(scene_index, config, meta)
    material_snapshot, material_cache, material_diagnostics = apply_neutral_material_override(neutral)
    bounce_snapshot = set_neutral_cycles_bounces()

    rendered = []
    point_rows = {int(row["id"]): row for row in meta["spatial"]["point_lights"]}
    try:
        for item in selected:
            point_id = int(item["point_light_id"])
            point_meta = point_rows[point_id]
            rel_base = f"pbr/neutral_material_shading/point_light_{point_id:03d}"
            out_path = scene_dir / f"{rel_base}.exr"
            if out_path.exists() and item.get("skip_existing"):
                print(f"[NeutralMaterial] exists {out_path}")
            else:
                light = relight.create_point_light(
                    f"TL_Neutral_Point_{point_id:03d}",
                    Vector(point_meta["world_position"]),
                    float(point_meta["world_energy"]),
                    float(point_meta["world_radius"]),
                    [1.0, 1.0, 1.0],
                )
                try:
                    relight.render_component(scene_dir, rel_base, config)
                finally:
                    bpy.data.objects.remove(light, do_unlink=True)
                print(f"[NeutralMaterial] rendered {out_path}")
            rendered.append({"point_light_id": point_id, "render": f"{rel_base}.exr"})
    finally:
        relight.restore_cycles_bounces(bounce_snapshot)
        relight.restore_object_materials(material_snapshot)
        for mat in material_cache.values():
            if mat.users == 0:
                bpy.data.materials.remove(mat)

    pass_meta = {
        "mode": "neutral_material_shading",
        "neutral_color": neutral,
        "definition": "Original material node trees copied; color/emission inputs neutralized; roughness/metallic/specular/IOR/alpha/transmission/normal/bump preserved when represented by non-color sockets.",
        "world": "black",
        "lights": "one white point light per render",
        "cycles": {
            "max_bounces": 8,
            "diffuse_bounces": 2,
            "glossy_bounces": 4,
            "transmission_bounces": 8,
            "transparent_max_bounces": 8,
            "volume_bounces": 0,
            "caustics": True,
        },
        "renders": rendered,
        "materials": material_diagnostics,
    }
    meta_path = scene_dir / "pbr" / "neutral_material_shading" / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(pass_meta, indent=2), encoding="utf-8")
    return pass_meta


def write_selection_manifest(path: Path, scenes_root: Path, items: list[dict], neutral: float) -> None:
    payload = {
        "schema": "neutral_material_shading_selection_v1",
        "source_root": str(scenes_root),
        "neutral_color": neutral,
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_previews(scenes_root: Path, preview_output_dir: Path, selection_manifest: Path) -> None:
    cmd = [
        "python3",
        str(SCRIPT_DIR / "make_relighting_aux_preview_grid.py"),
        "--scenes-root",
        str(scenes_root),
        "--output-dir",
        str(preview_output_dir),
        "--selection-manifest",
        str(selection_manifest),
        "--shading-rel-template",
        "pbr/neutral_material_shading/point_light_{point_id:03d}.exr",
        "--shading-label-template",
        "neutral shading {point_id:03d}",
        "--shading-panel-name",
        "neutral_material_shading",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config), args.samples)
    scenes_root = resolve_repo_path(args.scenes_root)
    preview_output_dir = resolve_repo_path(args.preview_output_dir)
    selection_manifest = preview_output_dir / "selection_manifest.json"
    rng = random.Random(args.seed)

    all_items = []
    for scene_index in args.scenes:
        scene_dir = scenes_root / f"scene_{scene_index:06d}"
        meta = existing_meta(scene_dir)
        selected = choose_points(scene_dir, meta, rng, args.variants_per_scene, args.min_light_mean, args.min_light_max)
        for item in selected:
            item["scene_id"] = scene_index
            item["skip_existing"] = bool(args.skip_existing)
        render_neutral_points(scene_index, scene_dir, config, selected, float(args.neutral_color))
        all_items.extend(
            {
                "scene_id": int(item["scene_id"]),
                "variant": int(item["variant"]),
                "point_light_id": int(item["point_light_id"]),
                "sampled_color_rgb": item["sampled_color_rgb"],
                "point_light_mean": float(item["point_light_mean"]),
                "point_light_max": float(item["point_light_max"]),
            }
            for item in selected
        )

    write_selection_manifest(selection_manifest, scenes_root, all_items, float(args.neutral_color))
    if not args.no_preview:
        make_previews(scenes_root, preview_output_dir, selection_manifest)
    print(f"[NeutralMaterial] selection manifest: {selection_manifest}")
    print(f"[NeutralMaterial] preview output: {preview_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
