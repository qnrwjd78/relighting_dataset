from __future__ import annotations

import argparse
import json
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

import render_neutral_material_shading_previews as neutral  # noqa: E402
import render_object_relighting as relight  # noqa: E402


CANDIDATES = {
    "matte_direct_a1": {
        "material_mode": "matte",
        "albedo": 1.0,
        "bounces": {
            "max_bounces": 0,
            "diffuse_bounces": 0,
            "glossy_bounces": 0,
            "transmission_bounces": 0,
            "transparent_max_bounces": 0,
            "volume_bounces": 0,
            "caustics": True,
        },
        "description": "All materials replaced by white diffuse albedo 1.0; direct lighting only.",
    },
    "matte_direct_a0p6": {
        "material_mode": "matte",
        "albedo": 0.6,
        "bounces": {
            "max_bounces": 0,
            "diffuse_bounces": 0,
            "glossy_bounces": 0,
            "transmission_bounces": 0,
            "transparent_max_bounces": 0,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "All materials replaced by neutral diffuse albedo 0.6; direct lighting only.",
    },
    "optical_direct_a0p6": {
        "material_mode": "optical",
        "albedo": 0.6,
        "bounces": {
            "max_bounces": 4,
            "diffuse_bounces": 0,
            "glossy_bounces": 1,
            "transmission_bounces": 4,
            "transparent_max_bounces": 4,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Neutral diffuse albedo 0.6 for opaque materials; colorless original optical materials preserved; no diffuse GI.",
    },
    "optical_soft_a0p6": {
        "material_mode": "optical",
        "albedo": 0.6,
        "bounces": {
            "max_bounces": 6,
            "diffuse_bounces": 1,
            "glossy_bounces": 2,
            "transmission_bounces": 6,
            "transparent_max_bounces": 6,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Neutral diffuse albedo 0.6 for opaque materials; optical preserved; one diffuse bounce, limited glossy/transmission.",
    },
    "neutral_direct_a0p5": {
        "material_mode": "neutral_original",
        "albedo": 0.5,
        "bounces": {
            "max_bounces": 4,
            "diffuse_bounces": 0,
            "glossy_bounces": 1,
            "transmission_bounces": 4,
            "transparent_max_bounces": 4,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Copied original material node trees; color/emission neutralized to 0.5; no diffuse GI, limited glossy/transmission.",
    },
    "neutral_soft_a0p5": {
        "material_mode": "neutral_original",
        "albedo": 0.5,
        "bounces": {
            "max_bounces": 6,
            "diffuse_bounces": 1,
            "glossy_bounces": 2,
            "transmission_bounces": 6,
            "transparent_max_bounces": 6,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Copied original material node trees; color/emission neutralized to 0.5; one diffuse bounce, limited glossy/transmission.",
    },
    "neutral_direct_a0p35": {
        "material_mode": "neutral_original",
        "albedo": 0.35,
        "bounces": {
            "max_bounces": 4,
            "diffuse_bounces": 0,
            "glossy_bounces": 1,
            "transmission_bounces": 4,
            "transparent_max_bounces": 4,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Copied original material node trees; color/emission neutralized to 0.35; no diffuse GI, limited glossy/transmission.",
    },
    "neutral_soft_a0p35": {
        "material_mode": "neutral_original",
        "albedo": 0.35,
        "bounces": {
            "max_bounces": 6,
            "diffuse_bounces": 1,
            "glossy_bounces": 2,
            "transmission_bounces": 6,
            "transparent_max_bounces": 6,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Copied original material node trees; color/emission neutralized to 0.35; one diffuse bounce, limited glossy/transmission.",
    },
    "neutral_direct_obj0p5_recv0p25": {
        "material_mode": "neutral_original_split",
        "object_albedo": 0.5,
        "receiver_albedo": 0.25,
        "bounces": {
            "max_bounces": 4,
            "diffuse_bounces": 0,
            "glossy_bounces": 1,
            "transmission_bounces": 4,
            "transparent_max_bounces": 4,
            "volume_bounces": 0,
            "caustics": True,
        },
        "description": "Original material node trees copied; subject colors neutralized to 0.5, receiver floor/wall colors to 0.25; no diffuse GI; caustics enabled.",
    },
    "point_direct_objorig_recv0p25": {
        "material_mode": "original_object_neutral_receiver",
        "receiver_albedo": 0.25,
        "bounces": {
            "max_bounces": 4,
            "diffuse_bounces": 0,
            "glossy_bounces": 1,
            "transmission_bounces": 4,
            "transparent_max_bounces": 4,
            "volume_bounces": 0,
            "caustics": True,
        },
        "description": "Original subject material colors/textures preserved; receiver floor/wall colors neutralized to 0.25; no diffuse GI; caustics enabled.",
    },
    "point_direct_original_all": {
        "material_mode": "original_all",
        "bounces": {
            "max_bounces": 4,
            "diffuse_bounces": 0,
            "glossy_bounces": 1,
            "transmission_bounces": 4,
            "transparent_max_bounces": 4,
            "volume_bounces": 0,
            "caustics": True,
        },
        "description": "All original subject and receiver material colors/textures preserved; no diffuse GI; limited glossy/transmission; caustics enabled.",
    },
    "neutral_soft_obj0p5_recv0p25": {
        "material_mode": "neutral_original_split",
        "object_albedo": 0.5,
        "receiver_albedo": 0.25,
        "bounces": {
            "max_bounces": 6,
            "diffuse_bounces": 1,
            "glossy_bounces": 2,
            "transmission_bounces": 6,
            "transparent_max_bounces": 6,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Original material node trees copied; subject colors neutralized to 0.5, receiver floor/wall colors to 0.25; one diffuse bounce.",
    },
    "optical_soft_a0p45": {
        "material_mode": "optical",
        "albedo": 0.45,
        "bounces": {
            "max_bounces": 6,
            "diffuse_bounces": 1,
            "glossy_bounces": 2,
            "transmission_bounces": 6,
            "transparent_max_bounces": 6,
            "volume_bounces": 0,
            "caustics": False,
        },
        "description": "Neutral diffuse albedo 0.45 for opaque materials; optical preserved; one diffuse bounce, limited glossy/transmission.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render single-light shading candidate passes.")
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json")
    parser.add_argument(
        "--scenes-root",
        default="outputs/objaverse_ratio3p5_cube1p6_full_scene4000_4039_640/scenes",
    )
    parser.add_argument(
        "--selection-manifest",
        default="outputs/relighting_aux_previews_ambient_point_white_depth_normal/manifest.json",
    )
    parser.add_argument("--scenes", nargs="*", type=int, default=None)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_selection(path: Path, scenes: set[int] | None) -> dict[int, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items", payload if isinstance(payload, list) else [])
    result: dict[int, list[dict]] = {}
    for row in rows:
        scene_id = int(row["scene_id"])
        if scenes is not None and scene_id not in scenes:
            continue
        result.setdefault(scene_id, []).append(row)
    for scene_rows in result.values():
        scene_rows.sort(key=lambda item: int(item.get("variant", 0)))
    return result


def make_diffuse_override_material(name: str, albedo: float) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = (albedo, albedo, albedo, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    relight.set_input(diffuse, "Color", (albedo, albedo, albedo, 1.0))
    relight.set_input(diffuse, "Roughness", 0.5)
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return mat


def apply_candidate_materials(candidate: dict):
    mode = str(candidate["material_mode"])
    if mode == "matte":
        albedo = float(candidate["albedo"])
        mat = make_diffuse_override_material(f"TL_candidate_matte_{albedo:.3f}".replace(".", "p"), albedo)
        snapshot, cache = relight.apply_white_diffuse_material_override(mat, preserve_optical=False)
        return snapshot, cache
    if mode == "optical":
        albedo = float(candidate["albedo"])
        mat = make_diffuse_override_material(f"TL_candidate_optical_diffuse_{albedo:.3f}".replace(".", "p"), albedo)
        snapshot, cache = relight.apply_white_diffuse_material_override(mat, preserve_optical=True)
        return snapshot, cache
    if mode == "neutral_original":
        albedo = float(candidate["albedo"])
        snapshot, cache, _diagnostics = neutral.apply_neutral_material_override(albedo)
        return snapshot, cache
    if mode == "neutral_original_split":
        return apply_split_neutral_material_override(
            float(candidate["object_albedo"]),
            float(candidate["receiver_albedo"]),
        )
    if mode == "original_object_neutral_receiver":
        return apply_receiver_neutral_material_override(float(candidate["receiver_albedo"]))
    if mode == "original_all":
        return relight.object_material_snapshot(), {}
    raise ValueError(f"Unknown material mode: {mode}")


def apply_split_neutral_material_override(
    object_neutral: float,
    receiver_neutral: float,
) -> tuple[dict[str, list[bpy.types.Material]], dict[str, bpy.types.Material]]:
    snapshot = relight.object_material_snapshot()
    cache: dict[str, bpy.types.Material] = {}
    default_object = neutral.make_default_neutral_material(object_neutral)
    default_receiver = neutral.make_default_neutral_material(receiver_neutral)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        neutral_value = receiver_neutral if bool(obj.get("TL_RECEIVER", False)) else object_neutral
        default_mat = default_receiver if bool(obj.get("TL_RECEIVER", False)) else default_object
        original = snapshot.get(obj.name, [])
        obj.data.materials.clear()
        if not original:
            obj.data.materials.append(default_mat)
            continue
        for source in original:
            if source is None:
                obj.data.materials.append(default_mat)
                continue
            key = f"{source.name}|{neutral_value:.4f}"
            if key not in cache:
                cache[key] = neutral.neutralize_original_material(source, neutral_value)
            obj.data.materials.append(cache[key])
    return snapshot, cache


def apply_receiver_neutral_material_override(
    receiver_neutral: float,
) -> tuple[dict[str, list[bpy.types.Material]], dict[str, bpy.types.Material]]:
    snapshot = relight.object_material_snapshot()
    cache: dict[str, bpy.types.Material] = {}
    default_receiver = neutral.make_default_neutral_material(receiver_neutral)
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not bool(obj.get("TL_RECEIVER", False)):
            continue
        original = snapshot.get(obj.name, [])
        obj.data.materials.clear()
        if not original:
            obj.data.materials.append(default_receiver)
            continue
        for source in original:
            if source is None:
                obj.data.materials.append(default_receiver)
                continue
            key = f"{source.name}|receiver|{receiver_neutral:.4f}"
            if key not in cache:
                cache[key] = neutral.neutralize_original_material(source, receiver_neutral)
            obj.data.materials.append(cache[key])
    return snapshot, cache


def set_cycles_for_candidate(candidate: dict) -> dict[str, int] | None:
    snapshot = relight.cycles_bounce_snapshot()
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return snapshot
    bounces = candidate["bounces"]
    for key, value in bounces.items():
        if key == "caustics":
            continue
        if hasattr(scene.cycles, key):
            setattr(scene.cycles, key, int(value))
    caustics = bool(bounces.get("caustics", False))
    for key in ("caustics_reflective", "caustics_refractive"):
        if hasattr(scene.cycles, key):
            setattr(scene.cycles, key, caustics)
    return snapshot


def recreate_scene(scene_index: int, config: dict, meta: dict) -> None:
    neutral.recreate_scene(scene_index, config, meta)
    relight.remove_all_lights()
    relight.set_black_world()


def render_candidate(
    scene_dir: Path,
    point_rows: dict[int, dict],
    config: dict,
    candidate_name: str,
    candidate: dict,
    selected: list[dict],
) -> list[dict]:
    material_snapshot, cache = apply_candidate_materials(candidate)
    bounce_snapshot = set_cycles_for_candidate(candidate)
    rendered = []
    try:
        for item in selected:
            point_id = int(item["point_light_id"])
            point_meta = point_rows[point_id]
            rel_base = f"pbr/shading_candidates/{candidate_name}/point_light_{point_id:03d}"
            out_path = scene_dir / f"{rel_base}.exr"
            if out_path.exists() and bool(item.get("skip_existing", False)):
                print(f"[Candidate] exists {out_path}")
            else:
                relight.remove_all_lights()
                relight.set_black_world()
                light = relight.create_point_light(
                    f"TL_Candidate_Point_{point_id:03d}",
                    Vector(point_meta["world_position"]),
                    float(point_meta["world_energy"]),
                    float(point_meta["world_radius"]),
                    [1.0, 1.0, 1.0],
                )
                try:
                    relight.render_component(scene_dir, rel_base, config)
                finally:
                    bpy.data.objects.remove(light, do_unlink=True)
                print(f"[Candidate] rendered {out_path}")
            rendered.append({"point_light_id": point_id, "render": f"{rel_base}.exr"})
    finally:
        relight.restore_cycles_bounces(bounce_snapshot)
        relight.restore_object_materials(material_snapshot)
        for mat in cache.values():
            if mat.users == 0:
                bpy.data.materials.remove(mat)
        relight.remove_all_lights()
        relight.set_black_world()
    return rendered


def write_scene_candidate_meta(scene_dir: Path, candidate_name: str, candidate: dict, rendered: list[dict]) -> None:
    meta_path = scene_dir / "pbr" / "shading_candidates" / candidate_name / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate": candidate_name,
        "definition": candidate["description"],
        "material_mode": candidate["material_mode"],
        "albedo": candidate.get("albedo"),
        "object_albedo": candidate.get("object_albedo"),
        "receiver_albedo": candidate.get("receiver_albedo"),
        "bounces": candidate["bounces"],
        "world": "black",
        "lights": "one white point light per render",
        "renders": rendered,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    unknown = [name for name in args.candidates if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"Unknown candidates: {unknown}. Available: {sorted(CANDIDATES)}")

    config = neutral.load_config(resolve_repo_path(args.config), int(args.samples))
    scenes_root = resolve_repo_path(args.scenes_root)
    scenes_filter = set(args.scenes) if args.scenes else None
    selections = load_selection(resolve_repo_path(args.selection_manifest), scenes_filter)
    manifest = {
        "schema": "shading_candidate_render_v1",
        "source_root": str(scenes_root),
        "selection_manifest": str(resolve_repo_path(args.selection_manifest)),
        "samples": int(args.samples),
        "candidates": {name: CANDIDATES[name] for name in args.candidates},
        "items": [],
    }

    for scene_index, selected in selections.items():
        scene_dir = scenes_root / f"scene_{scene_index:06d}"
        meta = neutral.existing_meta(scene_dir)
        recreate_scene(scene_index, config, meta)
        point_rows = {int(row["id"]): row for row in meta["spatial"]["point_lights"]}
        for item in selected:
            item["skip_existing"] = bool(args.skip_existing)
        for candidate_name in args.candidates:
            rendered = render_candidate(scene_dir, point_rows, config, candidate_name, CANDIDATES[candidate_name], selected)
            write_scene_candidate_meta(scene_dir, candidate_name, CANDIDATES[candidate_name], rendered)
            manifest["items"].append(
                {
                    "scene_id": scene_index,
                    "candidate": candidate_name,
                    "renders": rendered,
                }
            )

    out_path = scenes_root.parent / "shading_candidate_render_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Candidate] manifest {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
