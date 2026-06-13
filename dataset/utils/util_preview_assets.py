from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from utils.util_progress import progress_bar, progress_write

try:
    import bpy
    from mathutils import Matrix, Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("dataset/{portrait,object}/preview_*.py must be run by Blender Python.") from exc


SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".dae", ".ply", ".stl"}
PREFERRED_EXTS = {
    "3dscanstore_free_head": [".blend", ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl"],
    "blenderkit_human": [".blend", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"],
    "renderpeople": [".blend", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"],
    "renderpeople_free": [".blend", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"],
    "humano_free": [".blend", ".fbx", ".glb", ".gltf", ".obj", ".dae", ".ply", ".stl"],
    "hsrd100": [".obj", ".fbx", ".glb", ".gltf", ".ply", ".stl", ".blend"],
    "objaverse_xl": [".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".blend"],
    "sketchfab_human": [".glb", ".gltf", ".blend", ".fbx", ".obj", ".dae", ".ply", ".stl"],
    "thuman2": [".obj", ".fbx", ".glb", ".gltf", ".ply", ".stl", ".blend"],
}


def parse_args(default_dataset: str | None = None) -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build preview PNGs and metadata for portrait/object assets.")
    parser.add_argument("--dataset", choices=sorted(PREFERRED_EXTS), default=default_dataset, required=default_dataset is None)
    parser.add_argument("--root", required=True, help="Root folder containing the downloaded assets.")
    parser.add_argument("--out-dir", default=None, help="Preview image folder. Defaults to outputs/previews/<dataset>/img.")
    parser.add_argument("--metadata-dir", default=None, help="Per-item metadata folder. Defaults to outputs/previews/<dataset>/metadata.")
    parser.add_argument("--index-out", default=None, help="Index JSON path. Defaults to outputs/previews/<dataset>/<dataset>_index.json.")
    parser.add_argument("--manifest", default=None, help="Asset manifest path. Defaults to outputs/previews/<dataset>/<dataset>_objects.txt.")
    parser.add_argument("--metadata-out", default=None, help="Deprecated alias for --index-out.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-dedupe-assets",
        dest="dedupe_assets",
        action="store_false",
        help="Keep multiple formats/LODs for the same RenderPeople identity.",
    )
    parser.set_defaults(dedupe_assets=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_blend_objects(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    imported = []
    for obj in data_to.objects:
        if obj is None:
            continue
        bpy.context.collection.objects.link(obj)
        imported.append(obj)
    return [obj for obj in bpy.data.objects if obj not in before] or imported


def import_obj_asset(path: Path, forward_axis: str | None = None, up_axis: str | None = None) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    kwargs = {"filepath": str(path)}
    if forward_axis is not None:
        kwargs["forward_axis"] = forward_axis
    if up_axis is not None:
        kwargs["up_axis"] = up_axis
    if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(**kwargs)
    else:
        legacy_kwargs = {"filepath": str(path)}
        if forward_axis is not None:
            legacy_kwargs["axis_forward"] = forward_axis.replace("NEGATIVE_", "-")
        if up_axis is not None:
            legacy_kwargs["axis_up"] = up_axis
        bpy.ops.import_scene.obj(**legacy_kwargs)
    return [obj for obj in bpy.data.objects if obj not in before]


def import_asset(path: Path, obj_forward_axis: str | None = None, obj_up_axis: str | None = None) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext == ".blend":
        imported = import_blend_objects(path)
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
    elif ext == ".obj":
        imported = import_obj_asset(path, obj_forward_axis, obj_up_axis)
    elif ext == ".ply":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
    elif ext == ".stl":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
    else:
        raise ValueError(f"Unsupported asset extension: {path}")
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {path}")
    return mesh_objects


def first_obj_value(path: Path, prefix: str) -> str | None:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(prefix):
                value = stripped[len(prefix) :].strip()
                return value or None
    return None


def find_hsrd100_diffuse_texture(asset: Path, material_name: str | None) -> Path | None:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    candidates: list[Path] = []
    if material_name:
        candidates.extend(path for path in asset.parent.glob(f"{material_name}_diffuse.*") if path.suffix.lower() in suffixes)
    candidates.extend(path for path in asset.parent.glob(f"{asset.stem}*_diffuse.*") if path.suffix.lower() in suffixes)
    candidates.extend(path for path in asset.parent.glob("*diffuse.*") if path.suffix.lower() in suffixes)
    return sorted(set(candidates), key=str)[0] if candidates else None


def write_hsrd100_mtl(asset: Path, item_id: str, mtl_dir: Path) -> dict | None:
    if asset.suffix.lower() != ".obj":
        return None
    material_name = first_obj_value(asset, "usemtl ") or f"{asset.stem}_mat"
    obj_mtl_name = first_obj_value(asset, "mtllib ") or f"{asset.stem}.mtl"
    texture = find_hsrd100_diffuse_texture(asset, material_name)
    if texture is None:
        return None
    mtl_dir.mkdir(parents=True, exist_ok=True)
    mtl_path = mtl_dir / f"{item_id}.mtl"
    mtl_text = "\n".join(
        [
            f"# Generated preview material for {item_id}",
            f"# Source OBJ: {asset}",
            f"# Source texture: {texture}",
            f"newmtl {material_name}",
            "Ka 1.000000 1.000000 1.000000",
            "Kd 1.000000 1.000000 1.000000",
            "Ks 0.000000 0.000000 0.000000",
            "Ns 10.000000",
            "d 1.000000",
            "illum 2",
            f"map_Kd {texture.name}",
            "",
        ]
    )
    mtl_path.write_text(mtl_text, encoding="utf-8")
    return {
        "path": mtl_path,
        "obj_mtl_name": obj_mtl_name,
        "material_name": material_name,
        "texture": texture,
    }


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


def apply_texture_material(objects: list[bpy.types.Object], material_name: str, texture: Path) -> None:
    mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(texture), check_existing=True)
    if bsdf:
        mat.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.75
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def import_hsrd100_asset(asset: Path, mtl_info: dict | None) -> list[bpy.types.Object]:
    if asset.suffix.lower() != ".obj":
        return import_asset(asset, obj_forward_axis="NEGATIVE_Y", obj_up_axis="Z")
    link_path = asset.parent / (mtl_info["obj_mtl_name"] if mtl_info else f"{asset.stem}.mtl")
    made_link = False
    if mtl_info and not link_path.exists():
        try:
            link_path.symlink_to(mtl_info["path"])
            made_link = True
        except OSError:
            made_link = False
    try:
        objects = import_asset(asset, obj_forward_axis="NEGATIVE_Y", obj_up_axis="Z")
    finally:
        if made_link:
            link_path.unlink(missing_ok=True)
    if mtl_info and not objects_have_image_texture(objects):
        apply_texture_material(objects, mtl_info["material_name"], mtl_info["texture"])
    return objects


def mesh_bbox(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        eval_obj = obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            point = eval_obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
    return mins, maxs


def root_objects(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    object_set = set(objects)
    return [obj for obj in objects if obj.parent not in object_set]


def transform_roots(objects: list[bpy.types.Object], transform: Matrix) -> None:
    for obj in root_objects(objects):
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def upright_longest_axis(objects: list[bpy.types.Object]) -> None:
    bbox_min, bbox_max = mesh_bbox(objects)
    size = bbox_max - bbox_min
    dims = [size.x, size.y, size.z]
    longest_axis = max(range(3), key=lambda axis: dims[axis])
    if longest_axis == 0:
        transform_roots(objects, Matrix.Rotation(math.radians(-90.0), 4, "Y"))
    elif longest_axis == 1:
        transform_roots(objects, Matrix.Rotation(math.radians(90.0), 4, "X"))


def normalize_for_preview(
    objects: list[bpy.types.Object],
    target_height: float = 2.0,
    upright: bool = False,
) -> tuple[Vector, Vector]:
    if upright:
        upright_longest_axis(objects)
    bbox_min, bbox_max = mesh_bbox(objects)
    size = bbox_max - bbox_min
    height = max(size.z, 1e-6)
    scale = target_height / height
    center_xy = Vector(((bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5, 0.0))
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(Vector((-center_xy.x, -center_xy.y, -bbox_min.z)))
    transform_roots(objects, transform)
    return mesh_bbox(objects)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera_and_lights(bbox_min: Vector, bbox_max: Vector, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    center = (bbox_min + bbox_max) * 0.5
    size = bbox_max - bbox_min
    height = max(size.z, 1e-6)
    span = max(size.x, size.y, size.z, 1.0)
    target = Vector((center.x, center.y, bbox_min.z + height * 0.54))
    cam_data = bpy.data.cameras.new("PreviewCamera")
    cam = bpy.data.objects.new("PreviewCamera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(height, size.x, size.y) * 1.18
    camera_distance = span * 3.0
    if size.x >= size.y:
        cam.location = Vector((center.x, bbox_min.y - camera_distance, target.z))
    else:
        cam.location = Vector((bbox_min.x - camera_distance, center.y, target.z))
    look_at(cam, target)
    scene.camera = cam

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.78, 0.78, 0.78)

    key_data = bpy.data.lights.new("PreviewKey", type="AREA")
    key = bpy.data.objects.new("PreviewKey", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (target.x - span * 0.8, target.y - span * 1.2, target.z + span * 1.2)
    key_data.energy = 450
    key_data.size = max(span * 2.0, 2.0)
    look_at(key, target)

    fill_data = bpy.data.lights.new("PreviewFill", type="AREA")
    fill = bpy.data.objects.new("PreviewFill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (target.x + span * 1.2, target.y - span * 0.8, target.z + span * 0.6)
    fill_data.energy = 75
    fill_data.size = max(span * 2.4, 2.0)
    look_at(fill, target)


def render_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)


def candidate_score(path: Path, dataset: str) -> tuple[int, int, str]:
    preferred = PREFERRED_EXTS[dataset]
    ext_rank = preferred.index(path.suffix.lower()) if path.suffix.lower() in preferred else len(preferred)
    return (ext_rank, len(path.parts), str(path))


def renderpeople_identity_key(path: Path) -> str:
    stem = path.stem.lower()
    for pattern in (
        r"_(?:100k|200k|300k|60k|30k)$",
        r"_(?:2k|4k)$",
        r"_(?:yup|zup)_[at]$",
        r"_(?:u3d|ue4|ue)$",
    ):
        stem = re.sub(pattern, "", stem)
    for marker in ("_animated_", "_rigged_", "_posed_", "_4d_"):
        index = stem.find(marker)
        if index > 2 and stem[:index] != "rp":
            return stem[:index]
    return stem


def renderpeople_score(path: Path, dataset: str) -> tuple[int, int, int, int, int, str]:
    preferred = PREFERRED_EXTS[dataset]
    ext_rank = preferred.index(path.suffix.lower()) if path.suffix.lower() in preferred else len(preferred)
    lower_path = str(path).lower()
    format_order = ["_bld", "_fbx", "_obj", "_glb", "_c4d", "_max", "_maya", "_3dm", "_skp", "_u3d", "_ue4", "_ue"]
    format_rank = next((index for index, token in enumerate(format_order) if token in lower_path), len(format_order))
    lod_order = ["300k", "200k", "100k", "60k", "30k", "4k", "2k"]
    lod_rank = next((index for index, token in enumerate(lod_order) if token in path.stem.lower()), len(lod_order))
    pose_order = ["zup_t", "zup_a", "yup_t", "yup_a", "u3d", "ue4", "ue"]
    pose_rank = next((index for index, token in enumerate(pose_order) if token in path.stem.lower()), len(pose_order))
    return (ext_rank, format_rank, lod_rank, pose_rank, len(path.parts), str(path))


def find_assets(root: Path, dataset: str, dedupe_assets: bool = True) -> list[Path]:
    all_files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS]
    if dataset == "objaverse_xl":
        selected = sorted(all_files, key=lambda p: candidate_score(p, dataset))
    elif dataset in {"renderpeople", "renderpeople_free"} and dedupe_assets:
        grouped: dict[str, list[Path]] = {}
        for path in all_files:
            grouped.setdefault(renderpeople_identity_key(path), []).append(path)
        selected = [sorted(paths, key=lambda p: renderpeople_score(p, dataset))[0] for paths in grouped.values()]
    elif dataset == "thuman2":
        # THuman folders often contain one useful textured OBJ per subject. Keep one best candidate per folder.
        grouped: dict[Path, list[Path]] = {}
        for path in all_files:
            grouped.setdefault(path.parent, []).append(path)
        selected = [sorted(paths, key=lambda p: candidate_score(p, dataset))[0] for paths in grouped.values()]
    else:
        # RenderPeople may ship multiple formats per asset folder. Group by parent and prefer render-ready formats.
        grouped = {}
        for path in all_files:
            key = path.parent
            grouped.setdefault(key, []).append(path)
        selected = [sorted(paths, key=lambda p: candidate_score(p, dataset))[0] for paths in grouped.values()]
    return sorted(selected, key=str)


def safe_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    stem = "_".join(rel.with_suffix("").parts)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)


def vec_to_list(v: Vector) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def main(default_dataset: str | None = None) -> int:
    args = parse_args(default_dataset)
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Asset root does not exist: {root}")
    repo_root = Path(__file__).resolve().parents[2]
    preview_root = repo_root / "outputs" / "previews" / args.dataset
    out_dir = Path(args.out_dir).resolve() if args.out_dir else preview_root / "img"
    metadata_dir = Path(args.metadata_dir).resolve() if args.metadata_dir else preview_root / "metadata"
    index_out_arg = args.index_out or args.metadata_out
    index_out = Path(index_out_arg).resolve() if index_out_arg else preview_root / f"{args.dataset}_index.json"
    manifest = Path(args.manifest).resolve() if args.manifest else preview_root / f"{args.dataset}_objects.txt"
    mtl_dir = preview_root / "mtl" if args.dataset == "hsrd100" else None

    assets = find_assets(root, args.dataset, args.dedupe_assets)
    if args.limit is not None:
        assets = assets[: args.limit]
    if not assets:
        raise SystemExit(f"No supported assets found under {root}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    if mtl_dir is not None:
        mtl_dir.mkdir(parents=True, exist_ok=True)
    index_items = []
    manifest_lines = []
    with progress_bar(assets, total=len(assets), desc=f"Preview {args.dataset}", unit="asset") as pbar:
        for index, asset in enumerate(pbar):
            item_id = f"{args.dataset}_{index + 1:06d}"
            pbar.set_postfix(item=item_id)
            preview_path = out_dir / f"{item_id}.png"
            item_metadata_path = metadata_dir / f"{item_id}.json"
            mtl_info = write_hsrd100_mtl(asset, item_id, mtl_dir) if mtl_dir is not None else None
            progress_write(f"[PortraitPreview] {index + 1}/{len(assets)} {asset}")
            if not preview_path.exists() or args.overwrite:
                clear_scene()
                try:
                    objects = import_hsrd100_asset(asset, mtl_info) if args.dataset == "hsrd100" else import_asset(asset)
                    upright = args.dataset in {
                        "3dscanstore_free_head",
                        "blenderkit_human",
                        "hsrd100",
                        "renderpeople",
                        "renderpeople_free",
                        "humano_free",
                        "sketchfab_human",
                        "thuman2",
                    }
                    bbox_min, bbox_max = normalize_for_preview(objects, upright=upright)
                    setup_camera_and_lights(bbox_min, bbox_max, args.resolution)
                    render_preview(preview_path)
                    status = "ok"
                    error = None
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    bbox_min = Vector((0.0, 0.0, 0.0))
                    bbox_max = Vector((0.0, 0.0, 0.0))
                    progress_write(f"[PortraitPreview] Failed: {asset}: {exc}")
            else:
                status = "ok"
                error = None
                bbox_min = Vector((0.0, 0.0, 0.0))
                bbox_max = Vector((0.0, 0.0, 0.0))
            if status == "ok":
                manifest_lines.append(str(asset))
            item_metadata = {
                "id": item_id,
                "dataset": args.dataset,
                "asset": str(asset),
                "source_path": str(asset),
                "preview": str(preview_path),
                "mtl": str(mtl_info["path"]) if mtl_info else None,
                "asset_type": "portrait_asset",
                "status": status,
                "error": error,
                "bbox_min_preview_space": vec_to_list(bbox_min),
                "bbox_max_preview_space": vec_to_list(bbox_max),
            }
            item_metadata_path.write_text(json.dumps(item_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            index_items.append({"id": item_id, "metadata": str(item_metadata_path), "preview": str(preview_path), "source_path": str(asset), "status": status})

    manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    index_out.write_text(json.dumps({"dataset": args.dataset, "items": index_items}, indent=2, ensure_ascii=False), encoding="utf-8")
    progress_write(f"[PortraitPreview] Wrote manifest: {manifest}")
    progress_write(f"[PortraitPreview] Wrote index: {index_out}")
    progress_write(f"[PortraitPreview] Wrote metadata: {metadata_dir}")
    progress_write(f"[PortraitPreview] Wrote previews: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
