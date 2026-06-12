from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix, Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("scripts/build_portrait_asset_previews.py must be run by Blender Python.") from exc


SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl"}
PREFERRED_EXTS = {
    "renderpeople": [".blend", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"],
    "thuman2": [".obj", ".fbx", ".glb", ".gltf", ".ply", ".stl", ".blend"],
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build Blender preview PNGs and manifests for portrait assets.")
    parser.add_argument("--dataset", choices=["renderpeople", "thuman2"], required=True)
    parser.add_argument("--root", required=True, help="Root folder containing the downloaded assets.")
    parser.add_argument("--out-dir", default=None, help="Preview output folder. Defaults to previews/<dataset>.")
    parser.add_argument("--manifest", default=None, help="Manifest path. Defaults to manifests/<dataset>_objects.txt.")
    parser.add_argument("--metadata-out", default=None, help="Metadata JSON path. Defaults to manifests/<dataset>_objects_meta.json.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--overwrite", action="store_true")
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


def import_asset(path: Path) -> list[bpy.types.Object]:
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
    elif ext == ".obj":
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
        imported = [obj for obj in bpy.data.objects if obj not in before]
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


def normalize_for_preview(objects: list[bpy.types.Object], target_height: float = 2.0) -> tuple[Vector, Vector]:
    bbox_min, bbox_max = mesh_bbox(objects)
    size = bbox_max - bbox_min
    height = max(size.z, 1e-6)
    scale = target_height / height
    center_xy = Vector(((bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5, 0.0))
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(Vector((-center_xy.x, -center_xy.y, -bbox_min.z)))
    roots = [obj for obj in objects if obj.parent not in set(objects)]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()
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
    height = max((bbox_max - bbox_min).z, 1e-6)
    target = Vector((center.x, center.y, bbox_min.z + height * 0.58))
    cam_data = bpy.data.cameras.new("PreviewCamera")
    cam = bpy.data.objects.new("PreviewCamera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.lens = 65
    cam.location = Vector((0.0, -3.2, bbox_min.z + height * 0.58))
    look_at(cam, target)
    scene.camera = cam

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.78, 0.78, 0.78)

    key_data = bpy.data.lights.new("PreviewKey", type="AREA")
    key = bpy.data.objects.new("PreviewKey", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-1.2, -2.2, 2.8)
    key_data.energy = 450
    key_data.size = 4.0
    look_at(key, target)

    fill_data = bpy.data.lights.new("PreviewFill", type="AREA")
    fill = bpy.data.objects.new("PreviewFill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (1.8, -1.6, 1.8)
    fill_data.energy = 75
    fill_data.size = 5.0
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


def find_assets(root: Path, dataset: str) -> list[Path]:
    all_files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS]
    if dataset == "thuman2":
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


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Asset root does not exist: {root}")
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "previews" / args.dataset
    manifest = Path(args.manifest).resolve() if args.manifest else repo_root / "manifests" / f"{args.dataset}_objects.txt"
    metadata_out = Path(args.metadata_out).resolve() if args.metadata_out else repo_root / "manifests" / f"{args.dataset}_objects_meta.json"

    assets = find_assets(root, args.dataset)
    if args.limit is not None:
        assets = assets[: args.limit]
    if not assets:
        raise SystemExit(f"No supported assets found under {root}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = []
    manifest_lines = []
    for index, asset in enumerate(assets):
        name = safe_name(asset, root)
        preview_path = out_dir / f"{index:05d}_{name}.png"
        print(f"[PortraitPreview] {index + 1}/{len(assets)} {asset}", flush=True)
        if not preview_path.exists() or args.overwrite:
            clear_scene()
            try:
                objects = import_asset(asset)
                bbox_min, bbox_max = normalize_for_preview(objects)
                setup_camera_and_lights(bbox_min, bbox_max, args.resolution)
                render_preview(preview_path)
                status = "ok"
                error = None
            except Exception as exc:
                status = "failed"
                error = str(exc)
                bbox_min = Vector((0.0, 0.0, 0.0))
                bbox_max = Vector((0.0, 0.0, 0.0))
                print(f"[PortraitPreview] Failed: {asset}: {exc}", flush=True)
        else:
            status = "ok"
            error = None
            bbox_min = Vector((0.0, 0.0, 0.0))
            bbox_max = Vector((0.0, 0.0, 0.0))
        if status == "ok":
            manifest_lines.append(str(asset))
        metadata.append(
            {
                "asset": str(asset),
                "preview": str(preview_path),
                "status": status,
                "error": error,
                "bbox_min_preview_space": vec_to_list(bbox_min),
                "bbox_max_preview_space": vec_to_list(bbox_max),
            }
        )

    manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PortraitPreview] Wrote manifest: {manifest}", flush=True)
    print(f"[PortraitPreview] Wrote metadata: {metadata_out}", flush=True)
    print(f"[PortraitPreview] Wrote previews: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
