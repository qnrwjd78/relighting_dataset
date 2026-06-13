from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "facescape" / "tu_model" / "extracted"
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "previews" / "facescape_tu_001_300"
EXPRESSION_NAMES = {
    1: "neutral",
    2: "smile",
    3: "mouth_stretch",
    4: "anger",
    5: "jaw_left",
    6: "jaw_right",
    7: "jaw_forward",
    8: "mouth_left",
    9: "mouth_right",
    10: "dimpler",
    11: "chin_raiser",
    12: "lip_puckerer",
    13: "lip_funneler",
    14: "sadness",
    15: "lip_roll",
    16: "grin",
    17: "cheek_blowing",
    18: "eye_closed",
    19: "brow_raiser",
    20: "brow_lower",
}


@dataclass(frozen=True)
class FaceScapeAsset:
    subject: int
    expression: int
    expression_name: str
    obj: Path
    mtl: Path | None
    texture: Path | None
    dpmap: Path | None


def blender_args() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def parse_expression_list(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(1, 21))
    exprs: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            exprs.extend(range(int(start), int(end) + 1))
        else:
            exprs.append(int(token))
    bad = [expr for expr in exprs if expr < 1 or expr > 20]
    if bad:
        raise argparse.ArgumentTypeError(f"Expression ids must be in 1..20: {bad}")
    return sorted(dict.fromkeys(exprs))


def parse_args() -> argparse.Namespace:
    raw_args = blender_args()
    parser = argparse.ArgumentParser(description="Render FaceScape TU previews for the downloaded 1-300 subset.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help=f"FaceScape TU extracted root. Default: {DEFAULT_ROOT}")
    parser.add_argument("--out-dir", default=str(DEFAULT_PREVIEW_ROOT / "img"), help="Preview PNG directory.")
    parser.add_argument("--metadata-dir", default=str(DEFAULT_PREVIEW_ROOT / "metadata"), help="Per-preview metadata directory.")
    parser.add_argument("--index-out", default=str(DEFAULT_PREVIEW_ROOT / "facescape_tu_001_300_index.json"), help="Index JSON path.")
    parser.add_argument("--subject-start", type=int, default=1)
    parser.add_argument("--subject-end", type=int, default=300)
    parser.add_argument(
        "--expressions",
        type=parse_expression_list,
        default=parse_expression_list("1"),
        help="Expression ids: e.g. 1, 1,2,3, 1-5, or all. Default: 1 (neutral).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N selected assets.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "optix"), default="auto")
    parser.add_argument(
        "--orientation",
        choices=("z-up", "y-up-x90", "y-up-neg-x90"),
        default="z-up",
        help="Mesh coordinate conversion before rendering. FaceScape TU OBJ files are usually already z-up.",
    )
    parser.add_argument(
        "--view-axis",
        choices=("pos-y", "neg-y", "pos-x", "neg-x"),
        default="neg-y",
        help="Camera side after orientation conversion. Default neg-y is FaceScape front view for z-up meshes.",
    )
    parser.add_argument("--front-distance", type=float, default=2.8, help="Camera distance multiplier.")
    parser.add_argument("--target-height", type=float, default=2.0, help="Normalized face height in Blender units.")
    parser.add_argument("--add-eyes", action="store_true", help="Add synthetic proxy eyeballs/iris for preview renders.")
    parser.add_argument("--eye-color", choices=("dark_brown", "blue", "green", "gray", "random"), default="dark_brown")
    parser.add_argument("--add-hair-cap", action="store_true", help="Add a simple synthetic hair cap for preview renders.")
    parser.add_argument("--hair-color", choices=("black", "brown", "blond", "auburn", "gray", "random"), default="brown")
    parser.add_argument("--augmentation-seed", type=int, default=20260613)
    args = parser.parse_args(raw_args)
    provided = {token for token in raw_args if token.startswith("--")}
    if (args.add_eyes or args.add_hair_cap) and "--out-dir" not in provided:
        augmented_root = REPO_ROOT / "outputs" / "previews" / "facescape_tu_augmented"
        args.out_dir = str(augmented_root / "img")
        if "--metadata-dir" not in provided:
            args.metadata_dir = str(augmented_root / "metadata")
        if "--index-out" not in provided:
            args.index_out = str(augmented_root / "facescape_tu_augmented_index.json")
    return args


def import_blender_modules():
    try:
        import bpy  # type: ignore
        from mathutils import Matrix, Vector  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Run this script with Blender, e.g. blender -b --python dataset/portrait/preview_facescape_tu.py -- ..."
        ) from exc
    return bpy, Matrix, Vector


def parse_expression_from_obj(path: Path) -> tuple[int, str] | None:
    match = re.match(r"^(\d+)_(.+)\.obj$", path.name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def find_asset(root: Path, subject: int, expression: int) -> FaceScapeAsset | None:
    subject_dir = root / str(subject)
    models_dir = subject_dir / "models_reg"
    dpmap_dir = subject_dir / "dpmap"
    if not models_dir.exists():
        return None
    matches = sorted(models_dir.glob(f"{expression}_*.obj"))
    if not matches:
        return None
    obj = matches[0]
    parsed = parse_expression_from_obj(obj)
    expression_name = parsed[1] if parsed else EXPRESSION_NAMES.get(expression, f"expr_{expression}")
    mtl = obj.with_suffix(obj.suffix + ".mtl")
    texture = obj.with_suffix(".jpg")
    dpmap_matches = sorted(dpmap_dir.glob(f"{expression}_*.png"))
    return FaceScapeAsset(
        subject=subject,
        expression=expression,
        expression_name=expression_name,
        obj=obj,
        mtl=mtl if mtl.exists() else None,
        texture=texture if texture.exists() else None,
        dpmap=dpmap_matches[0] if dpmap_matches else None,
    )


def collect_assets(root: Path, subject_start: int, subject_end: int, expressions: list[int]) -> list[FaceScapeAsset]:
    assets: list[FaceScapeAsset] = []
    for subject in range(subject_start, subject_end + 1):
        for expression in expressions:
            asset = find_asset(root, subject, expression)
            if asset is not None:
                assets.append(asset)
    return assets


def clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_obj(bpy, path: Path) -> list:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    objects = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if not objects:
        objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {path}")
    return objects


def objects_have_image_texture(objects: list) -> bool:
    for obj in objects:
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                    return True
    return False


def apply_texture_material(bpy, objects: list, texture: Path) -> None:
    mat = bpy.data.materials.new("FaceScapeTexture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(texture), check_existing=True)
    if bsdf:
        mat.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.65
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def make_principled_material(bpy, name: str, color: tuple[float, float, float, float], roughness: float, metallic: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
    return mat


def color_choice(name: str, palette: dict[str, tuple[float, float, float, float]], rng: random.Random) -> tuple[str, tuple[float, float, float, float]]:
    if name == "random":
        name = rng.choice(sorted(palette))
    return name, palette[name]


def front_direction(Vector, view_axis: str):
    if view_axis == "neg-y":
        return Vector((0.0, -1.0, 0.0))
    if view_axis == "pos-y":
        return Vector((0.0, 1.0, 0.0))
    if view_axis == "neg-x":
        return Vector((-1.0, 0.0, 0.0))
    if view_axis == "pos-x":
        return Vector((1.0, 0.0, 0.0))
    raise ValueError(f"Unsupported view axis: {view_axis}")


def add_proxy_eyes(bpy, Vector, bbox_min, bbox_max, view_axis: str, eye_color: str, rng: random.Random) -> dict:
    eye_palette = {
        "dark_brown": (0.08, 0.035, 0.015, 1.0),
        "blue": (0.08, 0.22, 0.45, 1.0),
        "green": (0.08, 0.28, 0.12, 1.0),
        "gray": (0.18, 0.20, 0.21, 1.0),
    }
    chosen_name, iris_color = color_choice(eye_color, eye_palette, rng)
    size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) * 0.5
    front = front_direction(Vector, view_axis)
    radius = max(size.z * 0.033, 0.018)
    eye_z = bbox_min.z + size.z * 0.57
    side_span = size.x if abs(front.y) > 0.0 else size.y
    offsets = (-side_span * 0.16, side_span * 0.16)
    white = make_principled_material(bpy, "SyntheticEyeWhite", (0.92, 0.89, 0.84, 1.0), 0.18)
    iris = make_principled_material(bpy, f"SyntheticIris_{chosen_name}", iris_color, 0.26)
    pupil = make_principled_material(bpy, "SyntheticPupil", (0.002, 0.001, 0.001, 1.0), 0.2)

    for offset in offsets:
        loc = Vector((center.x + offset, center.y, eye_z)) if abs(front.y) > 0.0 else Vector((center.x, center.y + offset, eye_z))
        loc = loc + front * radius * 0.25
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
        bpy.context.object.data.materials.append(white)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius * 0.48, location=loc + front * radius * 0.70)
        bpy.context.object.data.materials.append(iris)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius * 0.20, location=loc + front * radius * 0.95)
        bpy.context.object.data.materials.append(pupil)
    return {"synthetic_eyes": True, "eye_color": chosen_name}


def add_proxy_hair_cap(bpy, Vector, bbox_min, bbox_max, view_axis: str, hair_color: str, rng: random.Random) -> dict:
    hair_palette = {
        "black": (0.015, 0.012, 0.01, 1.0),
        "brown": (0.16, 0.075, 0.032, 1.0),
        "blond": (0.78, 0.58, 0.28, 1.0),
        "auburn": (0.32, 0.09, 0.035, 1.0),
        "gray": (0.38, 0.36, 0.34, 1.0),
    }
    chosen_name, color = color_choice(hair_color, hair_palette, rng)
    size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) * 0.5
    front = front_direction(Vector, view_axis)
    back = -front
    loc = Vector((center.x, center.y, bbox_min.z + size.z * 0.82)) + back * max(size.x, size.y) * 0.05
    radius = max(size.x, size.y, size.z) * 0.18
    mat = make_principled_material(bpy, f"SyntheticHair_{chosen_name}", color, 0.72)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=16, radius=radius, location=loc)
    cap = bpy.context.object
    cap.name = "SyntheticHairCap"
    cap.scale.x = max(size.x / (radius * 2.0) * 0.74, 0.8)
    cap.scale.y = max(size.y / (radius * 2.0) * 0.72, 0.55)
    cap.scale.z = 0.52
    cap.data.materials.append(mat)
    return {"synthetic_hair": True, "hair_color": chosen_name, "hair_style": "simple_cap"}


def add_facescape_augmentation(bpy, Vector, bbox_min, bbox_max, args: argparse.Namespace, asset: FaceScapeAsset) -> dict:
    rng = random.Random(args.augmentation_seed + asset.subject * 100 + asset.expression)
    metadata = {"synthetic_eyes": False, "synthetic_hair": False}
    if args.add_eyes:
        metadata.update(add_proxy_eyes(bpy, Vector, bbox_min, bbox_max, args.view_axis, args.eye_color, rng))
    if args.add_hair_cap:
        metadata.update(add_proxy_hair_cap(bpy, Vector, bbox_min, bbox_max, args.view_axis, args.hair_color, rng))
    return metadata


def root_objects(objects: list) -> list:
    object_set = set(objects)
    return [obj for obj in objects if obj.parent not in object_set]


def transform_roots(objects: list, transform) -> None:
    for obj in root_objects(objects):
        obj.matrix_world = transform @ obj.matrix_world


def mesh_bbox(bpy, Vector, objects: list):
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


def normalize_facescape(bpy, Matrix, Vector, objects: list, target_height: float, orientation: str):
    # FaceScape TU meshes are already Z-up in the released OBJ files. Keep rotation optional
    # because older or converted copies may appear with a different up axis.
    if orientation == "y-up-x90":
        transform_roots(objects, Matrix.Rotation(math.radians(90.0), 4, "X"))
    elif orientation == "y-up-neg-x90":
        transform_roots(objects, Matrix.Rotation(math.radians(-90.0), 4, "X"))
    elif orientation != "z-up":
        raise ValueError(f"Unsupported orientation: {orientation}")
    bpy.context.view_layer.update()
    bbox_min, bbox_max = mesh_bbox(bpy, Vector, objects)
    size = bbox_max - bbox_min
    scale = target_height / max(size.z, 1e-6)
    center_xy = Vector(((bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5, 0.0))
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(Vector((-center_xy.x, -center_xy.y, -bbox_min.z)))
    transform_roots(objects, transform)
    bpy.context.view_layer.update()
    return mesh_bbox(bpy, Vector, objects)


def look_at(obj, target) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_cycles(bpy, device: str, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    if device == "cpu":
        scene.cycles.device = "CPU"
        return
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        if device in {"cuda", "auto"}:
            prefs.compute_device_type = "CUDA"
        elif device == "optix":
            prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        enabled = 0
        for cycles_device in prefs.devices:
            use_device = cycles_device.type != "CPU"
            cycles_device.use = use_device
            enabled += int(use_device)
        if enabled:
            scene.cycles.device = "GPU"
            print(f"[FaceScapePreview] Cycles GPU devices enabled: {enabled}")
        else:
            scene.cycles.device = "CPU"
            print("[FaceScapePreview] No Cycles GPU devices found; using CPU")
    except Exception as exc:
        scene.cycles.device = "CPU"
        print(f"[FaceScapePreview] Could not enable GPU rendering; using CPU: {exc}")


def camera_position(Vector, target, bbox_min, bbox_max, span: float, view_axis: str, distance: float):
    if view_axis == "pos-y":
        return Vector((target.x, bbox_max.y + span * distance, target.z))
    if view_axis == "neg-y":
        return Vector((target.x, bbox_min.y - span * distance, target.z))
    if view_axis == "pos-x":
        return Vector((bbox_max.x + span * distance, target.y, target.z))
    if view_axis == "neg-x":
        return Vector((bbox_min.x - span * distance, target.y, target.z))
    raise ValueError(f"Unsupported view axis: {view_axis}")


def view_side_sign(view_axis: str) -> tuple[float, float]:
    if view_axis == "pos-y":
        return (0.0, 1.0)
    if view_axis == "neg-y":
        return (0.0, -1.0)
    if view_axis == "pos-x":
        return (1.0, 0.0)
    if view_axis == "neg-x":
        return (-1.0, 0.0)
    raise ValueError(f"Unsupported view axis: {view_axis}")


def setup_camera_lights(bpy, Vector, bbox_min, bbox_max, resolution: int, front_distance: float, view_axis: str) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    center = (bbox_min + bbox_max) * 0.5
    size = bbox_max - bbox_min
    span = max(size.x, size.y, size.z, 1.0)
    target = Vector((center.x, center.y, bbox_min.z + size.z * 0.55))

    cam_data = bpy.data.cameras.new("FaceScapePreviewCamera")
    cam = bpy.data.objects.new("FaceScapePreviewCamera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(size.x, size.z) * 1.14
    cam.location = camera_position(Vector, target, bbox_min, bbox_max, span, view_axis, front_distance)
    look_at(cam, target)
    scene.camera = cam

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.74, 0.74, 0.74)

    side_x, side_y = view_side_sign(view_axis)

    key_data = bpy.data.lights.new("FaceScapeKey", type="AREA")
    key = bpy.data.objects.new("FaceScapeKey", key_data)
    bpy.context.collection.objects.link(key)
    key.location = Vector((target.x - span * 0.6 + side_x * span * 0.5, target.y + side_y * span * 1.1, target.z + span * 0.8))
    key_data.energy = 430
    key_data.size = max(span * 1.5, 1.5)
    look_at(key, target)

    fill_data = bpy.data.lights.new("FaceScapeFill", type="AREA")
    fill = bpy.data.objects.new("FaceScapeFill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = Vector((target.x + span * 0.7 + side_x * span * 0.25, target.y + side_y * span * 0.5, target.z + span * 0.25))
    fill_data.energy = 85
    fill_data.size = max(span * 2.0, 1.8)
    look_at(fill, target)


def render_png(bpy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)


def safe_preview_name(asset: FaceScapeAsset) -> str:
    expression = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in asset.expression_name)
    return f"facescape_tu_s{asset.subject:03d}_e{asset.expression:02d}_{expression}"


def render_asset(bpy, Matrix, Vector, asset: FaceScapeAsset, preview_path: Path, args: argparse.Namespace) -> dict:
    clear_scene(bpy)
    objects = import_obj(bpy, asset.obj)
    if asset.texture and not objects_have_image_texture(objects):
        apply_texture_material(bpy, objects, asset.texture)
    bbox_min, bbox_max = normalize_facescape(bpy, Matrix, Vector, objects, args.target_height, args.orientation)
    augmentation_metadata = add_facescape_augmentation(bpy, Vector, bbox_min, bbox_max, args, asset)
    setup_camera_lights(bpy, Vector, bbox_min, bbox_max, args.resolution, args.front_distance, args.view_axis)
    render_png(bpy, preview_path)
    return {
        "bbox_min_preview_space": [float(bbox_min.x), float(bbox_min.y), float(bbox_min.z)],
        "bbox_max_preview_space": [float(bbox_max.x), float(bbox_max.y), float(bbox_max.z)],
        **augmentation_metadata,
    }


def main() -> int:
    args = parse_args()
    bpy, Matrix, Vector = import_blender_modules()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    index_out = Path(args.index_out).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"FaceScape TU root does not exist: {root}")

    assets = collect_assets(root, args.subject_start, args.subject_end, args.expressions)
    if args.limit is not None:
        assets = assets[: args.limit]
    if not assets:
        raise SystemExit(f"No FaceScape TU assets found under {root}")

    configure_cycles(bpy, args.device, args.samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[FaceScapePreview] Root: {root}")
    print(f"[FaceScapePreview] Subjects: {args.subject_start}-{args.subject_end}")
    print(f"[FaceScapePreview] Expressions: {','.join(map(str, args.expressions))}")
    print(f"[FaceScapePreview] Orientation: {args.orientation}")
    print(f"[FaceScapePreview] View axis: {args.view_axis}")
    print(f"[FaceScapePreview] Synthetic eyes: {args.add_eyes}")
    print(f"[FaceScapePreview] Synthetic hair cap: {args.add_hair_cap}")
    print(f"[FaceScapePreview] Assets: {len(assets)}")
    print(f"[FaceScapePreview] Output: {out_dir}")

    index_items = []
    for idx, asset in enumerate(assets, start=1):
        item_id = safe_preview_name(asset)
        preview_path = out_dir / f"{item_id}.png"
        metadata_path = metadata_dir / f"{item_id}.json"
        print(f"[FaceScapePreview] {idx}/{len(assets)} subject={asset.subject} expr={asset.expression_name}")
        status = "ok"
        error = None
        extra_metadata: dict = {}
        if preview_path.exists() and not args.overwrite:
            print(f"[FaceScapePreview] Skip existing: {preview_path}")
        else:
            try:
                extra_metadata = render_asset(bpy, Matrix, Vector, asset, preview_path, args)
            except Exception as exc:
                status = "failed"
                error = str(exc)
                print(f"[FaceScapePreview] Failed: {asset.obj}: {exc}")

        item_metadata = {
            "id": item_id,
            "dataset": "facescape_tu",
            "subject": asset.subject,
            "expression": asset.expression,
            "expression_name": asset.expression_name,
            "obj": str(asset.obj),
            "mtl": str(asset.mtl) if asset.mtl else None,
            "texture": str(asset.texture) if asset.texture else None,
            "dpmap": str(asset.dpmap) if asset.dpmap else None,
            "preview": str(preview_path),
            "status": status,
            "error": error,
            "orientation": args.orientation,
            "view_axis": args.view_axis,
            **extra_metadata,
        }
        metadata_path.write_text(json.dumps(item_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        index_items.append(
            {
                "id": item_id,
                "subject": asset.subject,
                "expression": asset.expression,
                "metadata": str(metadata_path),
                "preview": str(preview_path),
                "status": status,
            }
        )

    index_out.write_text(
        json.dumps(
            {
                "dataset": "facescape_tu",
                "root": str(root),
                "subject_start": args.subject_start,
                "subject_end": args.subject_end,
                "expressions": args.expressions,
                "orientation": args.orientation,
                "view_axis": args.view_axis,
                "items": index_items,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[FaceScapePreview] Wrote index: {index_out}")
    print(f"[FaceScapePreview] Wrote metadata: {metadata_dir}")
    print(f"[FaceScapePreview] Wrote previews: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
