import argparse
import json
import math
import os
import shutil
import sys

import bpy
from mathutils import Vector


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_ENVMAP_DIR = os.path.join(PROJECT_DIR, "data", "envmap")
FALLBACK_ENVMAP_DIR = os.path.join(PROJECT_DIR, "assets", "envmap")
SUPPORTED_ENVMAP_EXTS = (".exr", ".hdr")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def resolve_path(path, base_dir=None):
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    if base_dir is not None:
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(path)


def resolve_envmap_dir(path):
    envmap_dir = resolve_path(path)
    if os.path.isdir(envmap_dir):
        return envmap_dir

    if path == DEFAULT_ENVMAP_DIR and os.path.isdir(FALLBACK_ENVMAP_DIR):
        return FALLBACK_ENVMAP_DIR

    return envmap_dir


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def compute_bounds(objs):
    bpy.context.view_layer.update()

    min_corner = Vector((1e9, 1e9, 1e9))
    max_corner = Vector((-1e9, -1e9, -1e9))

    for obj in objs:
        if obj.type != "MESH":
            continue

        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, wc.x)
            min_corner.y = min(min_corner.y, wc.y)
            min_corner.z = min(min_corner.z, wc.z)

            max_corner.x = max(max_corner.x, wc.x)
            max_corner.y = max(max_corner.y, wc.y)
            max_corner.z = max(max_corner.z, wc.z)

    center = (min_corner + max_corner) / 2.0
    size_vec = max_corner - min_corner
    max_size = max(size_vec.x, size_vec.y, size_vec.z)
    radius = size_vec.length / 2.0

    return min_corner, max_corner, center, max_size, radius


def remove_lights_only():
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT" or obj.name.startswith("EMISSIVE_PROXY"):
            bpy.data.objects.remove(obj, do_unlink=True)


def render_image(path, width, height, file_format="PNG"):
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = path

    if file_format.upper() == "PNG":
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "16"
        scene.render.image_settings.compression = 15

    elif file_format.upper() == "EXR":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"

    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    bpy.ops.render.render(write_still=True)


def load_object_setup_helpers():
    try:
        from render_relighting_pairs import (  # noqa: E402
            create_reflection_panels,
            create_source_lighting,
            create_studio_floor,
            improve_materials,
            load_asset,
            normalize_objects,
            setup_camera,
            setup_cycles,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "--object_setup requires scripts/render_relighting_pairs.py "
            "to exist next to this script."
        ) from exc

    return {
        "create_reflection_panels": create_reflection_panels,
        "create_source_lighting": create_source_lighting,
        "create_studio_floor": create_studio_floor,
        "improve_materials": improve_materials,
        "load_asset": load_asset,
        "normalize_objects": normalize_objects,
        "setup_camera": setup_camera,
        "setup_cycles": setup_cycles,
    }


def sanitize_name(name):
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("_")
    return cleaned or "envmap"


def find_envmaps(envmap_dir, recursive=False):
    paths = []

    if recursive:
        for root, _, files in os.walk(envmap_dir):
            for filename in files:
                path = os.path.join(root, filename)
                if os.path.splitext(filename)[1].lower() in SUPPORTED_ENVMAP_EXTS:
                    paths.append(path)
    else:
        for filename in os.listdir(envmap_dir):
            path = os.path.join(envmap_dir, filename)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(filename)[1].lower() in SUPPORTED_ENVMAP_EXTS:
                paths.append(path)

    return sorted(paths, key=lambda p: os.path.relpath(p, envmap_dir).lower())


def find_blend_assets(asset_dir, recursive=False):
    paths = []

    if recursive:
        for root, _, files in os.walk(asset_dir):
            for filename in files:
                if os.path.splitext(filename)[1].lower() == ".blend":
                    paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(asset_dir):
            path = os.path.join(asset_dir, filename)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(filename)[1].lower() == ".blend":
                paths.append(path)

    return sorted(paths, key=lambda p: os.path.relpath(p, asset_dir).lower())


def asset_output_name(asset_path):
    stem = os.path.splitext(os.path.basename(asset_path))[0]
    return sanitize_name(stem)


def open_scene_asset(asset_path):
    bpy.ops.wm.open_mainfile(filepath=asset_path)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found in scene.")
    return meshes


def get_scene_camera():
    scene = bpy.context.scene

    if scene.camera is not None:
        return scene.camera

    cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]
    if cameras:
        scene.camera = cameras[0]
        return cameras[0]

    raise RuntimeError("No camera found in scene.")


def blend_has_camera(asset_path):
    current_file = bpy.data.filepath
    bpy.ops.wm.open_mainfile(filepath=asset_path)
    has_camera = any(obj.type == "CAMERA" for obj in bpy.context.scene.objects)
    if current_file:
        bpy.ops.wm.open_mainfile(filepath=current_file)
    else:
        clear_scene()
    return has_camera


def apply_render_size(width, height):
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height


def safe_set_enum(obj, attr, value):
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def setup_gpu_cycles():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences

        for device_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()

                has_gpu = False
                for dev in prefs.devices:
                    if dev.type != "CPU":
                        dev.use = True
                        has_gpu = True
                    else:
                        dev.use = False

                if has_gpu:
                    bpy.context.scene.cycles.device = "GPU"
                    print(f"[INFO] Cycles GPU enabled: {device_type}")
                    return True
            except Exception:
                continue

        print("[WARN] No GPU found. Falling back to CPU.")
        bpy.context.scene.cycles.device = "CPU"
        return False

    except Exception as exc:
        print(f"[WARN] GPU setup failed: {exc}")
        bpy.context.scene.cycles.device = "CPU"
        return False


def apply_denoiser_settings(denoiser="OPENIMAGEDENOISE", denoising_device="CPU"):
    scene = bpy.context.scene
    if not hasattr(scene, "cycles"):
        return

    denoiser = denoiser.upper()
    if denoiser == "NONE":
        scene.cycles.use_denoising = False
        scene.cycles.use_preview_denoising = False
        print("[INFO] Cycles denoiser disabled.")
        return

    scene.cycles.use_denoising = True
    if not safe_set_enum(scene.cycles, "denoiser", denoiser):
        print(f"[WARN] Denoiser {denoiser} is unavailable. Keeping {scene.cycles.denoiser}.")

    use_gpu_denoising = denoising_device.upper() == "GPU"
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = use_gpu_denoising
    if hasattr(scene.cycles, "preview_denoising_use_gpu"):
        scene.cycles.preview_denoising_use_gpu = use_gpu_denoising

    print(
        "[INFO] Cycles denoiser: "
        f"{scene.cycles.denoiser}, device={denoising_device.upper()}"
    )


def apply_cycles_settings(samples, device="GPU", denoiser="OPENIMAGEDENOISE", denoising_device="CPU"):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    if not hasattr(scene, "cycles"):
        return

    if samples is not None:
        scene.cycles.samples = samples
        scene.cycles.preview_samples = min(256, samples)

    if device.upper() == "GPU":
        setup_gpu_cycles()
    else:
        scene.cycles.device = "CPU"
        print("[INFO] Cycles CPU enabled.")

    apply_denoiser_settings(denoiser, denoising_device)


def parse_render_resolution(resolution_values, width, height):
    if len(resolution_values) == 1:
        render_width = resolution_values[0]
        render_height = resolution_values[0]
    elif len(resolution_values) == 2:
        render_width, render_height = resolution_values
    else:
        raise ValueError("--resolution expects one value or two values: WIDTH [HEIGHT]")

    if width is not None:
        render_width = width
    if height is not None:
        render_height = height

    if render_width <= 0 or render_height <= 0:
        raise ValueError("Render width and height must be positive.")

    return render_width, render_height


def save_light_visibility():
    return [
        (obj, obj.hide_render)
        for obj in bpy.context.scene.objects
        if obj.type == "LIGHT"
    ]


def set_lights_hidden(hidden):
    for obj in bpy.context.scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = hidden


def restore_light_visibility(states):
    for obj, hide_render in states:
        if bpy.data.objects.get(obj.name) is obj:
            obj.hide_render = hide_render


def set_world_envmap(envmap_path, strength=1.0, rotation_degrees=0.0):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputWorld")
    bg = nodes.new(type="ShaderNodeBackground")
    tex = nodes.new(type="ShaderNodeTexEnvironment")

    tex.image = bpy.data.images.load(envmap_path, check_existing=True)
    tex.projection = "EQUIRECTANGULAR"
    bg.inputs["Strength"].default_value = strength

    if rotation_degrees:
        coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_degrees)
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

    links.new(tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def camera_view_yaw_degrees(cam):
    forward = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    horizontal = Vector((forward.x, forward.y, 0.0))

    if horizontal.length == 0:
        return 0.0

    horizontal.normalize()
    return math.degrees(math.atan2(horizontal.x, horizontal.y))


def compute_env_rotation(cam, manual_offset_degrees, align_to_camera=True):
    camera_yaw = camera_view_yaw_degrees(cam)

    if align_to_camera:
        applied_rotation = camera_yaw + manual_offset_degrees
    else:
        applied_rotation = manual_offset_degrees

    return {
        "align_to_camera": align_to_camera,
        "camera_view_yaw_degrees": camera_yaw,
        "manual_offset_degrees": manual_offset_degrees,
        "applied_rotation_degrees": applied_rotation,
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def camera_metadata(cam):
    return {
        "location": list(cam.location),
        "rotation_euler": list(cam.rotation_euler),
        "lens": cam.data.lens,
    }


def bbox_metadata(objs):
    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)
    return {
        "min": list(min_corner),
        "max": list(max_corner),
        "center": list(center),
        "max_size": max_size,
        "radius": radius,
    }


def render_source(args, objs, source_envmap_path):
    source_path = os.path.join(args.out_dir, "source_image.png")

    if source_envmap_path is None:
        if args.object_setup:
            helpers = load_object_setup_helpers()
            helpers["create_source_lighting"](objs)
            source_mode = "studio_lighting"
        else:
            source_mode = "scene_current"
        source_strength = None
    else:
        if args.object_setup:
            remove_lights_only()
        source_strength = (
            args.source_env_strength
            if args.source_env_strength is not None
            else args.env_strength
        )
        set_world_envmap(
            source_envmap_path,
            strength=source_strength,
            rotation_degrees=args.env_rotation["applied_rotation_degrees"],
        )
        source_mode = "envmap"

    render_image(source_path, args.render_width, args.render_height, file_format="PNG")

    return {
        "mode": source_mode,
        "path": source_path,
        "envmap": source_envmap_path,
        "env_strength": source_strength,
        "env_rotation": args.env_rotation,
    }


def copy_source_for_pair(source_path, pair_dir, copy_source_to_pairs):
    if not copy_source_to_pairs:
        return source_path

    pair_source_path = os.path.join(pair_dir, "source_image.png")
    if os.path.abspath(pair_source_path) != os.path.abspath(source_path):
        shutil.copy2(source_path, pair_source_path)
    return pair_source_path


def render_envmap_pairs(args, objs, cam, envmaps, source_info):
    bbox = bbox_metadata(objs)
    pairs = []
    light_states = save_light_visibility()

    if args.object_setup:
        remove_lights_only()
    elif args.disable_scene_lights:
        set_lights_hidden(True)

    for idx, envmap_path in enumerate(envmaps, start=1):
        env_stem = os.path.splitext(os.path.basename(envmap_path))[0]
        pair_name = f"{idx:04d}_{sanitize_name(env_stem)}"
        pair_dir = os.path.join(args.out_dir, pair_name)
        ensure_dir(pair_dir)

        pair_source_path = copy_source_for_pair(
            source_info["path"],
            pair_dir,
            args.copy_source_to_pairs,
        )

        set_world_envmap(
            envmap_path,
            strength=args.env_strength,
            rotation_degrees=args.env_rotation["applied_rotation_degrees"],
        )

        target_path = os.path.join(pair_dir, "target_env_render.png")
        render_image(target_path, args.render_width, args.render_height, file_format="PNG")

        metadata = {
            "asset_path": args.asset_path,
            "pair_name": pair_name,
            "source_image": pair_source_path,
            "source": source_info,
            "target_envmap": envmap_path,
            "target_env_render": target_path,
            "render_settings": {
                "engine": bpy.context.scene.render.engine,
                "resolution": args.resolution,
                "width": args.render_width,
                "height": args.render_height,
                "samples": args.samples,
                "device": args.device,
                "denoiser": args.denoiser,
                "denoising_device": args.denoising_device,
                "env_strength": args.env_strength,
                "env_rotation_degrees": args.env_rotation_degrees,
                "env_rotation": args.env_rotation,
                "target_size": args.target_size,
                "view": args.view,
                "object_setup": args.object_setup,
                "disable_scene_lights": args.disable_scene_lights,
                "copy_source_to_pairs": args.copy_source_to_pairs,
                "view_transform": bpy.context.scene.view_settings.view_transform,
                "look": bpy.context.scene.view_settings.look,
            },
            "camera": camera_metadata(cam),
            "bbox": bbox,
        }

        metadata_path = os.path.join(pair_dir, "metadata.json")
        write_json(metadata_path, metadata)

        pair_summary = {
            "pair_name": pair_name,
            "source_image": pair_source_path,
            "target_envmap": envmap_path,
            "target_env_render": target_path,
            "metadata": metadata_path,
        }
        pairs.append(pair_summary)
        print(f"[DONE] {pair_name}")

    if not args.object_setup:
        restore_light_visibility(light_states)

    return pairs


def render_one_asset(args, asset_path, out_dir, envmaps, source_envmap_path):
    run_args = argparse.Namespace(**vars(args))
    run_args.asset_path = asset_path
    run_args.out_dir = out_dir
    run_args.object_setup = args.object_setup

    ensure_dir(run_args.out_dir)

    ext = os.path.splitext(run_args.asset_path)[1].lower()
    if ext != ".blend":
        clear_scene()

    if ext == ".blend" and not run_args.object_setup and not blend_has_camera(run_args.asset_path):
        print("[INFO] .blend has no camera; using --object_setup automatically.")
        run_args.object_setup = True

    if ext == ".blend" and not run_args.object_setup:
        objs = open_scene_asset(run_args.asset_path)
        apply_render_size(run_args.render_width, run_args.render_height)
        apply_cycles_settings(
            run_args.samples,
            run_args.device,
            run_args.denoiser,
            run_args.denoising_device,
        )
        cam = get_scene_camera()
    else:
        run_args.object_setup = True
        helpers = load_object_setup_helpers()
        objs = helpers["load_asset"](run_args.asset_path)

        helpers["setup_cycles"](
            width=run_args.render_width,
            height=run_args.render_height,
            samples=run_args.samples if run_args.samples is not None else 2048,
            use_gpu=run_args.device == "GPU",
        )
        apply_denoiser_settings(run_args.denoiser, run_args.denoising_device)

        helpers["normalize_objects"](objs, target_size=run_args.target_size)

        helpers["improve_materials"](
            objs,
            add_bevel=not run_args.no_bevel,
            add_weighted_normal=not run_args.no_weighted_normal,
        )

        helpers["create_studio_floor"](objs)
        helpers["create_reflection_panels"](objs)
        cam = helpers["setup_camera"](objs, focal_length=75, view=run_args.view)

    run_args.env_rotation = compute_env_rotation(
        cam,
        manual_offset_degrees=run_args.env_rotation_degrees,
        align_to_camera=not run_args.no_align_env_to_camera,
    )

    print(f"[INFO] Rendering asset: {run_args.asset_path}")
    print(f"[INFO] Found {len(envmaps)} envmap(s).")
    print(
        "[INFO] Envmap rotation: "
        f"camera_yaw={run_args.env_rotation['camera_view_yaw_degrees']:.3f}, "
        f"offset={run_args.env_rotation['manual_offset_degrees']:.3f}, "
        f"applied={run_args.env_rotation['applied_rotation_degrees']:.3f}"
    )
    source_info = render_source(run_args, objs, source_envmap_path)
    pairs = render_envmap_pairs(run_args, objs, cam, envmaps, source_info)

    summary = {
        "asset_path": run_args.asset_path,
        "envmap_dir": run_args.envmap_dir,
        "source": source_info,
        "render_settings": {
            "engine": bpy.context.scene.render.engine,
            "resolution": run_args.resolution,
            "width": run_args.render_width,
            "height": run_args.render_height,
            "samples": run_args.samples,
            "device": run_args.device,
            "denoiser": run_args.denoiser,
            "denoising_device": run_args.denoising_device,
            "env_strength": run_args.env_strength,
            "env_rotation_degrees": run_args.env_rotation_degrees,
            "env_rotation": run_args.env_rotation,
            "target_size": run_args.target_size,
            "view": run_args.view,
            "object_setup": run_args.object_setup,
            "disable_scene_lights": run_args.disable_scene_lights,
            "copy_source_to_pairs": run_args.copy_source_to_pairs,
        },
        "camera": camera_metadata(cam),
        "pairs": pairs,
    }

    summary_path = os.path.join(run_args.out_dir, "summary.json")
    write_json(summary_path, summary)

    print(f"[ASSET DONE] {run_args.asset_path}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    asset_group = parser.add_mutually_exclusive_group(required=True)
    asset_group.add_argument("--asset_path")
    asset_group.add_argument(
        "--asset_dir",
        help="Render every .blend file in this directory. Each result is written to out_dir/<blend_file_stem>/.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--envmap_dir", default=DEFAULT_ENVMAP_DIR)

    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=[1536],
        metavar="SIZE",
        help="Render resolution. Use one value for square output or two values for WIDTH HEIGHT.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("GPU", "CPU"),
        default="GPU",
        help="Cycles render device.",
    )
    parser.add_argument(
        "--denoiser",
        choices=("OPENIMAGEDENOISE", "OPTIX", "NONE"),
        default="OPENIMAGEDENOISE",
        help="Cycles denoiser. OPENIMAGEDENOISE avoids OptiX denoiser allocation failures.",
    )
    parser.add_argument(
        "--denoising_device",
        choices=("CPU", "GPU"),
        default="CPU",
        help="Device used by the denoiser when supported. Rendering can still use --device GPU.",
    )
    parser.add_argument("--env_strength", type=float, default=1.0)
    parser.add_argument("--env_rotation_degrees", type=float, default=0.0)
    parser.add_argument("--no_align_env_to_camera", action="store_true")
    parser.add_argument("--target_size", type=float, default=3.2)
    parser.add_argument("--view", type=str, default="front_3q")
    parser.add_argument(
        "--object_setup",
        action="store_true",
        help="Use the old single-object setup: normalize asset, create studio floor/panels, and create a new camera.",
    )
    parser.add_argument(
        "--disable_scene_lights",
        action="store_true",
        help="Hide existing scene lights during target envmap renders. Default keeps original lights.",
    )

    parser.add_argument(
        "--source_envmap",
        default=None,
        help="Optional .exr/.hdr source lighting. Defaults to the current scene for .blend files.",
    )
    parser.add_argument(
        "--source_env_strength",
        type=float,
        default=None,
        help="Optional source envmap strength. Defaults to --env_strength.",
    )

    parser.add_argument("--recursive_envmaps", action="store_true")
    parser.add_argument(
        "--recursive_assets",
        action="store_true",
        help="When using --asset_dir, find .blend files recursively.",
    )
    parser.add_argument(
        "--max_assets",
        type=int,
        default=0,
        help="When using --asset_dir, render only the first N blend files. Default renders all.",
    )
    parser.add_argument("--max_envmaps", type=int, default=0)
    parser.add_argument(
        "--copy_source_to_pairs",
        action="store_true",
        help="Copy source_image.png into each pair directory. Default keeps one shared source at out_dir/source_image.png.",
    )
    parser.add_argument(
        "--no_copy_source",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no_bevel", action="store_true")
    parser.add_argument("--no_weighted_normal", action="store_true")

    args = parser.parse_args(argv)
    if args.no_copy_source:
        args.copy_source_to_pairs = False

    args.asset_path = resolve_path(args.asset_path) if args.asset_path else None
    args.asset_dir = resolve_path(args.asset_dir) if args.asset_dir else None
    args.out_dir = os.path.abspath(args.out_dir)
    args.envmap_dir = resolve_envmap_dir(args.envmap_dir)
    args.render_width, args.render_height = parse_render_resolution(
        args.resolution,
        args.width,
        args.height,
    )
    source_envmap_path = resolve_path(args.source_envmap, args.envmap_dir)

    if args.asset_path is not None and not os.path.exists(args.asset_path):
        raise FileNotFoundError(f"Asset not found: {args.asset_path}")
    if args.asset_dir is not None and not os.path.isdir(args.asset_dir):
        raise NotADirectoryError(f"Asset directory not found: {args.asset_dir}")
    if not os.path.isdir(args.envmap_dir):
        raise NotADirectoryError(f"Envmap directory not found: {args.envmap_dir}")
    if source_envmap_path is not None and not os.path.exists(source_envmap_path):
        raise FileNotFoundError(f"Source envmap not found: {source_envmap_path}")

    envmaps = find_envmaps(args.envmap_dir, recursive=args.recursive_envmaps)
    total_envmaps = len(envmaps)
    if args.max_envmaps > 0:
        print(
            f"[INFO] Limiting envmaps: rendering first {args.max_envmaps} "
            f"of {total_envmaps}. Remove --max_envmaps to render all."
        )
        envmaps = envmaps[: args.max_envmaps]
    if not envmaps:
        raise RuntimeError(
            f"No envmaps found in {args.envmap_dir}. "
            f"Put .hdr/.exr files there or pass --envmap_dir. "
            f"Supported extensions: {', '.join(SUPPORTED_ENVMAP_EXTS)}"
        )

    if args.asset_path is not None:
        summaries = [
            render_one_asset(
                args,
                asset_path=args.asset_path,
                out_dir=args.out_dir,
                envmaps=envmaps,
                source_envmap_path=source_envmap_path,
            )
        ]
    else:
        ensure_dir(args.out_dir)
        asset_paths = find_blend_assets(args.asset_dir, recursive=args.recursive_assets)
        total_assets = len(asset_paths)
        if args.max_assets > 0:
            print(
                f"[INFO] Limiting assets: rendering first {args.max_assets} "
                f"of {total_assets}. Remove --max_assets to render all."
            )
            asset_paths = asset_paths[: args.max_assets]
        if not asset_paths:
            raise RuntimeError(f"No .blend assets found in {args.asset_dir}.")

        summaries = []
        print(f"[INFO] Found {len(asset_paths)} blend asset(s).")
        for idx, asset_path in enumerate(asset_paths, start=1):
            asset_out_dir = os.path.join(args.out_dir, asset_output_name(asset_path))
            print(f"[ASSET {idx}/{len(asset_paths)}] {asset_path}")
            summaries.append(
                render_one_asset(
                    args,
                    asset_path=asset_path,
                    out_dir=asset_out_dir,
                    envmaps=envmaps,
                    source_envmap_path=source_envmap_path,
                )
            )

        batch_summary = {
            "asset_dir": args.asset_dir,
            "out_dir": args.out_dir,
            "envmap_dir": args.envmap_dir,
            "asset_count": len(asset_paths),
            "envmap_count": len(envmaps),
            "assets": [
                {
                    "asset_path": summary["asset_path"],
                    "out_dir": os.path.dirname(summary["source"]["path"]),
                    "summary": os.path.join(
                        os.path.dirname(summary["source"]["path"]),
                        "summary.json",
                    ),
                    "pair_count": len(summary["pairs"]),
                }
                for summary in summaries
            ],
        }
        write_json(os.path.join(args.out_dir, "batch_summary.json"), batch_summary)

    final_summary = {
        "mode": "single" if args.asset_path is not None else "batch",
        "out_dir": args.out_dir,
        "asset_count": len(summaries),
        "envmap_count": len(envmaps),
        "assets": [
            {
                "asset_path": summary["asset_path"],
                "out_dir": os.path.dirname(summary["source"]["path"]),
                "pair_count": len(summary["pairs"]),
            }
            for summary in summaries
        ],
    }

    print("[ALL DONE]")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    main(argv)
