#!/usr/bin/env python3
"""Lightweight material-shadow validation for sparse Objaverse scene lists.

The same file acts as a multi-GPU launcher under regular Python and as a
Blender worker when invoked with ``--worker``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIGHT_POSITIONS = [
    (-1.2, -1.2, 1.5),
    (1.2, -1.2, 1.5),
    (0.0, 0.9, 1.8),
]
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def command_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render only original-shadow, shadow-off, and opaque-geometry-shadow diagnostics "
            "for selected Objaverse assets."
        )
    )
    parser.add_argument("--gpus", nargs="+", default=None, help="Launcher mode GPU ids, e.g. 0 1 2 3.")
    parser.add_argument("--blender", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--object-manifest", default="metadata/objaverse_xl/front2000_objects.txt")
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--light-count", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--point-energy", type=float, default=1000.0)
    parser.add_argument("--object-size", type=float, default=0.9)
    parser.add_argument("--relative-loss-threshold", type=float, default=0.02)
    parser.add_argument("--normalized-loss-threshold", type=float, default=0.005)
    parser.add_argument("--min-shadow-coverage", type=float, default=0.05)
    parser.add_argument("--min-geometry-pixels", type=int, default=64)
    parser.add_argument("--save-debug", action="store_true", help="Save compact PNG diagnostics for every scene.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-scene-list", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-device", default="0", help=argparse.SUPPRESS)
    args = parser.parse_args(command_args())
    if not args.worker and not args.gpus:
        parser.error("launcher mode requires --gpus")
    if args.worker and not args.worker_scene_list:
        parser.error("worker mode requires --worker-scene-list")
    if args.resolution < 32 or args.samples < 1:
        parser.error("--resolution must be >= 32 and --samples must be positive")
    if args.point_energy <= 0.0 or args.object_size <= 0.0:
        parser.error("--point-energy and --object-size must be positive")
    if args.min_geometry_pixels < 1:
        parser.error("--min-geometry-pixels must be positive")
    return args


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_scene_ids(path: Path, manifest_count: int) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Scene list does not exist: {path}")
    result = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        token = line.strip().split()[0]
        try:
            index = int(token.split("_", 1)[1]) if token.startswith("scene_") else int(token)
        except (ValueError, IndexError) as exc:
            raise SystemExit(f"Invalid scene id in {path}: {token}") from exc
        if index < 0 or index >= manifest_count:
            raise SystemExit(f"Scene id is outside the {manifest_count}-row manifest: {token}")
        scene_id = f"scene_{index:06d}"
        if scene_id not in seen:
            result.append(scene_id)
            seen.add(scene_id)
    if not result:
        raise SystemExit(f"Scene list is empty: {path}")
    return result


def read_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise SystemExit(f"Object manifest does not exist: {path}")
    return [
        resolve_path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def split_evenly(values: list[str], count: int) -> list[list[str]]:
    worker_count = min(max(count, 1), len(values))
    base, remainder = divmod(len(values), worker_count)
    groups = []
    cursor = 0
    for index in range(worker_count):
        size = base + (1 if index < remainder else 0)
        groups.append(values[cursor : cursor + size])
        cursor += size
    return groups


def worker_command(args: argparse.Namespace, scene_list: Path, shard_output: Path) -> list[str]:
    command = shlex.split(args.blender) + [
        "-b",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--worker",
        "--worker-scene-list",
        str(scene_list),
        "--gpu-device",
        "0",
        "--object-manifest",
        str(resolve_path(args.object_manifest)),
        "--scene-list",
        str(resolve_path(args.scene_list)),
        "--output",
        str(shard_output),
        "--resolution",
        str(args.resolution),
        "--samples",
        str(args.samples),
        "--light-count",
        str(args.light_count),
        "--point-energy",
        str(args.point_energy),
        "--object-size",
        str(args.object_size),
        "--relative-loss-threshold",
        str(args.relative_loss_threshold),
        "--normalized-loss-threshold",
        str(args.normalized_loss_threshold),
        "--min-shadow-coverage",
        str(args.min_shadow_coverage),
        "--min-geometry-pixels",
        str(args.min_geometry_pixels),
    ]
    if args.save_debug:
        command.append("--save-debug")
    if args.no_resume:
        command.append("--no-resume")
    return command


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def aggregate(output: Path, shard_count: int, requested_count: int) -> dict[str, Any]:
    rows = []
    for shard_id in range(shard_count):
        rows.extend(load_json_lines(output / f"shard_{shard_id}" / "results.jsonl"))
    rows.sort(key=lambda row: row["scene_id"])
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    statuses = {
        status: [row["scene_id"] for row in rows if row.get("status") == status]
        for status in ("pass", "fail", "inconclusive", "error")
    }
    for status, scene_ids in statuses.items():
        write_lines(output / f"{status}_scenes.txt", scene_ids)
    summary = {
        "schema": "objaverse_shadow_material_validation_v1",
        "requested_scene_count": requested_count,
        "result_scene_count": len(rows),
        **{f"{status}_scene_count": len(scene_ids) for status, scene_ids in statuses.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def launcher_main(args: argparse.Namespace) -> int:
    manifest = read_manifest(resolve_path(args.object_manifest))
    scene_ids = read_scene_ids(resolve_path(args.scene_list), len(manifest))
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups = split_evenly(scene_ids, len(args.gpus))
    commands = []
    list_dir = output / "_scene_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    for shard_id, group in enumerate(groups):
        scene_list = list_dir / f"shard_{shard_id}.txt"
        write_lines(scene_list, group)
        shard_output = output / f"shard_{shard_id}"
        commands.append((shard_id, str(args.gpus[shard_id]), group, worker_command(args, scene_list, shard_output)))

    print(
        f"[shadow-material] scenes={len(scene_ids)} workers={len(commands)} "
        f"renders_per_scene={int(args.light_count) * 3} output={output}",
        flush=True,
    )
    for shard_id, gpu, group, command in commands:
        print(
            f"[shadow-material] shard_{shard_id} gpu={gpu} scenes={len(group)} "
            f"first={group[0]} last={group[-1]}\n  {shlex.join(command)}",
            flush=True,
        )
    if args.dry_run:
        return 0

    processes = []
    for shard_id, gpu, _group, command in commands:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(command, cwd=ROOT, env=environment)
        processes.append((shard_id, process))
        print(f"[shadow-material] started shard_{shard_id} pid={process.pid}", flush=True)
    return_codes = {}
    try:
        for shard_id, process in processes:
            return_codes[shard_id] = process.wait()
            print(f"[shadow-material] finished shard_{shard_id} rc={return_codes[shard_id]}", flush=True)
    except KeyboardInterrupt:
        for _shard_id, process in processes:
            if process.poll() is None:
                process.terminate()
        return 130

    summary = aggregate(output, len(groups), len(scene_ids))
    summary["worker_return_codes"] = return_codes
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if all(code == 0 for code in return_codes.values()) else 1


def glb_material_meta(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic, _version, _length = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF":
            return {"has_unlit": False, "alpha_modes": [], "parse_error": "not_glb"}
        json_length, chunk_type = struct.unpack("<I4s", handle.read(8))
        if chunk_type != b"JSON":
            return {"has_unlit": False, "alpha_modes": [], "parse_error": "missing_json_chunk"}
        data = json.loads(handle.read(json_length).decode("utf-8").rstrip("\x00 "))
    materials = data.get("materials", [])
    return {
        "has_unlit": any("KHR_materials_unlit" in material.get("extensions", {}) for material in materials),
        "alpha_modes": sorted({material.get("alphaMode", "OPAQUE") for material in materials}),
        "material_count": len(materials),
    }


def configure_cycles(bpy, gpu_device: str, resolution: int, samples: int) -> dict[str, Any]:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = False
    scene.cycles.seed = 7321
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        scene.cycles.device = "CPU"
        return {"backend": "CPU", "enabled": []}
    errors = []
    for backend in ("OPTIX", "CUDA"):
        try:
            cprefs = prefs.preferences
            cprefs.compute_device_type = backend
            cprefs.get_devices()
            gpu_devices = [device for device in cprefs.devices if str(device.type).upper() == backend]
            if not gpu_devices:
                continue
            index = min(max(int(gpu_device), 0), len(gpu_devices) - 1)
            selected = gpu_devices[index]
            for device in cprefs.devices:
                device.use = device == selected
            scene.cycles.device = "GPU"
            return {"backend": backend, "enabled": [selected.name]}
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    scene.cycles.device = "CPU"
    return {"backend": "CPU", "enabled": [], "errors": errors}


def look_at(obj, target, Vector) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def mesh_bounds(meshes, Vector) -> tuple[Any, Any]:
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for obj in meshes:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
    return minimum, maximum


def normalize_imported(imported, meshes, object_size: float, Matrix, Vector) -> dict[str, Any]:
    minimum, maximum = mesh_bounds(meshes, Vector)
    extent = maximum - minimum
    scale = float(object_size) / max(float(extent.x), float(extent.y), float(extent.z), 1.0e-8)
    center = (minimum + maximum) * 0.5
    offset = Vector((-center.x, -center.y, -minimum.z))
    imported_set = set(imported)
    roots = [obj for obj in imported if obj.parent not in imported_set]
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(offset)
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world
    return {
        "original_bbox_min": list(minimum),
        "original_bbox_max": list(maximum),
        "scale": scale,
    }


def make_material(bpy, name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 1.0
    return material


def setup_validation_scene(bpy, asset_path: Path, args: argparse.Namespace, Matrix, Vector):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(asset_path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB produced no mesh objects")
    transform_meta = normalize_imported(imported, meshes, args.object_size, Matrix, Vector)
    for obj in imported:
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 0.0, -0.005))
    receiver = bpy.context.object
    receiver.name = "ValidationReceiver"
    receiver.data.materials.append(make_material(bpy, "ValidationReceiverMaterial", (0.8, 0.8, 0.8, 1.0)))

    camera_data = bpy.data.cameras.new("ValidationCamera")
    camera = bpy.data.objects.new("ValidationCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, -3.2, 2.6)
    camera_data.lens = 48.0
    look_at(camera, (0.0, 0.15, 0.35), Vector)
    bpy.context.scene.camera = camera

    world = bpy.data.worlds.new("ValidationWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world
    return meshes, transform_meta


def create_light(bpy, location: tuple[float, float, float], energy: float):
    data = bpy.data.lights.new("ValidationPoint", type="POINT")
    data.energy = float(energy)
    data.color = (1.0, 1.0, 1.0)
    data.shadow_soft_size = 0.03
    light = bpy.data.objects.new("ValidationPoint", data)
    light.location = location
    bpy.context.collection.objects.link(light)
    return light


def render_rgb(bpy) -> np.ndarray:
    from array import array

    path = Path(f"/tmp/tokenlight_shadow_validation_{os.getpid()}.exr")
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        channel_count = int(image.channels)
        pixels = array("f", [0.0]) * (width * height * channel_count)
        image.pixels.foreach_get(pixels)
        values = np.asarray(pixels, dtype=np.float32).reshape(height, width, channel_count)
        return values[..., :3].copy()
    finally:
        bpy.data.images.remove(image)
        path.unlink(missing_ok=True)


def set_shadow_visibility(meshes, enabled: bool) -> None:
    for obj in meshes:
        if hasattr(obj, "visible_shadow"):
            obj.visible_shadow = bool(enabled)


def force_non_camera_opaque(bpy, meshes) -> None:
    materials = {material for obj in meshes for material in obj.data.materials if material is not None}
    for obj in meshes:
        if not obj.data.materials:
            obj.data.materials.append(make_material(bpy, f"Opaque_{obj.name}", (0.8, 0.8, 0.8, 1.0)))
    for material in materials:
        if not material.use_nodes:
            continue
        tree = material.node_tree
        output = next((node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
        if output is None:
            continue
        surface = output.inputs.get("Surface")
        if surface is None or not surface.is_linked:
            continue
        original = surface.links[0].from_socket
        tree.links.remove(surface.links[0])
        light_path = tree.nodes.new("ShaderNodeLightPath")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
        tree.links.new(diffuse.outputs["BSDF"], mix.inputs[1])
        tree.links.new(original, mix.inputs[2])
        tree.links.new(mix.outputs["Shader"], surface)


def luma(image: np.ndarray) -> np.ndarray:
    return np.maximum(image @ LUMA_WEIGHTS, 0.0)


def measure_light(actual: np.ndarray, off: np.ndarray, geometry: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    off_y = luma(off)
    actual_loss = np.maximum(off_y - luma(actual), 0.0)
    geometry_loss = np.maximum(off_y - luma(geometry), 0.0)
    scale = max(float(np.percentile(off_y, 99.0)), 1.0e-8)
    relative_geometry = geometry_loss / np.maximum(off_y, scale * 1.0e-6)
    normalized_geometry = geometry_loss / scale
    geometry_support = (
        (relative_geometry >= float(args.relative_loss_threshold))
        & (normalized_geometry >= float(args.normalized_loss_threshold))
    )
    relative_actual = actual_loss / np.maximum(off_y, scale * 1.0e-6)
    normalized_actual = actual_loss / scale
    actual_support = (
        (relative_actual >= float(args.relative_loss_threshold))
        & (normalized_actual >= float(args.normalized_loss_threshold))
    )
    geometry_pixels = int(geometry_support.sum())
    overlap = int((geometry_support & actual_support).sum())
    union = int((geometry_support | actual_support).sum())
    coverage = float(overlap / geometry_pixels) if geometry_pixels else 0.0
    iou = float(overlap / union) if union else 0.0
    energy_ratio = float(actual_loss[geometry_support].sum() / max(geometry_loss[geometry_support].sum(), 1.0e-8))
    eligible = geometry_pixels >= int(args.min_geometry_pixels)
    return {
        "eligible": eligible,
        "casts_shadow": bool(eligible and coverage >= float(args.min_shadow_coverage)),
        "geometry_shadow_pixels": geometry_pixels,
        "actual_shadow_pixels": int(actual_support.sum()),
        "overlap_pixels": overlap,
        "geometry_coverage": coverage,
        "shadow_iou": iou,
        "actual_to_geometry_loss_energy": energy_ratio,
        "off_p99_luminance": scale,
        "actual_relative_loss_p99_in_geometry": (
            float(np.percentile(relative_actual[geometry_support], 99.0)) if geometry_pixels else 0.0
        ),
        "_debug": {
            "actual": actual,
            "off": off,
            "geometry": geometry,
            "actual_support": actual_support,
            "geometry_support": geometry_support,
        },
    }


def save_png(bpy, path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.maximum(image, 0.0)
    mapped = mapped / (1.0 + mapped)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)
    alpha = np.ones((*mapped.shape[:2], 1), dtype=np.float32)
    rgba = np.concatenate([mapped.astype(np.float32), alpha], axis=2)
    data = bpy.data.images.new(
        f"ValidationDebug_{path.stem}", width=rgba.shape[1], height=rgba.shape[0], alpha=True
    )
    try:
        data.pixels.foreach_set(rgba.reshape(-1))
        data.filepath_raw = str(path)
        data.file_format = "PNG"
        data.save()
    finally:
        bpy.data.images.remove(data)


def save_mask_png(bpy, path: Path, mask: np.ndarray) -> None:
    gray = np.repeat(mask.astype(np.float32)[..., None], 3, axis=2)
    save_png(bpy, path, gray)


def save_debug(bpy, scene_dir: Path, light_id: int, measurement: dict[str, Any]) -> None:
    debug = measurement["_debug"]
    base = scene_dir / "debug" / f"light_{light_id:02d}"
    save_png(bpy, base / "actual_shadow.png", debug["actual"])
    save_png(bpy, base / "shadow_off.png", debug["off"])
    save_png(bpy, base / "opaque_geometry_shadow.png", debug["geometry"])
    save_mask_png(bpy, base / "actual_mask.png", debug["actual_support"])
    save_mask_png(bpy, base / "geometry_mask.png", debug["geometry_support"])


def validate_scene(bpy, asset_path: Path, scene_id: str, scene_dir: Path, args: argparse.Namespace, Matrix, Vector):
    meshes, transform_meta = setup_validation_scene(bpy, asset_path, args, Matrix, Vector)
    actual_renders = []
    off_renders = []
    lights_meta = []
    for light_id, location in enumerate(LIGHT_POSITIONS[: int(args.light_count)]):
        light = create_light(bpy, location, args.point_energy)
        set_shadow_visibility(meshes, True)
        actual_renders.append(render_rgb(bpy))
        set_shadow_visibility(meshes, False)
        off_renders.append(render_rgb(bpy))
        bpy.data.objects.remove(light, do_unlink=True)
        lights_meta.append({"light_id": light_id, "position": list(location)})

    force_non_camera_opaque(bpy, meshes)
    set_shadow_visibility(meshes, True)
    measurements = []
    for light_id, location in enumerate(LIGHT_POSITIONS[: int(args.light_count)]):
        light = create_light(bpy, location, args.point_energy)
        geometry_render = render_rgb(bpy)
        bpy.data.objects.remove(light, do_unlink=True)
        measurement = measure_light(actual_renders[light_id], off_renders[light_id], geometry_render, args)
        if args.save_debug:
            save_debug(bpy, scene_dir, light_id, measurement)
        measurement.pop("_debug")
        measurements.append({**lights_meta[light_id], **measurement})

    eligible = [row for row in measurements if row["eligible"]]
    if not eligible:
        status, reason = "inconclusive", "no_eligible_opaque_geometry_shadow"
    elif any(row["casts_shadow"] for row in eligible):
        status, reason = "pass", "original_material_casts_measurable_shadow"
    else:
        status, reason = "fail", "opaque_geometry_casts_shadow_but_original_material_does_not"
    return {
        "schema": "objaverse_shadow_material_scene_v1",
        "scene_id": scene_id,
        "status": status,
        "reason": reason,
        "object_path": str(asset_path),
        "material": glb_material_meta(asset_path),
        "transform": transform_meta,
        "light_count": len(measurements),
        "passing_light_count": sum(bool(row["casts_shadow"]) for row in eligible),
        "max_geometry_coverage": max((float(row["geometry_coverage"]) for row in eligible), default=0.0),
        "max_shadow_iou": max((float(row["shadow_iou"]) for row in eligible), default=0.0),
        "measurements": measurements,
        "thresholds": {
            "relative_loss": float(args.relative_loss_threshold),
            "normalized_loss": float(args.normalized_loss_threshold),
            "min_shadow_coverage": float(args.min_shadow_coverage),
            "min_geometry_pixels": int(args.min_geometry_pixels),
        },
    }


def worker_main(args: argparse.Namespace) -> int:
    import bpy
    from mathutils import Matrix, Vector

    manifest = read_manifest(resolve_path(args.object_manifest))
    scene_ids = read_scene_ids(Path(args.worker_scene_list), len(manifest))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gpu_meta = None
    results = []
    for index, scene_id in enumerate(scene_ids, start=1):
        scene_dir = output / "scenes" / scene_id
        meta_path = scene_dir / "meta.json"
        if not args.no_resume and meta_path.is_file():
            try:
                row = json.loads(meta_path.read_text(encoding="utf-8"))
                results.append(row)
                print(f"[{index}/{len(scene_ids)}] {scene_id} resumed status={row.get('status')}", flush=True)
                continue
            except (OSError, ValueError):
                pass
        try:
            scene_number = int(scene_id.split("_")[-1])
            asset_path = manifest[scene_number]
            if not asset_path.is_file():
                raise FileNotFoundError(asset_path)
            scene_dir.mkdir(parents=True, exist_ok=True)
            bpy.ops.wm.read_factory_settings(use_empty=True)
            gpu_meta = configure_cycles(bpy, args.gpu_device, args.resolution, args.samples)
            row = validate_scene(bpy, asset_path, scene_id, scene_dir, args, Matrix, Vector)
            row["render"] = {
                "resolution": int(args.resolution),
                "samples": int(args.samples),
                "point_energy": float(args.point_energy),
                "gpu": gpu_meta,
                "saved_debug": bool(args.save_debug),
            }
            meta_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print(
                f"[{index}/{len(scene_ids)}] {scene_id} status={row['status']} "
                f"coverage={row['max_geometry_coverage']:.4f}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "scene_id": scene_id,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            error_dir = output / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / f"{scene_id}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print(f"[{index}/{len(scene_ids)}] {scene_id} ERROR {row['reason']}", flush=True)
        results.append(row)

    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return 0


def main() -> int:
    args = parse_args()
    return worker_main(args) if args.worker else launcher_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
