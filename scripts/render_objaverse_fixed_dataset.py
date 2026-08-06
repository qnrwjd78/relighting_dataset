"""Render Objaverse datasets with the fixed 4x4x2 light grid and power set."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

MAX_POWER_VALUE = 1.2
DEFAULT_POWER_SCALE = 0.5
DEFAULT_POWER_VALUES = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20]
DEFAULT_AMBIENT_MANIFEST = "metadata/polyhaven_hdri/polyhaven_hdri_hdris.txt"
DEFAULT_RECEIVER_TEXTURE_MANIFEST = "metadata/polyhaven_textures/polyhaven_textures.json"
DEFAULT_SOURCE_COMPARISON_RESOLUTIONS = ["256x256", "512x512", "848x480"]

try:
    import bpy
    from mathutils import Vector
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("scripts/render_objaverse_fixed_dataset.py must be run by Blender Python.") from exc


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(
        description=(
            "Replay selected Objaverse scenes and render position/color/power lighting samples "
            "with final object/object-shadow/direct-lit masks."
        )
    )
    parser.add_argument("--scene-set", choices=["seen", "unseen"], default=None)
    parser.add_argument("--scene-config", default="configs/final_mask_dataset/selected_objaverse_seen_unseen.json")
    parser.add_argument("--source-root", default=None, help="Override source root from --scene-config.")
    parser.add_argument("--object-manifest", default=None, help="Render GLBs listed in this manifest without replay meta.")
    parser.add_argument(
        "--object-scene-list",
        default=None,
        help="Optional scene-id list selecting sparse rows from --object-manifest while preserving global scene ids.",
    )
    parser.add_argument("--object-offset", type=int, default=0)
    parser.add_argument("--object-limit", type=int, default=None)
    parser.add_argument("--base-config", default="configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", nargs="+", default=None, help="Optional scene ids to render.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--object-target-size",
        type=float,
        default=0.9,
        help="Maximum object bounding-box extent after size clamping (default: 0.9).",
    )
    parser.add_argument(
        "--object-min-size",
        type=float,
        default=0.6,
        help="Minimum object bounding-box extent after size clamping (default: 0.6).",
    )
    parser.add_argument(
        "--rig-reference-object-size",
        type=float,
        default=1.2,
        help="Reference object size recorded for the camera/light rig (default: 1.2).",
    )
    parser.add_argument(
        "--canonical-world-scale",
        type=float,
        default=0.75,
        help="Fixed camera/light canonical world scale (default: 0.75).",
    )
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default="both")
    parser.add_argument("--gpu-devices", default=None)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--positions-per-scene", type=int, default=64)
    parser.add_argument("--color-count", type=int, default=32)
    parser.add_argument("--power-values", nargs="+", type=float, default=DEFAULT_POWER_VALUES)
    parser.add_argument(
        "--fixed-upper-half-white-grid",
        action="store_true",
        help=(
            "Use only the fixed 4x4x2 upper-half cube grid, assign one of four fixed power values "
            "to each candidate, keep only valid candidates, and skip color/power sweeps."
        ),
    )
    parser.add_argument("--color-position-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--power-position-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--anchor-direct-lit-rank",
        type=int,
        default=10,
        help="Use this rank after sorting position samples by object direct-lit mask area descending for color/power anchor.",
    )
    parser.add_argument("--base-energy", type=float, default=500.0)
    parser.add_argument("--default-power-scale", type=float, default=DEFAULT_POWER_SCALE)
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--ambient-manifest", default=DEFAULT_AMBIENT_MANIFEST)
    parser.add_argument(
        "--receiver-texture-manifest",
        default=None,
        help="Override the base config receiver texture manifest.",
    )
    parser.add_argument(
        "--no-receiver-textures",
        action="store_true",
        help="Disable image textures on generated floor and wall receivers.",
    )
    parser.add_argument("--ambient-strength-range", nargs=2, type=float, default=[0.6, 1.0])
    parser.add_argument("--ambient-fallback-strength", type=float, default=0.8)
    parser.add_argument("--ambient-fallback-color", nargs=3, type=float, default=[0.78, 0.78, 0.78])
    parser.add_argument(
        "--source-comparison-resolutions",
        nargs="+",
        default=DEFAULT_SOURCE_COMPARISON_RESOLUTIONS,
        help="Extra ambient-only source comparison resolutions, formatted like 256x256.",
    )
    parser.add_argument("--grid-resolution", type=int, default=4)
    parser.add_argument("--max-refill-attempts", type=int, default=512)
    parser.add_argument("--p99-luminance-threshold", type=float, default=0.01)
    parser.add_argument("--nonzero-pixel-ratio-threshold", type=float, default=0.001)
    parser.add_argument("--nonzero-luminance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--shadow-threshold", type=float, default=0.1)
    parser.add_argument(
        "--shadow-mask-mode",
        choices=["photometric", "geometry-ray"],
        default="photometric",
        help="Build object shadows from white-render differences or object-only geometry rays.",
    )
    parser.add_argument(
        "--ambient-subtracted-shadow-ratio",
        action="store_true",
        help=(
            "Build shadow masks from the ambient-subtracted point-light shadow ratio instead of global loss magnitude."
        ),
    )
    parser.add_argument(
        "--shadow-support-threshold",
        type=float,
        default=0.001,
        help="Minimum unoccluded point response as a fraction of its receiver-region p99.",
    )
    parser.add_argument("--direct-lit-threshold", type=float, default=0.1)
    parser.add_argument(
        "--object-mask-erode-radius",
        type=int,
        default=1,
        help="Erode the object mask by this many pixels before direct-lit masking (default: 1).",
    )
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--pad-radius", type=int, default=16)
    parser.add_argument(
        "--shadow-loss-debug-position-id",
        type=int,
        default=None,
        help="Also render this accepted position with subject shadow casting disabled.",
    )
    parser.add_argument(
        "--shadow-loss-debug-top-k",
        type=int,
        default=0,
        help="Render shadow-off debug passes for the K accepted positions with the largest geometry shadow masks.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Resume an output shard by skipping scenes with a valid final meta.json.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if not args.object_manifest and not args.scene_set:
        parser.error("one of --object-manifest or --scene-set is required")
    too_bright = [value for value in args.power_values if value > MAX_POWER_VALUE]
    if too_bright:
        parser.error(f"--power-values must be <= {MAX_POWER_VALUE}; got {too_bright}")
    if args.default_power_scale < 0.0 or args.default_power_scale > MAX_POWER_VALUE:
        parser.error(f"--default-power-scale must be in [0, {MAX_POWER_VALUE}]; got {args.default_power_scale}")
    if args.object_target_size is not None and args.object_target_size <= 0.0:
        parser.error(f"--object-target-size must be positive; got {args.object_target_size}")
    if args.object_min_size <= 0.0:
        parser.error(f"--object-min-size must be positive; got {args.object_min_size}")
    if args.object_min_size > args.object_target_size:
        parser.error(
            f"--object-min-size must not exceed --object-target-size; "
            f"got {args.object_min_size} > {args.object_target_size}"
        )
    if args.rig_reference_object_size <= 0.0:
        parser.error(f"--rig-reference-object-size must be positive; got {args.rig_reference_object_size}")
    if args.canonical_world_scale <= 0.0:
        parser.error(f"--canonical-world-scale must be positive; got {args.canonical_world_scale}")
    if args.shadow_threshold < 0.0:
        parser.error(f"--shadow-threshold must be nonnegative; got {args.shadow_threshold}")
    if args.shadow_support_threshold < 0.0:
        parser.error(f"--shadow-support-threshold must be nonnegative; got {args.shadow_support_threshold}")
    if args.object_mask_erode_radius < 0:
        parser.error(
            f"--object-mask-erode-radius must be nonnegative; got {args.object_mask_erode_radius}"
        )
    if args.shadow_loss_debug_top_k < 0:
        parser.error("--shadow-loss-debug-top-k must be nonnegative")
    if args.shadow_loss_debug_position_id is not None and args.shadow_loss_debug_top_k > 0:
        parser.error("--shadow-loss-debug-position-id and --shadow-loss-debug-top-k are mutually exclusive")
    if int(args.anchor_direct_lit_rank) < 1:
        parser.error(f"--anchor-direct-lit-rank must be >= 1; got {args.anchor_direct_lit_rank}")
    if args.fixed_upper_half_white_grid and len(args.power_values) != 4:
        parser.error(
            "--fixed-upper-half-white-grid requires exactly four --power-values; "
            f"got {len(args.power_values)}"
        )
    lo, hi = args.ambient_strength_range
    if lo < 0.0 or hi < 0.0 or lo > hi:
        parser.error(f"--ambient-strength-range must be nonnegative and ordered; got {args.ambient_strength_range}")
    try:
        args.source_comparison_resolutions = [
            parse_resolution_pair(value) for value in args.source_comparison_resolutions
        ]
    except ValueError as exc:
        parser.error(str(exc))
    return args


def grid_v4_mask_config() -> dict:
    return {
        "effect_percentile": 95.0,
        "effect_threshold_fraction": 0.03,
        "lit_threshold_fraction": 0.1,
        "absolute_luminance_threshold": 1.0e-4,
        "hf_threshold_fraction": 0.2,
        "preserve_dilate_radius": 2,
        "possible_threshold_fraction": 0.03,
        "visibility_shadow_threshold": 0.70,
        "visibility_clear_threshold": 0.90,
        "contact_visibility_threshold": 0.45,
        "contact_object_dilate_radius": 9,
        "shadow_boundary_threshold_fraction": 0.20,
        "white_mode": "direct",
        "preserve_alpha_cutout": True,
        "eps": 1.0e-6,
    }


def configure_gpu_devices(gpu_devices: str | None) -> dict:
    if not gpu_devices:
        return {"requested": None, "selection": "default"}

    requested = str(gpu_devices).strip()
    tokens = [token.strip().lower() for token in requested.split(",") if token.strip()]
    if not tokens:
        return {"requested": requested, "selection": "default"}

    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return {"requested": requested, "selection": "ignored_non_cycles", "engine": scene.render.engine}

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        raise RuntimeError("Cycles preferences are not available; cannot select GPU devices.")

    cprefs = prefs.preferences
    want_all = "all" in tokens or "*" in tokens
    want_cpu = "cpu" in tokens
    errors = []

    for compute_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            cprefs.compute_device_type = compute_type
            cprefs.get_devices()
        except Exception as exc:
            errors.append(f"{compute_type}: {exc}")
            continue

        devices = list(cprefs.devices)
        backend_gpus = [
            device for device in devices if str(getattr(device, "type", "")).upper() == compute_type
        ]
        if not backend_gpus and not want_cpu:
            continue

        enabled = []
        gpu_enabled = False
        for device in devices:
            device_type = str(getattr(device, "type", "")).lower()
            is_cpu = device_type == "cpu"
            is_backend_gpu = str(getattr(device, "type", "")).upper() == compute_type
            gpu_index = backend_gpus.index(device) if device in backend_gpus else None
            name = str(getattr(device, "name", "")).lower()
            device_id = str(getattr(device, "id", "")).lower()

            use_device = False
            if is_cpu:
                use_device = want_cpu and not want_all
            elif is_backend_gpu:
                if want_all:
                    use_device = True
                else:
                    for token in tokens:
                        id_match = bool(
                            device_id and (token == device_id or (not token.isdigit() and token in device_id))
                        )
                        name_match = not token.isdigit() and token in name
                        if token != "cpu" and (token == str(gpu_index) or name_match or id_match):
                            use_device = True
                            break

            device.use = bool(use_device)
            if use_device:
                enabled.append(
                    {
                        "name": getattr(device, "name", ""),
                        "type": getattr(device, "type", ""),
                        "index": gpu_index,
                    }
                )
                gpu_enabled = gpu_enabled or not is_cpu

        if enabled:
            scene.cycles.device = "GPU" if gpu_enabled else "CPU"
            return {
                "requested": requested,
                "compute_device_type": compute_type,
                "scene_cycles_device": scene.cycles.device,
                "enabled": enabled,
            }

    raise RuntimeError(f"No matching Cycles devices for --gpu-devices={requested!r}. Tried: {'; '.join(errors)}")


def build_replay_config(meta: dict, args: argparse.Namespace) -> dict:
    render_meta = meta.get("render", {})
    resolution = args.resolution or render_meta.get("resolution", [640, 640])
    resolution_value = int(resolution) if isinstance(resolution, int) else [int(resolution[0]), int(resolution[1])]
    light_transport = dict(render_meta.get("light_transport", {}))
    light_transport.pop("mode", None)

    object_meta = meta.get("object", {})
    camera_meta = meta.get("camera", {})
    receiver_bounds = meta.get("spatial", {}).get("receiver_bounds", {})
    room_width = float(receiver_bounds.get("half_width", 12.0)) * 2.0
    back_y = float(receiver_bounds.get("back_y", 2.4))
    wall_height = float(receiver_bounds.get("wall_height", 10.0))
    object_orientation = object_meta.get("orientation") or {}

    return {
        "seed": 0,
        "render": {
            "engine": render_meta.get("engine", "CYCLES"),
            "samples": int(args.samples or render_meta.get("samples", 128)),
            "resolution": resolution_value,
            "component_format": args.component_format,
            "device": "GPU",
            "view_transform": "Standard",
            "look": "None",
            "exposure": 0.0,
            "gamma": 1.0,
            "direct_light_transport": light_transport,
        },
        "object": {
            "target_size": float(object_meta.get("target_size", 1.2)),
            "randomize_materials": False,
            "orientation_mode": str(object_orientation.get("mode", "keep")),
            "primitive_fallbacks": ["sphere"],
        },
        "camera": {
            "fov_degrees": float(camera_meta.get("fov_degrees", 39.6)),
            "mode": camera_meta.get("mode", "canonical_rig"),
        },
        "layout": {
            "ground": True,
            "wall_probability": 1.0,
            "ground_size": room_width,
            "ground_size_range": [room_width, room_width],
            "wall_distance": back_y,
            "wall_distance_range": [back_y, back_y],
            "wall_height": wall_height,
            "wall_height_range": [wall_height, wall_height],
            "randomize_receiver_material": False,
        },
        "canonical": meta.get("canonical")
        or {"center": [0.0, 0.0, 0.0], "position_range": {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1]}},
        "spatial": {"enabled": True},
        "relighting_masks": grid_v4_mask_config(),
        "_component_format": args.component_format,
        "_ambient_source": "hdri",
        "_point_light_mode": "component",
        "_hdri_mode": "on",
        "_pbr_white_shading_only": False,
        "_debug_preview_only": False,
        "_light_preview": False,
        "_render_pbr": False,
        "_soft_light_transport": False,
        "_render_pbr_white_shading": False,
        "_render_relighting_masks": True,
        "_runtime": {"objects": [], "hdris": [], "receiver_textures": [], "fixture_scenes": []},
    }


def create_camera_from_meta(relight, camera_meta: dict):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector(camera_meta["location"])
    cam_data.angle = math.radians(float(camera_meta.get("fov_degrees", 39.6)))
    similarity = camera_meta.get("similarity_transform")
    if similarity and similarity.get("axes"):
        right, forward, up = relight.similarity_axes_from_meta(similarity)
        relight.set_camera_axes(cam, right, up, forward)
    else:
        relight.look_at(cam, Vector(camera_meta["look_at"]))
    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()
    return cam


def parse_resolution_pair(value: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
    else:
        text = str(value).strip().lower().replace(",", "x")
        parts = [part for part in text.split("x") if part]
        if len(parts) != 2:
            raise ValueError(f"Invalid source comparison resolution '{value}', expected WIDTHxHEIGHT.")
        width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source comparison resolution '{value}', width/height must be positive.")
    return width, height


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_completed_scene_meta(output_root: Path, scene_id: str) -> dict | None:
    meta_path = output_root / "scenes" / scene_id / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = load_json(meta_path)
    except (OSError, ValueError, TypeError):
        return None
    if meta.get("scene_id") != scene_id:
        return None
    return meta


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def maybe_repo_relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def configure_receiver_textures(relight, config: dict, root: Path, args: argparse.Namespace) -> dict:
    return relight.configure_receiver_textures(
        config,
        root,
        manifest_override=args.receiver_texture_manifest,
        disabled=args.no_receiver_textures,
    )


def hdri_manifest_pool(relight, root: Path, args: argparse.Namespace) -> tuple[Path, list[str], list[str]]:
    manifest_path = resolve_path(root, args.ambient_manifest)
    listed_hdris = relight.load_path_lines(manifest_path, root) if manifest_path.exists() else []
    existing_hdris = [path for path in listed_hdris if Path(path).exists()]
    return manifest_path, listed_hdris, existing_hdris


def choose_hdri_path(rng: random.Random, hdris: list[str], exclude_path: str | None = None) -> str | None:
    if not hdris:
        return None
    if exclude_path:
        excluded = str(Path(exclude_path).resolve())
        candidates = [path for path in hdris if str(Path(path).resolve()) != excluded]
        if candidates:
            return rng.choice(candidates)
    return rng.choice(hdris)


def add_hdri_pool_meta(source: dict, root: Path, manifest_path: Path, listed_hdris: list[str], existing_hdris: list[str]) -> dict:
    source.update(
        {
            "manifest": maybe_repo_relative(root, manifest_path),
            "pool_count": len(existing_hdris),
            "manifest_entry_count": len(listed_hdris),
            "missing_manifest_entries": max(0, len(listed_hdris) - len(existing_hdris)),
        }
    )
    return source


def resolution_key(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def render_component_at_resolution(relight, scene_dir: Path, rel_base: str, config: dict, width: int, height: int) -> dict:
    render = bpy.context.scene.render
    original = (int(render.resolution_x), int(render.resolution_y), int(render.resolution_percentage))
    try:
        render.resolution_x = int(width)
        render.resolution_y = int(height)
        render.resolution_percentage = 100
        return relight.render_component(scene_dir, rel_base, config)
    finally:
        render.resolution_x, render.resolution_y, render.resolution_percentage = original


def component_meta_with_resolution(relight, output: dict, width: int, height: int) -> dict:
    meta = relight.component_meta(output)
    meta["resolution"] = [int(width), int(height)]
    return meta


def choose_and_set_scene_ambient(relight, root: Path, args: argparse.Namespace, scene_number: int) -> dict:
    ambient_rng = random.Random(int(args.seed) + int(scene_number) * 1009 + 7919)
    manifest_path, listed_hdris, existing_hdris = hdri_manifest_pool(relight, root, args)
    hdri_path = choose_hdri_path(ambient_rng, existing_hdris)
    strength_lo, strength_hi = [float(v) for v in args.ambient_strength_range]
    strength = ambient_rng.uniform(strength_lo, strength_hi) if hdri_path else float(args.ambient_fallback_strength)
    rotation_z = ambient_rng.random() * 2.0 * math.pi
    fallback_color = [float(c) for c in args.ambient_fallback_color[:3]]
    source = relight.set_hdri_world(hdri_path, strength, rotation_z, fallback_color)
    add_hdri_pool_meta(source, root, manifest_path, listed_hdris, existing_hdris)
    source["selection_policy"] = "one_random_hdri_per_scene_shared_by_all_samples"
    return source


def render_source_comparisons(
    relight,
    scene_dir: Path,
    config: dict,
    root: Path,
    args: argparse.Namespace,
    scene_number: int,
    ambient_source: dict,
) -> dict:
    fallback_color = [float(c) for c in args.ambient_fallback_color[:3]]
    comparison_strength = float(ambient_source.get("strength", args.ambient_fallback_strength))

    relight.remove_all_lights()
    constant_source = {
        "type": "constant",
        "color": fallback_color,
        "strength": comparison_strength,
        "selection_policy": "constant_ambient_reference_matching_source_strength",
    }

    manifest_path, listed_hdris, existing_hdris = hdri_manifest_pool(relight, root, args)
    alt_rng = random.Random(int(args.seed) + int(scene_number) * 1009 + 15485863)
    alt_path = choose_hdri_path(alt_rng, existing_hdris, str(ambient_source.get("path")) if ambient_source.get("path") else None)
    alt_rotation_z = alt_rng.random() * 2.0 * math.pi
    alt_source = relight.set_hdri_world(alt_path, comparison_strength, alt_rotation_z, fallback_color)
    add_hdri_pool_meta(alt_source, root, manifest_path, listed_hdris, existing_hdris)
    alt_source.update(
        {
            "excluded_source_path": ambient_source.get("path"),
            "selection_policy": "one_random_hdri_different_from_source_if_possible_matching_source_strength",
        }
    )

    renders = {}
    for width, height in args.source_comparison_resolutions:
        key = resolution_key(width, height)

        relight.set_ambient_source_from_meta(ambient_source, config)
        selected_output = render_component_at_resolution(relight, scene_dir, f"sources/{key}/source", config, width, height)

        relight.set_constant_world(tuple(fallback_color), comparison_strength)
        constant_output = render_component_at_resolution(
            relight,
            scene_dir,
            f"sources/{key}/source_constant",
            config,
            width,
            height,
        )

        relight.set_hdri_world(alt_path, comparison_strength, alt_rotation_z, fallback_color)
        alt_output = render_component_at_resolution(relight, scene_dir, f"sources/{key}/source_hdri_alt", config, width, height)

        renders[key] = {
            "resolution": [int(width), int(height)],
            "selected_hdri_ambient_only": component_meta_with_resolution(relight, selected_output, width, height),
            "constant_ambient_only": component_meta_with_resolution(relight, constant_output, width, height),
            "alt_hdri_ambient_only": component_meta_with_resolution(relight, alt_output, width, height),
        }

    relight.set_ambient_source_from_meta(ambient_source, config)
    return {
        "constant_ambient_source": constant_source,
        "alt_hdri_ambient_source": alt_source,
        "resolutions": [[int(width), int(height)] for width, height in args.source_comparison_resolutions],
        "renders": renders,
    }


def scene_token(value: str | int) -> str:
    text = str(value)
    if text.startswith("scene_"):
        return text
    return f"scene_{int(text):06d}"


def scenes_dir(source_root: Path) -> Path:
    return source_root / "scenes" if (source_root / "scenes").is_dir() else source_root


def selected_scene_specs(args: argparse.Namespace, root: Path) -> tuple[Path | None, list[dict]]:
    if args.object_manifest:
        manifest_path = resolve_path(root, args.object_manifest)
        assets = [
            resolve_path(root, line.strip())
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        specs = [
            {"scene_id": f"scene_{index:06d}", "asset_path": path}
            for index, path in enumerate(assets)
        ]
        if args.object_scene_list:
            scene_list_path = resolve_path(root, args.object_scene_list)
            requested = {
                scene_token(line.strip().split()[0])
                for line in scene_list_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            available = {spec["scene_id"] for spec in specs}
            missing = sorted(requested - available)
            if missing:
                raise RuntimeError(
                    f"Object scene list contains {len(missing)} ids outside the manifest; first={missing[0]}"
                )
            specs = [spec for spec in specs if spec["scene_id"] in requested]
        else:
            offset = max(int(args.object_offset), 0)
            specs = specs[offset:]
            if args.object_limit is not None:
                specs = specs[: max(int(args.object_limit), 0)]
        return None, specs

    config_path = resolve_path(root, args.scene_config)
    config = load_json(config_path)
    entry = config[args.scene_set]
    source_root = resolve_path(root, args.source_root or entry["source_root"])
    ids = [scene_token(value) for value in (args.scenes or entry["selected_scene_ids"])]
    if args.max_scenes is not None:
        ids = ids[: max(0, int(args.max_scenes))]
    return source_root, [{"scene_id": scene_id, "asset_path": None} for scene_id in ids]


def resolve_meta_asset(root: Path, source_meta: dict) -> Path | None:
    value = source_meta.get("object", {}).get("path")
    if not value:
        return None
    text = str(value)
    if text.startswith("/workspace/"):
        return (root / text[len("/workspace/") :]).resolve()
    path = Path(text)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def object_bbox_reject(point: Vector, bbox_min: Vector, bbox_max: Vector, radius: float, margin: float = 0.04) -> tuple[bool, str | None]:
    safety = float(radius) + float(margin)
    if (
        bbox_min.x - safety <= point.x <= bbox_max.x + safety
        and bbox_min.y - safety <= point.y <= bbox_max.y + safety
        and bbox_min.z - safety <= point.z <= bbox_max.z + safety
    ):
        return False, "inside_object_bbox"
    return True, None


def fixed_upper_half_grid_candidates(power_values: list[float], seed: int) -> list[dict]:
    xy_values = [-0.75, -0.25, 0.25, 0.75]
    z_values = [0.25, 0.75]
    powers = [float(value) for value in power_values for _ in range(8)]
    random.Random(int(seed) + 3242).shuffle(powers)
    candidates = []
    for position_id, (ix, x, iy, y, iz, z) in enumerate(
        (ix, x, iy, y, iz, z)
        for ix, x in enumerate(xy_values)
        for iy, y in enumerate(xy_values)
        for iz, z in enumerate(z_values)
    ):
        candidates.append(
            {
                "position_id": int(position_id),
                "canonical_position": [float(x), float(y), float(z)],
                "candidate_source": "fixed_upper_half_grid_4x4x2",
                "grid_cell": [int(ix), int(iy), int(iz)],
                "grid_resolution": [4, 4, 2],
                "power_scale": powers[position_id],
            }
        )
    return candidates


def hsv_palette(count: int) -> list[list[float]]:
    return [[float(c) for c in colorsys.hsv_to_rgb(i / float(max(count, 1)), 1.0, 1.0)] for i in range(count)]


def render_with_point_light(relight, scene_dir: Path, rel_base: str, config: dict, light: dict, color: list[float], energy_scale: float = 1.0) -> dict:
    light_obj = relight.create_point_light(
        f"TL_Final_{rel_base.replace('/', '_')}",
        Vector(light["world_position"]),
        float(light["world_energy"]) * float(energy_scale),
        float(light["world_radius"]),
        color,
    )
    try:
        return relight.render_component(scene_dir, rel_base, config)
    finally:
        bpy.data.objects.remove(light_obj, do_unlink=True)


def select_anchor_by_direct_lit_area(position_samples: list[dict], rank: int) -> tuple[int, dict, list[dict]]:
    if not position_samples:
        raise RuntimeError("Cannot select color/power anchor: no position samples.")
    ranked = sorted(
        enumerate(position_samples),
        key=lambda item: (
            -float(item[1].get("mask_stats", {}).get("object_direct_lit_clean_ratio", 0.0)),
            int(item[1].get("light", {}).get("id", item[0])),
        ),
    )
    rank_index = min(max(int(rank), 1) - 1, len(ranked) - 1)
    position_index, sample = ranked[rank_index]
    ranking = []
    for i, (sample_index, row) in enumerate(ranked):
        ranking.append(
            {
                "rank": i + 1,
                "position_index": int(sample_index),
                "light_id": int(row.get("light", {}).get("id", sample_index)),
                "object_direct_lit_clean_ratio": float(
                    row.get("mask_stats", {}).get("object_direct_lit_clean_ratio", 0.0)
                ),
            }
        )
    return position_index, sample, ranking


def choose_valid_lights(
    relight,
    scene_dir: Path,
    config: dict,
    rng: random.Random,
    camera,
    center: Vector,
    bbox_min: Vector,
    bbox_max: Vector,
    ambient_source: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    spatial = config["spatial"]
    valid_filter = relight.point_light_valid_filter_config(spatial)
    pr = config["canonical"]["position_range"]
    if args.fixed_upper_half_white_grid:
        initial_candidates = fixed_upper_half_grid_candidates(args.power_values, int(args.seed))
    else:
        initial_spatial = dict(spatial)
        initial_spatial["grid_resolution"] = int(args.grid_resolution)
        initial_spatial["jitter"] = 0.0
        initial_candidates = relight.sample_jittered_grid_candidates(
            pr,
            min(int(args.positions_per_scene), int(args.grid_resolution) ** 3),
            rng,
            initial_spatial,
            [],
            "grid_initial",
        )
    receiver_bounds = config["_runtime"].get("receiver_bounds")
    transform_meta = relight.canonical_transform_meta(config, camera, center)
    world_scale = float(transform_meta.get("light_world_scale", transform_meta["scale"]))
    final_positions: list[list[float]] = []
    lights: list[dict] = []
    attempts: list[dict] = []
    max_attempts = (
        len(initial_candidates)
        if args.fixed_upper_half_white_grid
        else int(args.positions_per_scene) + int(args.max_refill_attempts)
    )
    for attempt_index in range(max_attempts):
        if not args.fixed_upper_half_white_grid and len(lights) >= int(args.positions_per_scene):
            break
        if attempt_index < len(initial_candidates):
            candidate = initial_candidates[attempt_index]
        else:
            candidate = relight.sample_random_point_candidate(pr, rng, spatial, final_positions, "random_refill")
        p_can = [float(v) for v in candidate["canonical_position"]]
        power_scale = float(candidate.get("power_scale", args.default_power_scale))
        p_world = relight.canonical_to_world(p_can, camera, config, center)
        world_radius = float(args.radius) * world_scale
        geom_valid, skip_reason = relight.point_inside_receiver_bounds(p_world, receiver_bounds, world_radius)
        if geom_valid:
            geom_valid, skip_reason = object_bbox_reject(p_world, bbox_min, bbox_max, world_radius)
        output = None
        validation_stats = {"valid": False, "skip_reason": skip_reason or "geometry_rejected"}
        accepted = False
        attempt_base = f"validation/point_light_attempts/attempt_{attempt_index:04d}"
        if geom_valid:
            light_meta = {
                "world_position": [float(p_world.x), float(p_world.y), float(p_world.z)],
                "world_energy": float(args.base_energy) * world_scale * world_scale,
                "world_radius": world_radius,
            }
            try:
                relight.set_black_world()
                output = render_with_point_light(
                    relight,
                    scene_dir,
                    attempt_base,
                    config,
                    light_meta,
                    [1.0, 1.0, 1.0],
                    power_scale,
                )
                accepted, validation_stats = relight.validate_point_light_component(scene_dir, output, valid_filter)
            finally:
                relight.set_ambient_source_from_meta(ambient_source, config)
            skip_reason = None if accepted else str(validation_stats.get("skip_reason", "invalid_point_light"))
        if accepted:
            light_id = int(candidate.get("position_id", len(lights)))
            rel_base = f"samples/position/position_{light_id:03d}"
            final_output = render_with_point_light(
                relight,
                scene_dir,
                rel_base,
                config,
                light_meta,
                [1.0, 1.0, 1.0],
                power_scale,
            )
            light = {
                "id": light_id,
                "canonical_position": p_can,
                "world_position": [float(p_world.x), float(p_world.y), float(p_world.z)],
                "world_energy": float(args.base_energy) * world_scale * world_scale,
                "render_world_energy": float(args.base_energy) * world_scale * world_scale * power_scale,
                "power_scale": power_scale,
                "world_radius": world_radius,
                "canonical_energy": float(args.base_energy),
                "canonical_radius": float(args.radius),
                "render_color": [1.0, 1.0, 1.0],
                "component_color": [1.0, 1.0, 1.0],
                "candidate_source": candidate.get("candidate_source"),
                "grid_cell": candidate.get("grid_cell"),
                "grid_resolution": candidate.get("grid_resolution"),
                "validation": validation_stats,
            }
            light.update({"render": final_output["primary"], **final_output})
            lights.append(light)
            final_positions.append(p_can)
        if output:
            relight.remove_component_files(scene_dir, attempt_base, config)
        attempts.append(
            {
                "attempt_index": attempt_index,
                "candidate_source": candidate.get("candidate_source"),
                "grid_cell": candidate.get("grid_cell"),
                "grid_resolution": candidate.get("grid_resolution"),
                "position_id": candidate.get("position_id"),
                "power_scale": power_scale,
                "canonical_position": p_can,
                "world_position": [float(p_world.x), float(p_world.y), float(p_world.z)],
                "valid": accepted,
                "skip_reason": skip_reason,
                "validation": validation_stats,
            }
        )
    if not args.fixed_upper_half_white_grid and len(lights) < int(args.positions_per_scene):
        raise RuntimeError(f"Accepted {len(lights)}/{args.positions_per_scene} point lights")
    if args.fixed_upper_half_white_grid and not lights:
        raise RuntimeError("Accepted 0/32 fixed upper-half point lights")
    return lights, attempts


def render_scene(
    args: argparse.Namespace,
    source_root: Path | None,
    scene_id: str,
    output_root: Path,
    root: Path,
    asset_override: Path | None = None,
) -> dict:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import render_objaverse_random_dataset as relight
    import relighting_mask_pipeline as mask_pipeline

    rng = random.Random(int(args.seed) + int(scene_id.split("_")[-1]))
    if asset_override is not None:
        source_scene_dir = None
        asset_path = Path(asset_override).resolve()
        source_meta = {
            "object": {
                "path": str(asset_path),
                "primitive": None,
                "target_size": 1.2,
            }
        }
    else:
        if source_root is None:
            raise RuntimeError("source_root is required when no object manifest asset is provided")
        source_scene_dir = scenes_dir(source_root) / scene_id
        source_meta = load_json(source_scene_dir / "meta.json")
        asset_path = resolve_meta_asset(root, source_meta)
    if asset_path is None or not asset_path.exists():
        raise FileNotFoundError(f"Missing object asset for {scene_id}: {asset_path}")

    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    config_args = SimpleNamespace(
        resolution=int(args.resolution),
        samples=int(args.samples),
        component_format=args.component_format,
        mask_preset="grid-v4",
    )
    if asset_override is not None:
        config = load_json(resolve_path(root, args.base_config))
        config["render"].update(
            {
                "resolution": int(args.resolution),
                "resolution_x": int(args.resolution),
                "resolution_y": int(args.resolution),
                "samples": int(args.samples),
                "component_format": args.component_format,
            }
        )
        config["_runtime"] = {
            "objects": [],
            "hdris": [],
            "receiver_textures": [],
            "fixture_scenes": [],
        }
    else:
        config = build_replay_config(source_meta, config_args)
    relight.configure_object_rig_scale(
        config,
        args.object_min_size,
        args.object_target_size,
        args.rig_reference_object_size,
        args.canonical_world_scale,
    )
    source_meta.setdefault("object", {})["min_size"] = float(args.object_min_size)
    source_meta.setdefault("object", {})["target_size"] = float(args.object_target_size)
    config["_component_format"] = args.component_format
    config["_render_pbr"] = True
    config["_render_relighting_masks"] = False
    config["ambient"] = {
        "hdri_strength_range": [float(v) for v in args.ambient_strength_range],
        "hdri_rotation_random": True,
        "hdri_mode": "on",
        "hdri_probability": 1.0,
        "fallback_color": [float(c) for c in args.ambient_fallback_color[:3]],
        "fallback_strength": float(args.ambient_fallback_strength),
    }
    config["spatial"].update(
        {
            "positions_per_scene": 32 if args.fixed_upper_half_white_grid else int(args.positions_per_scene),
            "grid_resolution": int(args.grid_resolution),
            "jitter": 0.0,
            "receiver_bounds_filter": True,
            "base_energy": float(args.base_energy),
            "energy_range": [float(args.base_energy), float(args.base_energy)],
            "fixed_radius": float(args.radius),
            "radius_range": [float(args.radius), float(args.radius)],
            "color": [1.0, 1.0, 1.0],
            "valid_filter": {
                "enabled": True,
                "grid_resolution": int(args.grid_resolution),
                "p99_luminance_threshold": float(args.p99_luminance_threshold),
                "nonzero_pixel_ratio_threshold": float(args.nonzero_pixel_ratio_threshold),
                "nonzero_luminance_threshold": float(args.nonzero_luminance_threshold),
                "max_refill_attempts": int(args.max_refill_attempts),
                "keep_attempt_renders": False,
                "require_target_count": True,
            },
        }
    )
    receiver_texture_status = configure_receiver_textures(relight, config, root, args)

    relight.clear_scene()
    relight.setup_render_settings(config)
    gpu_meta = configure_gpu_devices(args.gpu_devices)

    subject_objects = relight.import_asset_or_primitive(str(asset_path), source_meta.get("object", {}).get("primitive") or "sphere", rng, config)
    bbox_min, bbox_max = relight.mesh_bbox(subject_objects)
    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)
    center = (bbox_min + bbox_max) * 0.5
    if asset_override is not None:
        camera, camera_meta = relight.create_camera(config, rng, center)
    else:
        camera_meta = source_meta["camera"]
        similarity = camera_meta.get("similarity_transform")
        if similarity:
            config["_runtime"]["similarity_transform"] = similarity
            config["_runtime"]["canonical_scale"] = float(
                similarity.get("scale", config["_runtime"].get("canonical_scale", 1.0))
            )
        camera = create_camera_from_meta(relight, camera_meta)
    relight.create_receivers(config, rng, camera, center)
    relight.remove_all_lights()
    relight.set_black_world()

    pbr_maps = relight.render_pbr_maps(scene_dir, config)
    object_mask_rel = relight.render_object_mask(scene_dir, subject_objects)
    receiver_masks = relight.render_receiver_masks(scene_dir)
    shape = (int(bpy.context.scene.render.resolution_y), int(bpy.context.scene.render.resolution_x))
    object_mask = mask_pipeline.load_png_mask(scene_dir / object_mask_rel, shape)
    receiver_mask = mask_pipeline.load_png_mask(scene_dir / receiver_masks["receiver"], shape)

    scene_number = int(scene_id.split("_")[-1])
    relight.remove_all_lights()
    ambient_source = choose_and_set_scene_ambient(relight, root, args, scene_number)
    source_output = relight.render_component(scene_dir, "source", config)
    source_comparisons = None
    if not args.fixed_upper_half_white_grid:
        source_comparisons = render_source_comparisons(
            relight,
            scene_dir,
            config,
            root,
            args,
            scene_number,
            ambient_source,
        )

    position_lights, candidate_attempts = choose_valid_lights(
        relight,
        scene_dir,
        config,
        rng,
        camera,
        center,
        bbox_min,
        bbox_max,
        ambient_source,
        args,
    )
    shadow_loss_debug = None
    ambient_white_reference = None
    if args.shadow_mask_mode == "photometric" and args.ambient_subtracted_shadow_ratio:
        ambient_white_reference = mask_pipeline.render_ambient_white_reference(
            relight,
            scene_dir,
            config,
            subject_objects,
            ambient_source,
        )
    samples = []
    for light in position_lights:
        rel_base = f"samples/position/position_{int(light['id']):03d}"
        mask_meta = mask_pipeline.render_white_pair_and_masks(
            relight,
            scene_dir,
            rel_base,
            config,
            camera,
            subject_objects,
            light,
            object_mask,
            receiver_mask,
            ambient_source,
            ambient_white_reference,
            args,
        )
        row = {
            "name": f"position_{int(light['id']):03d}",
            "task": "position",
            "light": light,
            "image": light["primary"],
            "masks": mask_meta["masks"],
            "mask_stats": mask_meta["mask_stats"],
        }
        samples.append(row)

    debug_lights = []
    if args.shadow_loss_debug_position_id is not None:
        debug_light = next(
            (
                light
                for light in position_lights
                if int(light["id"]) == int(args.shadow_loss_debug_position_id)
            ),
            None,
        )
        if debug_light is None:
            raise RuntimeError(
                f"Debug position_{int(args.shadow_loss_debug_position_id):03d} was not accepted"
            )
        debug_lights = [debug_light]
    elif args.shadow_loss_debug_top_k > 0:
        ranked_samples = sorted(
            samples,
            key=lambda row: (
                -float(row.get("mask_stats", {}).get("object_shadow_clean_ratio", 0.0)),
                int(row.get("light", {}).get("id", 0)),
            ),
        )
        debug_lights = [row["light"] for row in ranked_samples[: int(args.shadow_loss_debug_top_k)]]

    if debug_lights:
        debug_rows = []
        snapshot = mask_pipeline.mesh_shadow_snapshot(subject_objects)
        try:
            mask_pipeline.set_mesh_shadow_visibility(subject_objects, False)
            for debug_light in debug_lights:
                position_id = int(debug_light["id"])
                debug_output = render_with_point_light(
                    relight,
                    scene_dir,
                    f"debug_shadow_loss/position_{position_id:03d}_no_object_shadow",
                    config,
                    debug_light,
                    [1.0, 1.0, 1.0],
                    float(debug_light["power_scale"]),
                )
                debug_rows.append(
                    {
                        "position_id": position_id,
                        "geometry_shadow_ratio": float(
                            next(
                                row["mask_stats"]["object_shadow_clean_ratio"]
                                for row in samples
                                if int(row["light"]["id"]) == position_id
                            )
                        ),
                        "shadow_on": debug_light["primary"],
                        "shadow_off": debug_output["primary"],
                    }
                )
        finally:
            mask_pipeline.restore_mesh_shadow_visibility(snapshot)
        shadow_loss_debug = debug_rows[0] if args.shadow_loss_debug_position_id is not None else debug_rows

    anchor_sampling_meta = {}
    if not args.fixed_upper_half_white_grid:
        position_samples = [row for row in samples if row["task"] == "position"]
        anchor_position_index, anchor_sample, anchor_direct_lit_ranking = select_anchor_by_direct_lit_area(
            position_samples,
            int(args.anchor_direct_lit_rank),
        )
        anchor_masks = anchor_sample["masks"]
        anchor_rank_used = next(
            row["rank"] for row in anchor_direct_lit_ranking if row["position_index"] == int(anchor_position_index)
        )
        anchor_direct_lit_ratio = float(anchor_sample["mask_stats"]["object_direct_lit_clean_ratio"])

        colors = hsv_palette(int(args.color_count))
        color_anchor = position_lights[anchor_position_index]
        for color_index, color in enumerate(colors):
            rel_base = f"samples/color/color_{color_index:03d}"
            output = render_with_point_light(
                relight,
                scene_dir,
                rel_base,
                config,
                color_anchor,
                color,
                float(args.default_power_scale),
            )
            masks = mask_pipeline.copy_mask_bundle(scene_dir, anchor_masks, rel_base)
            samples.append(
                {
                    "name": f"color_{color_index:03d}",
                    "task": "color",
                    "image": output["primary"],
                    "output": output,
                    "anchor_position_light_id": int(color_anchor["id"]),
                    "anchor_position_index": int(anchor_position_index),
                    "anchor_direct_lit_rank": int(anchor_rank_used),
                    "anchor_object_direct_lit_clean_ratio": anchor_direct_lit_ratio,
                    "power_scale": float(args.default_power_scale),
                    "color": color,
                    "masks": masks,
                    "mask_source": "copied_from_anchor_position_normalized_masks",
                }
            )

        power_anchor = position_lights[anchor_position_index]
        for power_index, power in enumerate(args.power_values):
            rel_base = f"samples/power/power_{power_index:03d}"
            output = render_with_point_light(
                relight, scene_dir, rel_base, config, power_anchor, [1.0, 1.0, 1.0], float(power)
            )
            masks = mask_pipeline.copy_mask_bundle(scene_dir, anchor_masks, rel_base)
            samples.append(
                {
                    "name": f"power_{power_index:03d}",
                    "task": "power",
                    "image": output["primary"],
                    "output": output,
                    "anchor_position_light_id": int(power_anchor["id"]),
                    "anchor_position_index": int(anchor_position_index),
                    "anchor_direct_lit_rank": int(anchor_rank_used),
                    "anchor_object_direct_lit_clean_ratio": anchor_direct_lit_ratio,
                    "power_scale": float(power),
                    "masks": masks,
                    "mask_source": "copied_from_anchor_position_normalized_masks",
                }
            )
        anchor_sampling_meta = {
            "color_power_anchor_requested_rank": int(args.anchor_direct_lit_rank),
            "color_power_anchor_used_rank": int(anchor_rank_used),
            "color_power_anchor_position_index": int(anchor_position_index),
            "color_power_anchor_light_id": int(color_anchor["id"]),
            "color_power_anchor_object_direct_lit_clean_ratio": anchor_direct_lit_ratio,
            "color_power_anchor_direct_lit_ranking": anchor_direct_lit_ranking,
        }

    if args.fixed_upper_half_white_grid:
        sampling_meta = {
            "position_candidate_count": 32,
            "position_count": len(position_lights),
            "rejected_position_count": 32 - len(position_lights),
            "color_count": 0,
            "power_values": [float(v) for v in args.power_values],
            "base_energy": float(args.base_energy),
            "radius": float(args.radius),
            "position_policy": "fixed canonical upper-half 4x4x2 grid centers; reject invalid candidates; no refill",
            "position_axes": {
                "x": [-0.75, -0.25, 0.25, 0.75],
                "y": [-0.75, -0.25, 0.25, 0.75],
                "z": [0.25, 0.75],
            },
            "power_policy": "four fixed white-light power scales, balanced 8 each and seed-shuffled once across 32 positions",
            "color_policy": "fixed white RGB [1,1,1]",
            "ambient_policy": "one random HDRI per scene shared by source, every accepted point-light target, and every mask render",
        }
    else:
        sampling_meta = {
            "position_count": len(position_lights),
            "color_count": int(args.color_count),
            "power_values": [float(v) for v in args.power_values],
            "base_energy": float(args.base_energy),
            "default_power_scale": float(args.default_power_scale),
            "radius": float(args.radius),
            "position_policy": "4x4x4 grid centers first, configured default power scale, reject invalid/dark/overlap, random refill",
            "color_policy": "HSV hues with S=1,V=1 at one anchor position, configured default power scale",
            "power_policy": "configured power scales at one anchor position",
            "color_power_anchor_policy": (
                "sort position samples by object_direct_lit_thr010_clean_minarea24 area descending and use requested rank"
            ),
            "ambient_policy": (
                "one random HDRI per scene shared by all position/color/power samples; comparison renders include "
                "selected HDRI, constant, and alternate HDRI ambient-only sources"
            ),
            **anchor_sampling_meta,
        }

    scene_meta = {
        "schema": "tokenlight_final_objaverse_light_mask_scene_v1",
        "scene_id": scene_id,
        "source_root": str(source_root) if source_root is not None else None,
        "source_scene": str(source_scene_dir) if source_scene_dir is not None else None,
        "object_manifest_mode": bool(asset_override is not None),
        "object": source_meta.get("object", {}),
        "object_resolved_path": str(asset_path),
        "camera": camera_meta,
        "render": {
            "resolution": [int(bpy.context.scene.render.resolution_x), int(bpy.context.scene.render.resolution_y)],
            "samples": int(args.samples),
            "component_format": args.component_format,
            "gpu_devices": gpu_meta,
        },
        "source": {
            "ambient_only": relight.component_meta(source_output),
            "ambient_source": ambient_source,
            "comparisons": source_comparisons,
        },
        "shadow_loss_debug": shadow_loss_debug,
        "common": {
            "pbr_maps": pbr_maps,
            "object_mask": object_mask_rel,
            "receiver_masks": receiver_masks,
            "receiver_texture_pool": receiver_texture_status,
            "receiver_materials": config.get("_runtime", {}).get("receiver_materials", []),
            "ambient_white_mask_reference": (
                {
                    "full_occ": relight.component_meta(ambient_white_reference["full_occ"]),
                    "no_object_shadow": relight.component_meta(ambient_white_reference["no_object_shadow"]),
                }
                if ambient_white_reference is not None
                else None
            ),
        },
        "sampling": sampling_meta,
        "final_masks": {
            "object_mask": object_mask_rel,
            "per_sample_masks": {
                "object_shadow_clean": (
                    "object-only geometry visibility ray mask, object excluded, min_area cleanup"
                    if args.shadow_mask_mode == "geometry-ray"
                    else (
                        "object-only shadow loss from white diffuse renders with the selected scene ambient source enabled, "
                        f"object excluded, threshold mode "
                        f"{'ambient-subtracted local ratio' if args.ambient_subtracted_shadow_ratio else ('any positive loss' if args.shadow_threshold <= 0.0 else 'normalized p99')}, "
                        f"threshold {float(args.shadow_threshold)}, min_area cleanup"
                    )
                ),
                "object_shadow_clean_pad16": (
                    f"object_shadow_clean dilated by {int(args.pad_radius)} px, object excluded"
                ),
                "object_direct_lit_clean": (
                    "front-facing object-only geometry visibility ray mask, "
                    f"object mask eroded by {int(args.object_mask_erode_radius)} px, min_area cleanup"
                    if args.shadow_mask_mode == "geometry-ray"
                    else (
                        "object direct-lit mask from white diffuse full_occ with selected scene ambient source enabled, "
                        f"object mask eroded by {int(args.object_mask_erode_radius)} px, "
                        "norm-p95 threshold 0.1, min_area cleanup"
                    )
                ),
            },
        },
        "candidate_attempts": candidate_attempts,
        "samples": samples,
        "sample_counts": {
            "position": len([s for s in samples if s["task"] == "position"]),
            "color": len([s for s in samples if s["task"] == "color"]),
            "power": len([s for s in samples if s["task"] == "power"]),
            "total": len(samples),
        },
    }
    write_json(scene_dir / "meta.json", scene_meta)
    return scene_meta


def main() -> int:
    args = parse_args()
    root = repo_root()
    source_root, scene_specs = selected_scene_specs(args, root)
    output_root = resolve_path(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metas = []
    failed = []
    skipped_completed = 0
    for spec in scene_specs:
        scene_id = spec["scene_id"]
        error_path = output_root / "failed_scenes" / f"{scene_id}_error.json"
        if args.skip_completed:
            completed_meta = load_completed_scene_meta(output_root, scene_id)
            if completed_meta is not None:
                metas.append(completed_meta)
                skipped_completed += 1
                error_path.unlink(missing_ok=True)
                print(f"[resume] skip completed {scene_id}", flush=True)
                continue
        try:
            meta = render_scene(
                args,
                source_root,
                scene_id,
                output_root,
                root,
                asset_override=spec.get("asset_path"),
            )
            metas.append(meta)
            error_path.unlink(missing_ok=True)
        except Exception as exc:
            record = {
                "scene_id": scene_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            failed.append(record)
            write_json(error_path, record)
            if args.fail_fast:
                raise
    manifest = {
        "schema": "tokenlight_final_objaverse_light_mask_dataset_v1",
        "scene_set": args.scene_set,
        "source_root": str(source_root) if source_root is not None else None,
        "object_manifest": args.object_manifest,
        "object_offset": int(args.object_offset),
        "object_limit": args.object_limit,
        "output_root": str(output_root),
        "scene_count_requested": len(scene_specs),
        "scene_count_written": len(metas),
        "scene_count_failed": len(failed),
        "scene_count_skipped_completed": skipped_completed,
        "settings": {
            "resolution": int(args.resolution),
            "samples": int(args.samples),
            "object_target_size": float(args.object_target_size),
            "object_min_size": float(args.object_min_size),
            "rig_reference_object_size": float(args.rig_reference_object_size),
            "canonical_world_scale": float(args.canonical_world_scale),
            "canonical_scale_multiplier_range": [1.0, 1.0],
            "fixed_upper_half_white_grid": bool(args.fixed_upper_half_white_grid),
            "position_candidate_count": 32 if args.fixed_upper_half_white_grid else int(args.positions_per_scene),
            "color_count": 0 if args.fixed_upper_half_white_grid else int(args.color_count),
            "power_values": [float(v) for v in args.power_values],
            "default_power_scale": float(args.default_power_scale),
            "anchor_direct_lit_rank": int(args.anchor_direct_lit_rank),
            "min_area": int(args.min_area),
            "pad_radius": int(args.pad_radius),
            "shadow_threshold": float(args.shadow_threshold),
            "shadow_mask_mode": args.shadow_mask_mode,
            "ambient_subtracted_shadow_ratio": bool(args.ambient_subtracted_shadow_ratio),
            "shadow_support_threshold": float(args.shadow_support_threshold),
            "direct_lit_threshold": float(args.direct_lit_threshold),
            "ambient_manifest": args.ambient_manifest,
            "receiver_texture_manifest": (
                metas[0]["common"]["receiver_texture_pool"]["manifest"]
                if metas
                else (args.receiver_texture_manifest or DEFAULT_RECEIVER_TEXTURE_MANIFEST)
            ),
            "receiver_textures_enabled": not args.no_receiver_textures,
            "skip_completed": bool(args.skip_completed),
            "receiver_texture_usable_count": (
                metas[0]["common"]["receiver_texture_pool"]["usable_count"] if metas else 0
            ),
            "ambient_strength_range": [float(v) for v in args.ambient_strength_range],
            "ambient_fallback_strength": float(args.ambient_fallback_strength),
            "ambient_fallback_color": [float(c) for c in args.ambient_fallback_color[:3]],
            "source_comparison_resolutions": [
                [int(width), int(height)] for width, height in args.source_comparison_resolutions
            ],
        },
        "scenes": [{"scene_id": m["scene_id"], "meta": f"scenes/{m['scene_id']}/meta.json"} for m in metas],
        "failed_scenes": failed,
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    print(f"wrote {output_root / 'dataset_manifest.json'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
