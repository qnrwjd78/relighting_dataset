"""Build source/target/HDR pair data from selected fixed Objaverse scenes.

Run with Blender Python, for example:

  blender -b --python scripts/build_relight_pair_dataset.py -- --models both
"""

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
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = "outputs/scene2000_2099_selected50_fixed32_geometry_erode1_exr_shards"
DEFAULT_SCENE_LIST = "metadata/objaverse_xl/scene2000_2099_selected50_scenes.txt"
DEFAULT_OUTPUT_ROOT = "outputs/scene2000_2099_selected50_hdr_pairs"
DEFAULT_BASE_CONFIG = "configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json"
DEFAULT_HDRI_MANIFEST = "metadata/polyhaven_hdri/polyhaven_hdri_hdris.txt"
DEFAULT_RECEIVER_TEXTURE_MANIFEST = "metadata/polyhaven_textures/polyhaven_textures.json"
DEFAULT_SEED = 20260722

MODEL_SPECS = {
    "unirelight": {
        "display_name": "UniRelight",
        "rgb_size": [848, 480],
        "env_size": [848, 480],
        "frame_count": 57,
    },
    "diffusion_renderer": {
        "display_name": "Diffusion Renderer",
        "rgb_size": [512, 512],
        "env_size": [512, 512],
        "frame_count": 24,
    },
}


def blender_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each selected fixed Objaverse scene, render one source image and one target image "
            "under a newly selected HDRI, then export the target HDRI and one geometry-ray shadow mask."
        )
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--scene-list", default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--hdri-manifest", default=DEFAULT_HDRI_MANIFEST)
    parser.add_argument("--receiver-texture-manifest", default=DEFAULT_RECEIVER_TEXTURE_MANIFEST)
    parser.add_argument("--no-receiver-textures", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["both"],
        choices=["both", "unirelight", "diffusion_renderer", "diffusion-renderer"],
        help="Model-sized outputs to write. 'both' writes UniRelight and Diffusion Renderer folders.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--component-format", choices=["png", "both"], default="png")
    parser.add_argument("--gpu-devices", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--object-min-size", type=float, default=0.6)
    parser.add_argument("--object-target-size", type=float, default=0.9)
    parser.add_argument("--rig-reference-object-size", type=float, default=1.2)
    parser.add_argument("--canonical-world-scale", type=float, default=0.75)
    parser.add_argument("--ambient-fallback-strength", type=float, default=0.8)
    parser.add_argument("--ambient-fallback-color", nargs=3, type=float, default=[0.78, 0.78, 0.78])
    parser.add_argument(
        "--target-env-rot-degrees",
        type=float,
        default=180.0,
        help="Camera-relative target HDRI yaw used for inference-style fixed_pose env conditioning.",
    )
    parser.add_argument(
        "--world-fixed-hdri",
        action="store_true",
        help="Use --target-env-rot-degrees directly as Blender world yaw instead of adding camera azimuth.",
    )
    parser.add_argument("--target-initial-strength", type=float, default=1.0)
    parser.add_argument("--target-max-strength", type=float, default=32.0)
    parser.add_argument("--target-min-mean", type=float, default=0.12)
    parser.add_argument("--target-min-p90", type=float, default=0.35)
    parser.add_argument("--brightness-attempts", type=int, default=6)
    parser.add_argument(
        "--mask-mode",
        choices=["hdr-loss", "geometry-ray"],
        default="hdr-loss",
        help=(
            "hdr-loss renders target HDRI-only white-diffuse object shadow on/off and thresholds the loss. "
            "geometry-ray keeps the older object-only ray mask."
        ),
    )
    parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=0.10,
        help="Threshold10 policy for hdr-loss masks: normalized positive loss > threshold.",
    )
    parser.add_argument(
        "--mask-direct-lit-threshold",
        type=float,
        default=0.10,
        help="Only used by --mask-mode geometry-ray: prefer samples whose direct-lit object ratio is at least this value.",
    )
    parser.add_argument(
        "--mask-selection-rank",
        type=int,
        default=1,
        help="After threshold filtering and shadow-area sorting, select this 1-based rank.",
    )
    parser.add_argument("--object-mask-erode-radius", type=int, default=1)
    parser.add_argument("--mask-min-area", type=int, default=0)
    parser.add_argument("--mask-pad-radius", type=int, default=2)
    parser.add_argument("--link-mode", choices=["copy", "symlink", "hardlink"], default="hardlink")
    parser.add_argument("--write-frame-stacks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(blender_argv())

    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.brightness_attempts < 1:
        parser.error("--brightness-attempts must be >= 1")
    if args.target_initial_strength <= 0.0 or args.target_max_strength <= 0.0:
        parser.error("target strengths must be positive")
    if args.target_initial_strength > args.target_max_strength:
        parser.error("--target-initial-strength must not exceed --target-max-strength")
    if args.mask_selection_rank < 1:
        parser.error("--mask-selection-rank must be >= 1")
    return args


def import_blender_modules():
    try:
        import bpy
    except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
        raise SystemExit("This script must be run by Blender Python: blender -b --python ... -- [args]") from exc

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import render_objaverse_fixed_dataset as fixed
    import render_objaverse_random_dataset as relight
    import relighting_mask_pipeline as mask_pipeline

    return bpy, fixed, relight, mask_pipeline


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("/workspace/"):
        return (ROOT / text[len("/workspace/") :]).resolve()
    path = Path(text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def repo_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_scene_ids(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    scene_ids = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.strip().split()[0] if raw.strip() else ""
        if not token or token.startswith("#"):
            continue
        if token.startswith("scene_"):
            scene_id = token
        else:
            scene_id = f"scene_{int(token):06d}"
        if scene_id not in seen:
            scene_ids.append(scene_id)
            seen.add(scene_id)
    return scene_ids


def scene_meta_index(input_root: Path) -> dict[str, Path]:
    candidates = []
    candidates.extend(sorted(input_root.glob("scenes/scene_*/meta.json")))
    candidates.extend(sorted(input_root.glob("shard_*/scenes/scene_*/meta.json")))
    return {path.parent.name: path for path in candidates}


def selected_models(raw_models: list[str]) -> list[str]:
    normalized = ["diffusion_renderer" if item == "diffusion-renderer" else item for item in raw_models]
    if "both" in normalized:
        return ["unirelight", "diffusion_renderer"]
    result = []
    for name in normalized:
        if name not in result:
            result.append(name)
    return result


def load_manifest_paths(path: Path) -> list[Path]:
    paths = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        resolved = resolve_repo_path(line)
        if resolved is not None and resolved.exists():
            paths.append(resolved)
    if not paths:
        raise SystemExit(f"No existing HDRIs found in manifest: {path}")
    return paths


def source_ambient_from_meta(meta: dict) -> dict:
    source = dict(meta.get("source", {}).get("ambient_source", {}))
    if source.get("path"):
        resolved = resolve_repo_path(source["path"])
        source["path"] = str(resolved) if resolved else source["path"]
    return source


def object_path_from_meta(meta: dict) -> Path:
    value = meta.get("object_resolved_path") or meta.get("object", {}).get("path")
    path = resolve_repo_path(value)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing object asset for {meta.get('scene_id')}: {value}")
    return path


def choose_target_hdri(scene_id: str, source_ambient: dict, hdris: list[Path], seed: int) -> Path:
    scene_number = int(scene_id.split("_")[-1])
    rng = random.Random(int(seed) + scene_number * 1009 + 15485863)
    source_path = resolve_repo_path(source_ambient.get("path"))
    source_resolved = str(source_path.resolve()) if source_path else None
    candidates = [path for path in hdris if str(path.resolve()) != source_resolved]
    return rng.choice(candidates or hdris)


def target_rotation_z(meta: dict, args: argparse.Namespace) -> float:
    base = math.radians(float(args.target_env_rot_degrees))
    if args.world_fixed_hdri:
        return base
    azimuth = float(meta.get("camera", {}).get("azimuth_degrees", 0.0))
    return base + math.radians(azimuth)


def materialize_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if mode == "symlink":
            os.symlink(src.resolve(), dst)
        elif mode == "hardlink":
            os.link(src.resolve(), dst)
        else:
            raise ValueError(mode)
    except OSError:
        shutil.copy2(src, dst)


def image_luminance_stats(bpy, path: Path) -> dict:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        channels = max(int(image.channels), 1)
        pixels = array("f", [0.0]) * (width * height * channels)
        image.pixels.foreach_get(pixels)
        values = []
        for pixel_index in range(width * height):
            offset = pixel_index * channels
            r = float(pixels[offset])
            g = float(pixels[offset + min(1, channels - 1)])
            b = float(pixels[offset + min(2, channels - 1)])
            lum = max(0.0, min(1.0, 0.2126 * r + 0.7152 * g + 0.0722 * b))
            if math.isfinite(lum):
                values.append(lum)
        if not values:
            return {"width": width, "height": height, "valid": False}
        values.sort()

        def percentile(q: float) -> float:
            index = min(max(int(round((len(values) - 1) * q)), 0), len(values) - 1)
            return float(values[index])

        return {
            "width": width,
            "height": height,
            "valid": True,
            "mean": float(sum(values) / len(values)),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": float(values[-1]),
        }
    finally:
        bpy.data.images.remove(image)


def brightness_ok(stats: dict, args: argparse.Namespace) -> bool:
    return bool(
        stats.get("valid")
        and float(stats.get("mean", 0.0)) >= float(args.target_min_mean)
        and float(stats.get("p90", 0.0)) >= float(args.target_min_p90)
    )


def next_strength(strength: float, stats: dict, args: argparse.Namespace) -> float:
    mean = max(float(stats.get("mean", 0.0)), 1.0e-4)
    p90 = max(float(stats.get("p90", 0.0)), 1.0e-4)
    mean_factor = float(args.target_min_mean) / mean
    p90_factor = float(args.target_min_p90) / p90
    factor = max(mean_factor, p90_factor, 1.25)
    factor = min(max(factor, 1.25), 2.5)
    return min(float(args.target_max_strength), strength * factor)


def component_png_path(scene_dir: Path, output: dict) -> Path:
    rel = output.get("png") or output.get("primary")
    if rel is None:
        raise RuntimeError(f"Component has no PNG output: {output}")
    return scene_dir / str(rel)


def set_resolution(bpy, config: dict, width: int, height: int) -> None:
    config["render"]["resolution"] = [int(width), int(height)]
    config["render"]["resolution_x"] = int(width)
    config["render"]["resolution_y"] = int(height)
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100


def build_render_scene(
    meta: dict,
    model_scene_dir: Path,
    width: int,
    height: int,
    args: argparse.Namespace,
    fixed,
    relight,
) -> tuple[dict, object, list, object, dict]:
    scene_id = str(meta["scene_id"])
    scene_number = int(scene_id.split("_")[-1])
    rng = random.Random(int(args.seed) + scene_number)
    asset_path = object_path_from_meta(meta)
    config = read_json(resolve_repo_path(args.base_config))
    config["render"].update(
        {
            "resolution": [int(width), int(height)],
            "resolution_x": int(width),
            "resolution_y": int(height),
            "samples": int(args.samples),
            "component_format": args.component_format,
        }
    )
    config["_runtime"] = {"objects": [], "hdris": [], "receiver_textures": [], "fixture_scenes": []}
    relight.configure_object_rig_scale(
        config,
        float(args.object_min_size),
        float(args.object_target_size),
        float(args.rig_reference_object_size),
        float(args.canonical_world_scale),
    )
    config["_component_format"] = args.component_format
    config["_render_pbr"] = False
    config["_render_relighting_masks"] = False
    config["ambient"] = {
        "hdri_strength_range": [float(args.target_initial_strength), float(args.target_initial_strength)],
        "hdri_rotation_random": False,
        "hdri_mode": "on",
        "hdri_probability": 1.0,
        "fallback_color": [float(c) for c in args.ambient_fallback_color[:3]],
        "fallback_strength": float(args.ambient_fallback_strength),
    }
    receiver_texture_status = relight.configure_receiver_textures(
        config,
        ROOT,
        manifest_override=args.receiver_texture_manifest,
        disabled=bool(args.no_receiver_textures),
    )

    relight.clear_scene()
    relight.setup_render_settings(config)
    fixed.configure_gpu_devices(args.gpu_devices)

    source_meta = {
        "object": {
            "path": str(asset_path),
            "primitive": None,
            "target_size": float(args.object_target_size),
            "min_size": float(args.object_min_size),
        }
    }
    subject_objects = relight.import_asset_or_primitive(str(asset_path), "sphere", rng, config)
    bbox_min, bbox_max = relight.mesh_bbox(subject_objects)
    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)
    center = (bbox_min + bbox_max) * 0.5
    camera, camera_meta = relight.create_camera(config, rng, center)
    relight.create_receivers(config, rng, camera, center)
    model_scene_dir.mkdir(parents=True, exist_ok=True)
    return config, camera, subject_objects, center, {
        "asset_path": str(asset_path),
        "camera": camera_meta,
        "receiver_texture_pool": receiver_texture_status,
        "object": source_meta["object"],
    }


def select_mask_sample(meta: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    samples = [
        sample
        for sample in meta.get("samples", [])
        if sample.get("task") == "position"
        and sample.get("light", {}).get("world_position") is not None
        and sample.get("masks", {}).get("object_shadow_clean")
    ]
    if not samples:
        raise RuntimeError(f"No position samples with geometry masks for {meta.get('scene_id')}")
    threshold = float(args.mask_direct_lit_threshold)
    eligible = [
        sample
        for sample in samples
        if float(sample.get("mask_stats", {}).get("object_direct_lit_clean_ratio", 0.0)) >= threshold
    ]
    pool = eligible or samples
    ranked = sorted(
        pool,
        key=lambda sample: (
            -float(sample.get("mask_stats", {}).get("object_shadow_clean_ratio", 0.0)),
            -float(sample.get("mask_stats", {}).get("object_direct_lit_clean_ratio", 0.0)),
            int(sample.get("light", {}).get("id", 0)),
        ),
    )
    index = min(int(args.mask_selection_rank) - 1, len(ranked) - 1)
    selected = ranked[index]
    selection = {
        "policy": "threshold10_direct_lit_then_shadow_area_rank",
        "direct_lit_threshold": threshold,
        "requested_rank": int(args.mask_selection_rank),
        "used_rank": int(index + 1),
        "eligible_count": len(eligible),
        "fallback_used": len(eligible) == 0,
        "sample_name": selected.get("name"),
        "light_id": selected.get("light", {}).get("id"),
        "object_shadow_clean_ratio": float(selected.get("mask_stats", {}).get("object_shadow_clean_ratio", 0.0)),
        "object_direct_lit_clean_ratio": float(
            selected.get("mask_stats", {}).get("object_direct_lit_clean_ratio", 0.0)
        ),
        "source_mask": selected.get("masks", {}).get("object_shadow_clean"),
    }
    return selected, selection


def render_single_mask(
    bpy,
    model_scene_dir: Path,
    config: dict,
    camera,
    subject_objects: list,
    selected_sample: dict,
    source_ambient: dict,
    args: argparse.Namespace,
    relight,
    mask_pipeline,
) -> dict:
    relight.set_black_world()
    object_mask_rel = relight.render_object_mask(model_scene_dir, subject_objects)
    receiver_masks = relight.render_receiver_masks(model_scene_dir)
    shape = (int(bpy.context.scene.render.resolution_y), int(bpy.context.scene.render.resolution_x))
    object_mask = mask_pipeline.load_png_mask(model_scene_dir / object_mask_rel, shape)
    receiver_mask = mask_pipeline.load_png_mask(model_scene_dir / receiver_masks["receiver"], shape)
    mask_args = SimpleNamespace(
        min_area=int(args.mask_min_area),
        pad_radius=int(args.mask_pad_radius),
        object_mask_erode_radius=int(args.object_mask_erode_radius),
        default_power_scale=float(selected_sample.get("light", {}).get("power_scale", 1.0)),
    )
    mask_meta = mask_pipeline.render_geometry_ray_masks(
        relight,
        model_scene_dir,
        "mask_reference/selected",
        config,
        camera,
        subject_objects,
        selected_sample["light"],
        object_mask,
        receiver_mask,
        source_ambient,
        mask_args,
    )
    src_png = model_scene_dir / mask_meta["masks"]["object_shadow_clean"]
    src_npy = src_png.with_suffix(".npy")
    dst_png = model_scene_dir / "object_shadow_geometry_ray_clean_minarea00.png"
    dst_npy = model_scene_dir / "object_shadow_geometry_ray_clean_minarea00.npy"
    shutil.copy2(src_png, dst_png)
    if src_npy.exists():
        shutil.copy2(src_npy, dst_npy)
    return {
        "object_mask": object_mask_rel,
        "receiver_masks": receiver_masks,
        "raw_mask_meta": mask_meta,
        "exported_png": dst_png.name,
        "exported_npy": dst_npy.name if dst_npy.exists() else None,
    }


def render_hdr_loss_mask(
    bpy,
    model_scene_dir: Path,
    config: dict,
    subject_objects: list,
    target_ambient: dict,
    args: argparse.Namespace,
    relight,
    mask_pipeline,
) -> dict:
    original_format = config.get("_component_format", config.get("render", {}).get("component_format", "png"))
    config["_component_format"] = "both"
    config.setdefault("render", {})["component_format"] = "both"
    try:
        relight.remove_all_lights()
        relight.set_ambient_source_from_meta(target_ambient, config)
        object_mask_rel = relight.render_object_mask(model_scene_dir, subject_objects)
        receiver_masks = relight.render_receiver_masks(model_scene_dir)
        shape = (int(bpy.context.scene.render.resolution_y), int(bpy.context.scene.render.resolution_x))
        object_mask = mask_pipeline.load_png_mask(model_scene_dir / object_mask_rel, shape)
        receiver_mask = mask_pipeline.load_png_mask(model_scene_dir / receiver_masks["receiver"], shape)

        white_config = relight.relighting_white_shading_config(config)
        full_output = relight.render_white_diffuse_component(
            model_scene_dir,
            "mask_reference/hdr_loss/full_occ",
            config,
            white_config,
            shadows_enabled=True,
        )
        snapshot = mask_pipeline.mesh_shadow_snapshot(subject_objects)
        try:
            mask_pipeline.set_mesh_shadow_visibility(subject_objects, False)
            no_shadow_output = relight.render_white_diffuse_component(
                model_scene_dir,
                "mask_reference/hdr_loss/no_object_shadow",
                config,
                white_config,
                shadows_enabled=True,
            )
        finally:
            mask_pipeline.restore_mesh_shadow_visibility(snapshot)

        full_y = mask_pipeline.load_exr_luminance(mask_pipeline.component_exr_path(model_scene_dir, full_output))
        no_shadow_y = mask_pipeline.load_exr_luminance(
            mask_pipeline.component_exr_path(model_scene_dir, no_shadow_output)
        )
        loss = np.maximum(no_shadow_y - full_y, 0.0)
        loss[object_mask] = 0.0
        loss[~receiver_mask] = 0.0
        shadow_scale = mask_pipeline.positive_percentile(loss[receiver_mask], 99.0)
        if float(args.shadow_threshold) <= 0.0:
            shadow_raw = loss > 0.0
            threshold_mode = "any_positive_loss"
        else:
            shadow_raw = receiver_mask & ((loss / max(shadow_scale, 1.0e-8)) > float(args.shadow_threshold))
            threshold_mode = "normalized_p99_hdr_loss"

        shadow_clean, shadow_stats = mask_pipeline.remove_small_components(
            shadow_raw & ~object_mask,
            int(args.mask_min_area),
        )
        shadow_pad = relight.dilate_binary(shadow_clean, int(args.mask_pad_radius)) & ~object_mask

        mask_dir = model_scene_dir / "mask_reference" / "hdr_loss_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        threshold = mask_pipeline.threshold_token(float(args.shadow_threshold))
        stem = f"object_shadow_hdr_loss_thr{threshold}_clean_minarea{int(args.mask_min_area):02d}"
        pad_stem = f"{stem}_pad{int(args.mask_pad_radius):02d}"
        canonical_png = mask_dir / f"{stem}.png"
        canonical_npy = mask_dir / f"{stem}.npy"
        pad_png = mask_dir / f"{pad_stem}.png"
        pad_npy = mask_dir / f"{pad_stem}.npy"
        relight.save_gray_png(canonical_png, shadow_clean.astype(np.float32))
        relight.save_gray_png(pad_png, shadow_pad.astype(np.float32))
        np.save(canonical_npy, shadow_clean.astype(np.uint8))
        np.save(pad_npy, shadow_pad.astype(np.uint8))

        alias_png = model_scene_dir / "object_shadow_geometry_ray_clean_minarea00.png"
        alias_npy = model_scene_dir / "object_shadow_geometry_ray_clean_minarea00.npy"
        shutil.copy2(canonical_png, alias_png)
        shutil.copy2(canonical_npy, alias_npy)
        return {
            "object_mask": object_mask_rel,
            "receiver_masks": receiver_masks,
            "full_occ": full_output,
            "no_object_shadow": no_shadow_output,
            "canonical_png": repo_relative(canonical_png),
            "canonical_npy": repo_relative(canonical_npy),
            "pad_png": repo_relative(pad_png),
            "pad_npy": repo_relative(pad_npy),
            "exported_png": alias_png.name,
            "exported_npy": alias_npy.name,
            "stats": {
                "mode": "hdr_only_white_diffuse_object_shadow_loss",
                "threshold_mode": threshold_mode,
                "shadow_threshold": float(args.shadow_threshold),
                "shadow_p99_positive_receiver": float(shadow_scale),
                "object_shadow_clean_ratio": float(shadow_clean.mean()),
                "object_shadow_clean_pad_ratio": float(shadow_pad.mean()),
                "shadow_cleanup": shadow_stats,
                "loss_semantics": "max(no_object_shadow_luminance - full_occ_luminance, 0) under target HDRI only",
                "target_hdri_path": target_ambient.get("path"),
                "target_hdri_strength": target_ambient.get("strength"),
                "target_hdri_rotation_z": target_ambient.get("rotation_z"),
                "alias_name_note": (
                    "object_shadow_geometry_ray_clean_minarea00.png is kept as a compatibility alias; "
                    "the canonical mask is object_shadow_hdr_loss_thr010_clean_minarea00.png."
                ),
            },
        }
    finally:
        config["_component_format"] = original_format
        config.setdefault("render", {})["component_format"] = original_format


def render_target_with_brightness(
    bpy,
    model_scene_dir: Path,
    config: dict,
    target_hdri: Path,
    rotation_z: float,
    args: argparse.Namespace,
    relight,
) -> tuple[dict, dict, float, list[dict]]:
    strength = float(args.target_initial_strength)
    attempts = []
    output = {}
    stats = {}
    for attempt in range(int(args.brightness_attempts)):
        relight.remove_all_lights()
        target_ambient = relight.set_hdri_world(
            str(target_hdri),
            strength,
            float(rotation_z),
            [float(c) for c in args.ambient_fallback_color[:3]],
        )
        output = relight.render_component(model_scene_dir, "target", config)
        stats = image_luminance_stats(bpy, component_png_path(model_scene_dir, output))
        ok = brightness_ok(stats, args)
        attempts.append({"attempt": attempt + 1, "strength": strength, "stats": stats, "accepted": ok})
        if ok or strength >= float(args.target_max_strength):
            return output, target_ambient, strength, attempts
        updated = next_strength(strength, stats, args)
        if updated <= strength + 1.0e-6:
            return output, target_ambient, strength, attempts
        strength = updated
    return output, target_ambient, strength, attempts


def write_frame_stack(scene_dir: Path, image_name: str, folder: str, frame_count: int, overwrite: bool) -> list[str]:
    src = scene_dir / image_name
    out_dir = scene_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_idx in range(frame_count):
        dst = out_dir / f"{frame_idx:06d}.png"
        materialize_file(src, dst, "hardlink", overwrite)
        frames.append(f"{folder}/{dst.name}")
    return frames


def render_model_scene(
    bpy,
    fixed,
    relight,
    mask_pipeline,
    model: str,
    meta: dict,
    out_model_root: Path,
    target_hdri: Path,
    target_rotation: float,
    args: argparse.Namespace,
) -> dict:
    spec = MODEL_SPECS[model]
    width, height = spec["rgb_size"]
    scene_id = str(meta["scene_id"])
    model_scene_dir = out_model_root / scene_id
    if (model_scene_dir / "meta.json").exists() and not args.overwrite:
        return read_json(model_scene_dir / "meta.json")

    config, camera, subject_objects, _center, replay_meta = build_render_scene(
        meta, model_scene_dir, width, height, args, fixed, relight
    )
    set_resolution(bpy, config, width, height)

    source_ambient = source_ambient_from_meta(meta)
    relight.remove_all_lights()
    relight.set_ambient_source_from_meta(source_ambient, config)
    source_output = relight.render_component(model_scene_dir, "source", config)
    source_stats = image_luminance_stats(bpy, component_png_path(model_scene_dir, source_output))

    target_output, target_ambient, target_strength, brightness_attempts = render_target_with_brightness(
        bpy,
        model_scene_dir,
        config,
        target_hdri,
        target_rotation,
        args,
        relight,
    )
    target_stats = image_luminance_stats(bpy, component_png_path(model_scene_dir, target_output))

    selected_sample = None
    mask_selection = None
    if args.mask_mode == "hdr-loss":
        mask_export = render_hdr_loss_mask(
            bpy,
            model_scene_dir,
            config,
            subject_objects,
            target_ambient,
            args,
            relight,
            mask_pipeline,
        )
    else:
        selected_sample, mask_selection = select_mask_sample(meta, args)
        mask_export = render_single_mask(
            bpy,
            model_scene_dir,
            config,
            camera,
            subject_objects,
            selected_sample,
            source_ambient,
            args,
            relight,
            mask_pipeline,
        )

    hdr_dst = model_scene_dir / f"target{target_hdri.suffix.lower() or '.hdr'}"
    materialize_file(target_hdri, hdr_dst, args.link_mode, args.overwrite)

    source_frames = None
    target_frames = None
    if args.write_frame_stacks:
        source_frames = write_frame_stack(
            model_scene_dir, "source.png", "source_frames", int(spec["frame_count"]), args.overwrite
        )
        target_frames = write_frame_stack(
            model_scene_dir, "target.png", "target_frames", int(spec["frame_count"]), args.overwrite
        )

    item_meta = {
        "schema": "tokenlight_relight_pair_scene_v1",
        "scene_id": scene_id,
        "model": model,
        "model_display_name": spec["display_name"],
        "model_input": {
            "rgb_size": spec["rgb_size"],
            "env_condition_size": spec["env_size"],
            "frame_count": int(spec["frame_count"]),
            "target_image_count": 1,
            "source_frame_stack": source_frames,
            "target_frame_stack": target_frames,
        },
        "files": {
            "source": source_output.get("png") or source_output.get("primary"),
            "target": target_output.get("png") or target_output.get("primary"),
            "target_hdr": hdr_dst.name,
            "object_shadow_geometry_ray_clean_minarea00": mask_export["exported_png"],
            "object_shadow_geometry_ray_clean_minarea00_npy": mask_export["exported_npy"],
        },
        "source": {
            "ambient_source": source_ambient,
            "render": source_output,
            "brightness": source_stats,
        },
        "target": {
            "ambient_source": {
                **target_ambient,
                "path": str(target_hdri),
                "exported_hdr": hdr_dst.name,
                "strength": float(target_strength),
                "rotation_z": float(target_rotation),
                "rotation_degrees": float(math.degrees(target_rotation)),
                "env_rot_degrees": float(args.target_env_rot_degrees),
                "camera_relative": not bool(args.world_fixed_hdri),
                "fixed_pose_inference_expected": True,
                "direction_convention": {
                    "up": "+Y",
                    "down": "-Y",
                    "azimuth_axis": "Y",
                    "default_env_rot_degrees": 180.0,
                },
            },
            "render": target_output,
            "brightness": target_stats,
            "brightness_policy": {
                "min_mean": float(args.target_min_mean),
                "min_p90": float(args.target_min_p90),
                "initial_strength": float(args.target_initial_strength),
                "max_strength": float(args.target_max_strength),
                "attempts": brightness_attempts,
            },
        },
        "mask": {
            "selection": mask_selection,
            "export": mask_export,
            "name": "object_shadow_geometry_ray_clean_minarea00",
            "mode": args.mask_mode,
            "min_area": int(args.mask_min_area),
            "object_mask_erode_radius": int(args.object_mask_erode_radius),
            "pad_radius": int(args.mask_pad_radius),
            "shadow_threshold": float(args.shadow_threshold) if args.mask_mode == "hdr-loss" else None,
        },
        "replay": replay_meta,
        "source_scene_meta": repo_relative(resolve_repo_path(meta["_meta_path"])),
    }
    write_json(model_scene_dir / "meta.json", item_meta)
    return item_meta


def write_manifests(output_root: Path, records: list[dict]) -> None:
    jsonl = output_root / "manifest.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema": "tokenlight_relight_pair_manifest_v1",
        "item_count": len(records),
        "models": sorted(set(record["model"] for record in records)),
        "scenes": sorted(set(record["scene_id"] for record in records)),
        "manifest_jsonl": jsonl.name,
        "items": records,
    }
    write_json(output_root / "manifest.json", summary)


def dry_run(args: argparse.Namespace) -> int:
    input_root = resolve_repo_path(args.input_root)
    scene_list = read_scene_ids(resolve_repo_path(args.scene_list))
    metas = scene_meta_index(input_root)
    scene_ids = scene_list or sorted(metas)
    if args.limit is not None:
        scene_ids = scene_ids[: args.limit]
    missing = [scene_id for scene_id in scene_ids if scene_id not in metas]
    models = selected_models(args.models)
    print(f"[dry-run] input_root={input_root}")
    print(f"[dry-run] output_root={resolve_repo_path(args.output_root)}")
    print(f"[dry-run] models={models}")
    print(f"[dry-run] scenes={len(scene_ids)} missing={len(missing)}")
    if missing:
        print(f"[dry-run] first_missing={missing[0]}")
    return 0 if not missing else 1


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run(args)

    bpy, fixed, relight, mask_pipeline = import_blender_modules()
    input_root = resolve_repo_path(args.input_root)
    output_root = resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hdri_manifest = resolve_repo_path(args.hdri_manifest)
    hdris = load_manifest_paths(hdri_manifest)
    scene_ids = read_scene_ids(resolve_repo_path(args.scene_list))
    meta_paths = scene_meta_index(input_root)
    if scene_ids is None:
        scene_ids = sorted(meta_paths)
    if args.limit is not None:
        scene_ids = scene_ids[: args.limit]
    models = selected_models(args.models)

    records = []
    for scene_offset, scene_id in enumerate(scene_ids, 1):
        meta_path = meta_paths.get(scene_id)
        if meta_path is None:
            raise FileNotFoundError(f"Missing rendered scene meta for {scene_id} under {input_root}")
        meta = read_json(meta_path)
        meta["_meta_path"] = str(meta_path)
        source_ambient = source_ambient_from_meta(meta)
        target_hdri = choose_target_hdri(scene_id, source_ambient, hdris, int(args.seed))
        rotation = target_rotation_z(meta, args)
        print(
            f"[pair] {scene_offset}/{len(scene_ids)} {scene_id} "
            f"target_hdri={target_hdri.name} rotation_deg={math.degrees(rotation):.2f}",
            flush=True,
        )
        for model in models:
            item_meta = render_model_scene(
                bpy,
                fixed,
                relight,
                mask_pipeline,
                model,
                meta,
                output_root / model,
                target_hdri,
                rotation,
                args,
            )
            records.append(
                {
                    "scene_id": scene_id,
                    "model": model,
                    "root": repo_relative(output_root / model / scene_id),
                    "source": repo_relative(output_root / model / scene_id / "source.png"),
                    "target": repo_relative(output_root / model / scene_id / "target.png"),
                    "target_hdr": repo_relative(output_root / model / scene_id / item_meta["files"]["target_hdr"]),
                    "mask": repo_relative(
                        output_root
                        / model
                        / scene_id
                        / item_meta["files"]["object_shadow_geometry_ray_clean_minarea00"]
                    ),
                    "meta": repo_relative(output_root / model / scene_id / "meta.json"),
                }
            )

    write_manifests(output_root, records)
    print(f"[pair] wrote {len(records)} items -> {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
