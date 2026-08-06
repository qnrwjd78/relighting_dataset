#!/usr/bin/env python3
"""Generate LGI maps from a TokenLight fixed-light scene.

The renderer stores Blender Z-pass depth (distance along the camera ray). This
script converts it to OpenCV optical-axis depth before applying the LGI lift,
sample, reproject, and angular aggregation procedure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import OpenEXR
except ImportError as exc:  # pragma: no cover - dependency error is user-facing
    raise SystemExit("OpenEXR is required: python3 -m pip install OpenEXR") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene-dir", type=Path, help="Generate one scene.")
    source.add_argument("--dataset-root", type=Path, help="Generate every scene below a dataset root.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for --scene-dir mode.")
    parser.add_argument("--output-root", type=Path, help="Output root for --dataset-root mode.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write position_NNN channel directories directly inside each source scene.",
    )
    parser.add_argument(
        "--channel-files",
        action="store_true",
        help="Store min/max/nearest and debug arrays as separate NPY files.",
    )
    parser.add_argument("--num-ray-samples", type=int, default=16)
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--hard-threshold-deg", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel scene workers in dataset mode.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate completed scene outputs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed scene.")
    parser.add_argument(
        "--position-id",
        type=int,
        action="append",
        default=None,
        help="Generate only this fixed-grid position id; repeat for multiple ids.",
    )
    return parser.parse_args()


def read_exr(path: Path) -> np.ndarray:
    exr = OpenEXR.File(str(path))
    channels = exr.channels()
    if "RGB" in channels:
        image = np.asarray(channels["RGB"].pixels, dtype=np.float32)
    else:
        names = [name for name in ("R", "G", "B") if name in channels]
        if not names:
            raise ValueError(f"No RGB or R channel in {path}")
        image = np.stack(
            [np.asarray(channels[name].pixels, dtype=np.float32) for name in names],
            axis=-1,
        )
    return image


def camera_model(meta: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    camera = meta["camera"]
    fov_x = math.radians(float(camera["fov_degrees"]))
    fx = 0.5 * width / math.tan(0.5 * fov_x)
    fy = fx
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    axes = camera.get("similarity_transform", {}).get("axes", {})
    right = np.asarray(axes.get("x_axis_world"), dtype=np.float32)
    forward = np.asarray(axes.get("y_axis_world"), dtype=np.float32)
    up = np.asarray(axes.get("z_axis_world"), dtype=np.float32)
    if right.shape != (3,) or forward.shape != (3,) or up.shape != (3,):
        raise ValueError("camera.similarity_transform.axes is required for exact camera orientation")
    right /= np.linalg.norm(right)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)

    # World to OpenCV camera: +X right, +Y down, +Z forward.
    rotation_world_to_cv = np.stack((right, -up, forward), axis=0)
    camera_center = np.asarray(camera["location"], dtype=np.float32)
    world_to_cv = np.eye(4, dtype=np.float32)
    world_to_cv[:3, :3] = rotation_world_to_cv
    world_to_cv[:3, 3] = -rotation_world_to_cv @ camera_center
    return intrinsic, world_to_cv


def ray_distance_to_z(depth_ray: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth_ray.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    x = (u - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) / intrinsic[1, 1]
    ray_norm = np.sqrt(x * x + y * y + 1.0)
    return depth_ray / ray_norm


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    inside = (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
    safe_u = np.clip(u, 0.0, width - 1)
    safe_v = np.clip(v, 0.0, height - 1)
    x0 = np.floor(safe_u).astype(np.int32)
    y0 = np.floor(safe_v).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = safe_u - x0
    wy = safe_v - y0
    sampled = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    valid = inside & np.isfinite(sampled) & (sampled > 0.0) & (sampled < 1.0e6)
    return sampled.astype(np.float32), valid


def generate_lgi(
    depth_z: np.ndarray,
    intrinsic: np.ndarray,
    light_cv: np.ndarray,
    num_ray_samples: int,
    tile_rows: int,
    hard_threshold_rad: float,
) -> dict[str, np.ndarray]:
    height, width = depth_z.shape
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    deltas = np.linspace(
        1.0 / (num_ray_samples + 1),
        num_ray_samples / (num_ray_samples + 1),
        num_ray_samples,
        dtype=np.float32,
    )[:, None, None, None]

    lgi = np.zeros((3, height, width), dtype=np.float32)
    valid_map = np.zeros((height, width), dtype=bool)
    min_abs_map = np.zeros((height, width), dtype=np.float32)

    u_base = np.arange(width, dtype=np.float32)[None, :]
    for row_start in range(0, height, tile_rows):
        row_end = min(row_start + tile_rows, height)
        v_base = np.arange(row_start, row_end, dtype=np.float32)[:, None]
        z = depth_z[row_start:row_end]
        source_valid = np.isfinite(z) & (z > 0.0) & (z < 1.0e6)
        x = (u_base - cx) * z / fx
        y = (v_base - cy) * z / fy
        points = np.stack(np.broadcast_arrays(x, y, z), axis=-1).astype(np.float32)

        to_light = light_cv[None, None, :] - points
        light_norm = np.linalg.norm(to_light, axis=-1)
        light_valid = source_valid & (light_norm > 1.0e-7)
        light_elevation = np.arcsin(
            np.clip(to_light[..., 2] / np.maximum(light_norm, 1.0e-7), -1.0, 1.0)
        )

        ray_points = points[None, ...] + deltas * to_light[None, ...]
        ray_z = ray_points[..., 2]
        projectable = ray_z > 1.0e-7
        projected_u = fx * ray_points[..., 0] / np.maximum(ray_z, 1.0e-7) + cx
        projected_v = fy * ray_points[..., 1] / np.maximum(ray_z, 1.0e-7) + cy
        sampled_z, sampled_valid = bilinear_sample(depth_z, projected_u, projected_v)

        sample_x = (projected_u - cx) * sampled_z / fx
        sample_y = (projected_v - cy) * sampled_z / fy
        visible_points = np.stack((sample_x, sample_y, sampled_z), axis=-1)
        surface_vectors = visible_points - points[None, ...]
        surface_norm = np.linalg.norm(surface_vectors, axis=-1)
        sample_valid = (
            sampled_valid
            & projectable
            & light_valid[None, ...]
            & (surface_norm > 1.0e-7)
        )
        surface_elevation = np.arcsin(
            np.clip(
                surface_vectors[..., 2] / np.maximum(surface_norm, 1.0e-7),
                -1.0,
                1.0,
            )
        )
        angle_diff = surface_elevation - light_elevation[None, ...]

        any_valid = np.any(sample_valid, axis=0)
        min_values = np.min(np.where(sample_valid, angle_diff, np.inf), axis=0)
        max_values = np.max(np.where(sample_valid, angle_diff, -np.inf), axis=0)
        abs_values = np.where(sample_valid, np.abs(angle_diff), np.inf)
        nearest_indices = np.argmin(abs_values, axis=0)
        nearest_values = np.take_along_axis(angle_diff, nearest_indices[None, ...], axis=0)[0]

        lgi[0, row_start:row_end] = np.where(any_valid, min_values, 0.0)
        lgi[1, row_start:row_end] = np.where(any_valid, max_values, 0.0)
        lgi[2, row_start:row_end] = np.where(any_valid, nearest_values, 0.0)
        valid_map[row_start:row_end] = any_valid
        min_abs_map[row_start:row_end] = np.where(any_valid, np.abs(nearest_values), 0.0)

    hard = valid_map & (min_abs_map < hard_threshold_rad)
    return {
        "lgi": lgi,
        "valid": valid_map.astype(np.uint8),
        "min_abs": min_abs_map,
        "hard": hard.astype(np.uint8),
    }


def signed_angle_rgb(values: np.ndarray, valid: np.ndarray, limit: float = math.pi / 2) -> np.ndarray:
    t = np.clip(values / limit, -1.0, 1.0)
    positive = np.maximum(t, 0.0)
    negative = np.maximum(-t, 0.0)
    rgb = np.stack(
        (1.0 - negative, 1.0 - np.abs(t), 1.0 - positive),
        axis=-1,
    )
    rgb[~valid] = 0.0
    return np.uint8(np.clip(rgb * 255.0, 0.0, 255.0))


def save_preview(path: Path, result: dict[str, np.ndarray], title: str) -> None:
    valid = result["valid"].astype(bool)
    panels = [signed_angle_rgb(result["lgi"][i], valid) for i in range(3)]
    panels.append(np.repeat(result["hard"][..., None] * 255, 3, axis=-1).astype(np.uint8))
    height, width = valid.shape
    header = 30
    canvas = Image.new("RGB", (width * 4, height + header), "black")
    draw = ImageDraw.Draw(canvas)
    labels = ("c1 min", "c2 max", "c3 nearest", "hard |c3| < threshold")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(panel), (index * width, header))
        draw.text((index * width + 8, 8), label, fill="white")
    draw.text((width * 4 - 220, 8), title, fill="white")
    canvas.save(path)


def load_binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def shadow_comparison(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    target_count = int(np.count_nonzero(target))
    return {
        "gt_ratio": float(target.mean()),
        "iou": float(intersection / union) if union else 1.0,
        "recall": float(intersection / target_count) if target_count else 1.0,
    }


def save_hard_overview(path: Path, output_dir: Path, samples: list[dict], width: int, height: int) -> None:
    columns = 8
    rows = math.ceil(len(samples) / columns)
    thumb_width = 160
    thumb_height = max(1, round(height * thumb_width / width))
    label_height = 20
    canvas = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, sample in enumerate(samples):
        position_id = int(sample["position_id"])
        if "lgi_npz" in sample:
            hard = np.load(output_dir / sample["lgi_npz"])["hard"] * 255
        else:
            hard = np.load(output_dir / sample["channels"]["hard"]) * 255
        image = Image.fromarray(hard.astype(np.uint8), mode="L").convert("RGB")
        image = image.resize((thumb_width, thumb_height), resample=Image.NEAREST)
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 5, y + 3), f"position_{position_id:03d}", fill="white")
    canvas.save(path)


def position_samples(meta: dict, selected_ids: set[int] | None) -> list[dict]:
    samples = []
    for sample in meta.get("samples", []):
        if sample.get("task") != "position":
            continue
        position_id = int(sample["light"]["id"])
        if selected_ids is None or position_id in selected_ids:
            samples.append(sample)
    return samples


def generate_scene(
    scene_dir: Path,
    output_dir: Path,
    num_ray_samples: int,
    tile_rows: int,
    hard_threshold_deg: float,
    selected_position_ids: tuple[int, ...] | None,
    verbose_lights: bool,
    channel_files: bool,
    index_filename: str,
) -> dict:
    scene_dir = scene_dir.resolve()
    output_dir = output_dir.resolve()
    meta_path = scene_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    width, height = map(int, meta["render"]["resolution"])
    depth_rel = meta["common"]["pbr_maps"]["depth"]
    depth_rgb = read_exr(scene_dir / depth_rel)
    depth_ray = depth_rgb[..., 0] if depth_rgb.ndim == 3 else depth_rgb
    if depth_ray.shape != (height, width):
        raise ValueError(f"Depth shape {depth_ray.shape} does not match metadata {(height, width)}")

    intrinsic, world_to_cv = camera_model(meta, width, height)
    depth_z = ray_distance_to_z(depth_ray, intrinsic)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = set(selected_position_ids) if selected_position_ids else None
    samples = position_samples(meta, selected_ids)
    if not samples:
        raise ValueError("No matching position samples found")

    output_samples = []
    threshold_rad = math.radians(hard_threshold_deg)
    for sample in samples:
        light = sample["light"]
        position_id = int(light["id"])
        light_world_h = np.append(np.asarray(light["world_position"], dtype=np.float32), 1.0)
        light_cv = (world_to_cv @ light_world_h)[:3]
        result = generate_lgi(
            depth_z,
            intrinsic,
            light_cv,
            num_ray_samples,
            tile_rows,
            threshold_rad,
        )
        stem = f"position_{position_id:02d}" if channel_files else f"position_{position_id:03d}"
        sample_output = {
            "position_id": position_id,
            "sample_name": sample["name"],
            "light_world": [float(value) for value in light["world_position"]],
            "light_cv": [float(value) for value in light_cv],
            "valid_ratio": float(result["valid"].mean()),
            "hard_ratio": float(result["hard"].mean()),
        }
        if channel_files:
            position_dir = output_dir / stem
            position_dir.mkdir(parents=True, exist_ok=True)
            arrays = {
                "min": result["lgi"][0],
                "max": result["lgi"][1],
                "nearest": result["lgi"][2],
                "valid": result["valid"],
                "min_abs": result["min_abs"],
                "hard": result["hard"],
            }
            channel_paths = {}
            for name, array in arrays.items():
                relative_path = Path(stem) / f"{name}.npy"
                np.save(output_dir / relative_path, array)
                channel_paths[name] = str(relative_path)
            camera_path = Path(stem) / "camera_light.npz"
            np.savez(
                output_dir / camera_path,
                intrinsic=intrinsic,
                world_to_cv=world_to_cv,
                light_world=np.asarray(light["world_position"], dtype=np.float32),
                light_cv=light_cv.astype(np.float32),
            )
            preview_path = Path(stem) / "preview.png"
            save_preview(output_dir / preview_path, result, stem)
            sample_output.update(
                {
                    "channels": channel_paths,
                    "camera_light": str(camera_path),
                    "preview": str(preview_path),
                }
            )
        else:
            npz_path = output_dir / f"{stem}.npz"
            np.savez_compressed(
                npz_path,
                **result,
                intrinsic=intrinsic,
                world_to_cv=world_to_cv,
                light_world=np.asarray(light["world_position"], dtype=np.float32),
                light_cv=light_cv.astype(np.float32),
            )
            preview_path = f"{stem}_preview.png"
            save_preview(output_dir / preview_path, result, stem)
            sample_output.update({"lgi_npz": npz_path.name, "preview": preview_path})
        gt_shadow_rel = sample.get("masks", {}).get("object_shadow_clean")
        if gt_shadow_rel and (scene_dir / gt_shadow_rel).is_file():
            gt_shadow = load_binary_mask(scene_dir / gt_shadow_rel)
            sample_output["blender_gt_shadow"] = {
                "path": gt_shadow_rel,
                **shadow_comparison(result["hard"].astype(bool), gt_shadow),
            }
        output_samples.append(sample_output)
        if verbose_lights:
            print(
                f"{stem}: valid={output_samples[-1]['valid_ratio']:.4f}, "
                f"hard={output_samples[-1]['hard_ratio']:.4f}"
            )

    index = {
        "schema": "tokenlight_lgi_fixed32_v1",
        "scene_id": meta["scene_id"],
        "source_scene": str(scene_dir),
        "depth_semantics": "Blender Z-pass ray distance converted to OpenCV optical-axis depth",
        "coordinate_system": "OpenCV camera: +X right, +Y down, +Z forward",
        "lgi_channels": ["min_angle_difference", "max_angle_difference", "signed_nearest_to_zero"],
        "angle_unit": "radian",
        "num_ray_samples": num_ray_samples,
        "hard_threshold_degrees": hard_threshold_deg,
        "intrinsic": intrinsic.tolist(),
        "world_to_cv": world_to_cv.tolist(),
        "samples": output_samples,
    }
    overview_filename = "lgi_hard_overview.png" if index_filename == "lgi_index.json" else "hard_overview.png"
    save_hard_overview(output_dir / overview_filename, output_dir, output_samples, width, height)
    index["storage_layout"] = "separate_channel_npy" if channel_files else "compressed_npz"
    index["hard_overview"] = overview_filename
    (output_dir / index_filename).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return {
        "scene_id": meta["scene_id"],
        "source_scene": str(scene_dir),
        "output_dir": str(output_dir),
        "position_count": len(output_samples),
    }


def discover_scenes(dataset_root: Path) -> list[Path]:
    scenes = {
        meta_path.parent.resolve()
        for meta_path in dataset_root.resolve().rglob("meta.json")
        if meta_path.parent.name.startswith("scene_")
    }
    return sorted(scenes, key=lambda path: (path.name, str(path)))


def output_is_complete(
    scene_dir: Path,
    output_dir: Path,
    selected_position_ids: tuple[int, ...] | None,
    num_ray_samples: int,
    hard_threshold_deg: float,
    index_filename: str,
) -> bool:
    index_path = output_dir / index_filename
    if not index_path.is_file():
        return False
    try:
        scene_meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = position_samples(
            scene_meta,
            set(selected_position_ids) if selected_position_ids else None,
        )
        outputs = index.get("samples", [])
        return (
            index.get("num_ray_samples") == num_ray_samples
            and math.isclose(float(index.get("hard_threshold_degrees")), hard_threshold_deg)
            and len(outputs) == len(expected)
            and all(
                (
                    (output_dir / sample["lgi_npz"]).is_file()
                    if "lgi_npz" in sample
                    else all((output_dir / path).is_file() for path in sample["channels"].values())
                )
                for sample in outputs
            )
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def run_scene_job(job: tuple) -> dict:
    return generate_scene(*job)


def main() -> None:
    args = parse_args()
    if args.num_ray_samples <= 0 or args.tile_rows <= 0 or args.workers <= 0:
        raise SystemExit("--num-ray-samples, --tile-rows, and --workers must be positive")
    if args.scene_dir and args.output_root:
        raise SystemExit("--output-root is only valid with --dataset-root")
    if args.dataset_root and args.output_dir:
        raise SystemExit("--output-dir is only valid with --scene-dir")
    if args.in_place and not args.dataset_root:
        raise SystemExit("--in-place requires --dataset-root")
    if args.in_place and args.output_root:
        raise SystemExit("--in-place writes to the dataset, so do not pass --output-root")
    if args.in_place:
        args.channel_files = True

    selected_position_ids = tuple(args.position_id) if args.position_id else None
    if args.scene_dir:
        scene_dir = args.scene_dir.resolve()
        output_dir = (args.output_dir or (scene_dir / "lgi_fixed32")).resolve()
        result = generate_scene(
            scene_dir,
            output_dir,
            args.num_ray_samples,
            args.tile_rows,
            args.hard_threshold_deg,
            selected_position_ids,
            True,
            args.channel_files,
            "index.json",
        )
        print(f"Wrote {result['position_count']} LGI maps to {output_dir}")
        return

    dataset_root = args.dataset_root.resolve()
    output_root = (
        dataset_root
        if args.in_place
        else (args.output_root or dataset_root.with_name(f"{dataset_root.name}_lgi_fixed32"))
    ).resolve()
    scene_index_filename = "lgi_index.json" if args.in_place else "index.json"
    scenes = discover_scenes(dataset_root)
    if not scenes:
        raise SystemExit(f"No scene_*/meta.json found below {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    completed = []
    skipped = []
    failed = []
    jobs = []
    for scene_dir in scenes:
        output_dir = scene_dir if args.in_place else output_root / "scenes" / scene_dir.name
        if not args.overwrite and output_is_complete(
            scene_dir,
            output_dir,
            selected_position_ids,
            args.num_ray_samples,
            args.hard_threshold_deg,
            scene_index_filename,
        ):
            skipped.append(scene_dir.name)
            continue
        jobs.append(
            (
                scene_dir,
                output_dir,
                args.num_ray_samples,
                args.tile_rows,
                args.hard_threshold_deg,
                selected_position_ids,
                False,
                args.channel_files,
                scene_index_filename,
            )
        )

    print(
        f"Found {len(scenes)} scenes: {len(jobs)} pending, "
        f"{len(skipped)} already complete; workers={args.workers}"
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(run_scene_job, job): job for job in jobs}
        for done_index, future in enumerate(concurrent.futures.as_completed(future_to_job), start=1):
            scene_dir = future_to_job[future][0]
            try:
                result = future.result()
                completed.append(result)
                print(
                    f"[{done_index}/{len(jobs)}] {result['scene_id']}: "
                    f"{result['position_count']} LGI maps"
                )
            except Exception as exc:
                failure = {
                    "scene_id": scene_dir.name,
                    "source_scene": str(scene_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failed.append(failure)
                print(f"[{done_index}/{len(jobs)}] {scene_dir.name}: FAILED: {failure['error']}")
                if args.fail_fast:
                    for pending in future_to_job:
                        pending.cancel()
                    break

    dataset_index = {
        "schema": "tokenlight_lgi_fixed32_dataset_v1",
        "source_dataset": str(dataset_root),
        "output_root": str(output_root),
        "scene_count": len(scenes),
        "completed_count": len(completed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "num_ray_samples": args.num_ray_samples,
        "hard_threshold_degrees": args.hard_threshold_deg,
        "selected_position_ids": list(selected_position_ids) if selected_position_ids else None,
        "storage_layout": "separate_channel_npy" if args.channel_files else "compressed_npz",
        "in_place": args.in_place,
        "completed": sorted(completed, key=lambda item: item["scene_id"]),
        "skipped": sorted(skipped),
        "failed": sorted(failed, key=lambda item: item["scene_id"]),
    }
    dataset_index_path = output_root / ("lgi_dataset_index.json" if args.in_place else "dataset_index.json")
    dataset_index_path.write_text(
        json.dumps(dataset_index, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Dataset complete: completed={len(completed)}, skipped={len(skipped)}, "
        f"failed={len(failed)}; index={dataset_index_path}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
