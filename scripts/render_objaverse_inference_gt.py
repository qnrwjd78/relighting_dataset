"""Replay selected Objaverse scenes and render the 20x20 inference light grid.

This script must be run with Blender Python.  It deliberately reuses the final
Objaverse renderer's scene replay and PNG conversion code so that geometry,
camera, receiver materials, render settings, and output encoding stay aligned
with the existing final dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

try:
    import bpy
    from mathutils import Vector
except ModuleNotFoundError as exc:  # pragma: no cover - Blender-only entry point
    raise SystemExit("render_objaverse_inference_gt.py must be run by Blender Python") from exc


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
DEFAULT_SCENES = [
    "scene_002512",
    "scene_002525",
    "scene_002572",
    "scene_002583",
    "scene_002592",
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render exact GT for the 20x20 zigzag inference light grid")
    parser.add_argument("--source-root", required=True, help="Dataset root containing scenes/scene_*/meta.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-config", default="configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--light-start", type=int, default=0, help="Inclusive light index")
    parser.add_argument("--light-end", type=int, default=400, help="Exclusive light index")
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-energy", type=float, default=500.0)
    parser.add_argument("--power-scale", type=float, default=0.7)
    parser.add_argument("--canonical-radius", type=float, default=0.06)
    parser.add_argument("--canonical-z", type=float, default=0.75)
    parser.add_argument("--ambient-color", nargs=3, type=float, default=[0.78, 0.78, 0.78])
    parser.add_argument("--png-gamma", type=float, default=2.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.light_start < args.light_end <= 400:
        parser.error("light range must satisfy 0 <= --light-start < --light-end <= 400")
    if args.resolution <= 0 or args.samples <= 0:
        parser.error("--resolution and --samples must be positive")
    return args


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def scene_id(value: str) -> str:
    token = str(value).strip()
    if token.startswith("scene_"):
        token = token.split("_", 1)[1]
    return f"scene_{int(token):06d}"


def zigzag_grid(z: float) -> list[list[float]]:
    values = [(-95 + 10 * index) / 100.0 for index in range(20)]
    points = []
    for x_index, x in enumerate(values):
        ys = values if x_index % 2 == 0 else reversed(values)
        points.extend([[float(x), float(y), float(z)] for y in ys])
    if len(points) != 400:
        raise AssertionError(f"Expected 400 grid points, got {len(points)}")
    return points


def bbox_error(actual_min: Vector, actual_max: Vector, object_meta: dict) -> float | None:
    if not object_meta.get("bbox_min") or not object_meta.get("bbox_max"):
        return None
    expected = [*object_meta["bbox_min"], *object_meta["bbox_max"]]
    actual = [*actual_min, *actual_max]
    return max(abs(float(a) - float(b)) for a, b in zip(actual, expected))


def render_scene(args: argparse.Namespace, source_root: Path, output_root: Path, name: str) -> dict:
    import render_objaverse_fixed_dataset as fixed
    import render_objaverse_random_dataset as relight

    meta_path = source_root / "scenes" / name / "meta.json"
    source_meta = read_json(meta_path)
    if source_meta.get("scene_id") != name:
        raise RuntimeError(f"Scene id mismatch in {meta_path}")
    asset_path = fixed.resolve_meta_asset(ROOT, source_meta)
    if asset_path is None or not asset_path.is_file():
        raise FileNotFoundError(f"Missing asset for {name}: {asset_path}")

    base_config = read_json(resolve(args.base_config))
    config_args = SimpleNamespace(
        resolution=int(args.resolution),
        samples=int(args.samples),
        component_format="png",
        mask_preset="grid-v4",
    )
    config = fixed.build_replay_config(source_meta, config_args)
    # build_replay_config intentionally keeps only generic canonical defaults.
    # Exact inference GT needs the dataset's axis_scale (1.244444... on this set).
    config["canonical"] = copy.deepcopy(base_config["canonical"])
    target_size = float(source_meta.get("object", {}).get("target_size", 1.2))
    # The original replay path falls back to target_size when min_size was not
    # recorded.  This matters for small source GLBs such as scene_002572.
    config["object"]["min_size"] = float(source_meta.get("object", {}).get("min_size", target_size))
    config["object"]["target_size"] = target_size
    config["render"]["component_png_gamma"] = float(args.png_gamma)
    config["_component_format"] = "png"

    rng = random.Random(int(args.seed) + int(name.split("_")[-1]))
    relight.clear_scene()
    relight.setup_render_settings(config)
    gpu_meta = fixed.configure_gpu_devices(args.gpu_devices)

    objects = relight.import_asset_or_primitive(
        str(asset_path),
        source_meta.get("object", {}).get("primitive") or "sphere",
        rng,
        config,
    )
    bbox_min, bbox_max = relight.mesh_bbox(objects)
    error = bbox_error(bbox_min, bbox_max, source_meta.get("object", {}))
    if error is not None and error > 1.0e-4:
        raise RuntimeError(f"{name}: reconstructed object bbox differs from meta (max error {error:.8g})")

    relight.set_canonical_runtime_transform(config, bbox_min, bbox_max)
    camera_meta = source_meta.get("camera", {})
    similarity = camera_meta.get("similarity_transform")
    if not similarity or not similarity.get("axes"):
        raise RuntimeError(f"{name}: camera.similarity_transform.axes is required")
    config["_runtime"]["similarity_transform"] = copy.deepcopy(similarity)
    config["_runtime"]["canonical_scale"] = float(similarity["scale"])
    camera = fixed.create_camera_from_meta(relight, camera_meta)
    center = (bbox_min + bbox_max) * 0.5
    relight.create_receivers(config, rng, camera, center)
    relight.remove_all_lights()

    ambient_strength = float(source_meta["source"]["ambient_source"]["strength"])
    ambient_color = tuple(float(value) for value in args.ambient_color)
    relight.set_constant_world(ambient_color, ambient_strength)

    axis_scale = relight.canonical_axis_scale(config)
    scale_xyz = [float(similarity["scale"]) * float(axis_scale[i]) for i in range(3)]
    light_scale = abs(scale_xyz[0] * scale_xyz[1] * scale_xyz[2]) ** (1.0 / 3.0)
    world_energy = float(args.base_energy) * light_scale * light_scale * float(args.power_scale)
    world_radius = float(args.canonical_radius) * light_scale
    points = zigzag_grid(float(args.canonical_z))

    frames = []
    for index in range(int(args.light_start), int(args.light_end)):
        canonical = points[index]
        output_path = output_root / f"{name}_light_{index:03d}.png"
        world = relight.transformed_canonical_point(config, canonical, apply_axis_scale=True)
        row = {
            "index": index,
            "canonical_position": canonical,
            "world_position": [float(world.x), float(world.y), float(world.z)],
            "world_energy": world_energy,
            "world_radius": world_radius,
            "image": output_path.name,
            "skipped": bool(output_path.is_file() and not args.overwrite),
        }
        if not row["skipped"]:
            light = relight.create_point_light(
                f"TL_InferenceGT_{name}_{index:03d}",
                world,
                world_energy,
                world_radius,
                [1.0, 1.0, 1.0],
            )
            try:
                relight.render_component(output_root, output_path.stem, config)
            finally:
                bpy.data.objects.remove(light, do_unlink=True)
        frames.append(row)
        print(f"[inference-gt] {name} light={index:03d} {'skip' if row['skipped'] else 'done'}", flush=True)

    return {
        "scene_id": name,
        "source_meta": str(meta_path),
        "object_path": str(asset_path),
        "camera_similarity_transform": similarity,
        "canonical_axis_scale": [float(axis_scale.x), float(axis_scale.y), float(axis_scale.z)],
        "light_scale": light_scale,
        "world_energy": world_energy,
        "world_radius": world_radius,
        "ambient": {"type": "constant", "color": list(ambient_color), "strength": ambient_strength},
        "render": {"resolution": [args.resolution, args.resolution], "samples": args.samples, "gpu": gpu_meta},
        "frames": frames,
    }


def main() -> int:
    args = parse_args()
    source_root = resolve(args.source_root)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_scenes = [scene_id(value) for value in args.scenes]
    completed = []
    failed = []
    for name in normalized_scenes:
        try:
            completed.append(render_scene(args, source_root, output_root, name))
        except Exception as exc:
            failed.append({"scene_id": name, "error": str(exc), "traceback": traceback.format_exc()})
            print(f"[inference-gt] FAILED {name}: {exc}", file=sys.stderr, flush=True)

    worker_meta = {
        "schema": "objaverse_inference_gt_worker_v1",
        "light_range": [args.light_start, args.light_end],
        "completed": completed,
        "failed": failed,
    }
    worker_path = output_root / "_workers" / f"lights_{args.light_start:03d}_{args.light_end - 1:03d}.json"
    write_json(worker_path, worker_meta)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
