#!/usr/bin/env python3
"""Launch balanced Blender workers for inference-grid GT rendering."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = ["scene_002512", "scene_002525", "scene_002572", "scene_002583", "scene_002592"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render inference-grid GT with one Blender process per GPU")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-config", default="configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-energy", type=float, default=500.0)
    parser.add_argument("--power-scale", type=float, default=0.7)
    parser.add_argument("--canonical-radius", type=float, default=0.06)
    parser.add_argument("--canonical-z", type=float, default=0.75)
    parser.add_argument("--ambient-color", nargs=3, type=float, default=[0.78, 0.78, 0.78])
    parser.add_argument("--png-gamma", type=float, default=2.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must not contain duplicates")
    return args


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def split_ranges(count: int, workers: int) -> list[tuple[int, int]]:
    base, remainder = divmod(count, workers)
    ranges = []
    start = 0
    for index in range(workers):
        size = base + (1 if index < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def normalize_scene_id(value: str) -> str:
    token = str(value).strip()
    if token.startswith("scene_"):
        token = token.split("_", 1)[1]
    return f"scene_{int(token):06d}"


def ensure_rgb_pngs(paths: list[Path]) -> int:
    """Match the existing converted dataset's 8-bit RGB PNG representation."""
    from PIL import Image

    converted = 0
    for path in paths:
        if not path.is_file():
            continue
        with Image.open(path) as image:
            if image.mode == "RGB":
                continue
            rgb = image.convert("RGB")
            rgb.save(path, compress_level=1)
            converted += 1
    return converted


def worker_command(args: argparse.Namespace, start: int, end: int) -> list[str]:
    command = shlex.split(args.blender) + [
        "-b",
        "--python",
        str(ROOT / "scripts" / "render_objaverse_inference_gt.py"),
        "--",
        "--source-root", str(resolve(args.source_root)),
        "--output-root", str(resolve(args.output_root)),
        "--base-config", str(resolve(args.base_config)),
        "--scenes", *args.scenes,
        "--light-start", str(start),
        "--light-end", str(end),
        "--resolution", str(args.resolution),
        "--samples", str(args.samples),
        "--gpu-devices", "0",
        "--seed", str(args.seed),
        "--base-energy", str(args.base_energy),
        "--power-scale", str(args.power_scale),
        "--canonical-radius", str(args.canonical_radius),
        "--canonical-z", str(args.canonical_z),
        "--ambient-color", *[str(value) for value in args.ambient_color],
        "--png-gamma", str(args.png_gamma),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    args.scenes = [normalize_scene_id(value) for value in args.scenes]
    worker_count = min(len(args.gpus), 400)
    ranges = split_ranges(400, worker_count)
    commands = [worker_command(args, start, end) for start, end in ranges]
    for gpu, (start, end), command in zip(args.gpus, ranges, commands):
        print(f"[multi-gpu] gpu={gpu} lights=[{start},{end})\n  {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0

    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, light_range, command in zip(args.gpus, ranges, commands):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(command, cwd=ROOT, env=environment)
        processes.append((gpu, light_range, process))
        print(f"[multi-gpu] started gpu={gpu} pid={process.pid}", flush=True)

    return_codes = []
    try:
        for gpu, light_range, process in processes:
            code = process.wait()
            return_codes.append(code)
            print(f"[multi-gpu] finished gpu={gpu} lights={light_range} return_code={code}", flush=True)
    except KeyboardInterrupt:
        for _gpu, _light_range, process in processes:
            if process.poll() is None:
                process.terminate()
        return 130

    expected = [output_root / f"{scene}_light_{index:03d}.png" for scene in args.scenes for index in range(400)]
    missing = [str(path) for path in expected if not path.is_file()]
    rgb_normalized_count = ensure_rgb_pngs(expected)
    manifest = {
        "schema": "objaverse_inference_gt_multi_gpu_v1",
        "source_root": str(resolve(args.source_root)),
        "output_root": str(output_root),
        "scenes": args.scenes,
        "gpus": args.gpus,
        "worker_ranges": [list(value) for value in ranges],
        "expected_image_count": len(expected),
        "completed_image_count": len(expected) - len(missing),
        "rgb_normalized_image_count": rgb_normalized_count,
        "missing_images": missing,
        "return_codes": return_codes,
    }
    with (output_root / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        f"[multi-gpu] completed={manifest['completed_image_count']}/{manifest['expected_image_count']} "
        f"manifest={output_root / 'dataset_manifest.json'}",
        flush=True,
    )
    return 1 if missing or any(return_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
