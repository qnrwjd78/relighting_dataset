"""Launch fixed Objaverse rendering as one Blender worker per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECT_MANIFEST = "metadata/objaverse_xl/front2000_objects.txt"
DEFAULT_BASE_CONFIG = "configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split fixed Objaverse scenes across GPUs and run one Blender process per GPU."
    )
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical GPU ids, for example: --gpus 0 1 2 3")
    parser.add_argument("--object-manifest", default=DEFAULT_OBJECT_MANIFEST)
    parser.add_argument(
        "--object-scene-list",
        default=None,
        help="Optional sparse scene-id list selecting rows from the full object manifest.",
    )
    parser.add_argument(
        "--object-offset",
        "--start-index",
        dest="object_offset",
        type=int,
        default=0,
        help="Global manifest index of the first object (default: 0).",
    )
    parser.add_argument("--object-count", type=int, default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default="exr")
    parser.add_argument("--object-min-size", type=float, default=0.6)
    parser.add_argument("--object-target-size", type=float, default=0.9)
    parser.add_argument("--rig-reference-object-size", type=float, default=1.2)
    parser.add_argument("--canonical-world-scale", type=float, default=0.75)
    parser.add_argument("--power-values", nargs=4, type=float, default=[0.30, 0.60, 0.90, 1.20])
    parser.add_argument("--shadow-threshold", type=float, default=0.05)
    parser.add_argument("--shadow-support-threshold", type=float, default=0.001)
    parser.add_argument("--shadow-mask-mode", choices=["photometric", "geometry-ray"], default="photometric")
    parser.add_argument("--object-mask-erode-radius", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--pad-radius", type=int, default=16)
    parser.add_argument(
        "--shadow-loss-debug-top-k",
        type=int,
        default=0,
        help="Render shadow-off debug passes for the K largest geometry shadow masks per scene.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--ambient-manifest", default=None)
    parser.add_argument("--receiver-texture-manifest", default=None)
    parser.add_argument("--no-receiver-textures", action="store_true")
    parser.add_argument(
        "--expected-manifest-sha256",
        default=None,
        help="Abort unless the object manifest has this SHA256 fingerprint.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Render completed scenes again instead of skipping them.")
    parser.add_argument("--dry-run", action="store_true", help="Print worker commands without starting Blender.")
    args = parser.parse_args()

    if args.object_offset < 0:
        parser.error("--object-offset must be nonnegative")
    if args.object_count is not None and args.object_count <= 0:
        parser.error("--object-count must be positive")
    if args.object_min_size <= 0.0 or args.object_target_size <= 0.0:
        parser.error("object sizes must be positive")
    if args.object_min_size > args.object_target_size:
        parser.error("--object-min-size must not exceed --object-target-size")
    if args.object_mask_erode_radius < 0:
        parser.error("--object-mask-erode-radius must be nonnegative")
    if args.shadow_loss_debug_top_k < 0:
        parser.error("--shadow-loss-debug-top-k must be nonnegative")
    if args.object_scene_list and (args.object_offset != 0 or args.object_count is not None):
        parser.error("--object-scene-list cannot be combined with --object-offset/--start-index or --object-count")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must not contain duplicate ids")
    return args


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_manifest_rows(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Object manifest does not exist: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def split_ranges(offset: int, count: int, gpus: list[str]) -> list[dict]:
    worker_count = min(len(gpus), count)
    base, remainder = divmod(count, worker_count)
    ranges = []
    cursor = offset
    for shard_id, gpu in enumerate(gpus[:worker_count]):
        limit = base + (1 if shard_id < remainder else 0)
        ranges.append({"shard_id": shard_id, "gpu": gpu, "offset": cursor, "limit": limit})
        cursor += limit
    return ranges


def read_scene_ids(path: Path, manifest_count: int) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Object scene list does not exist: {path}")
    scene_ids = []
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
            seen.add(scene_id)
            scene_ids.append(scene_id)
    if not scene_ids:
        raise SystemExit(f"Object scene list is empty: {path}")
    return scene_ids


def split_scene_ids(scene_ids: list[str], gpus: list[str]) -> list[dict]:
    worker_count = min(len(gpus), len(scene_ids))
    base, remainder = divmod(len(scene_ids), worker_count)
    items = []
    cursor = 0
    for shard_id, gpu in enumerate(gpus[:worker_count]):
        count = base + (1 if shard_id < remainder else 0)
        selected = scene_ids[cursor : cursor + count]
        items.append({"shard_id": shard_id, "gpu": gpu, "scene_ids": selected, "limit": count})
        cursor += count
    return items


def worker_command(args: argparse.Namespace, item: dict, manifest: Path, output_root: Path) -> list[str]:
    command = shlex.split(args.blender) + [
        "-b",
        "--python",
        str(ROOT / "scripts" / "render_objaverse_fixed_dataset.py"),
        "--",
        "--object-manifest",
        str(manifest),
        "--base-config",
        str(resolve_path(args.base_config)),
        "--output-root",
        str(output_root / f"shard_{item['shard_id']}"),
        "--object-min-size",
        str(args.object_min_size),
        "--object-target-size",
        str(args.object_target_size),
        "--rig-reference-object-size",
        str(args.rig_reference_object_size),
        "--canonical-world-scale",
        str(args.canonical_world_scale),
        "--resolution",
        str(args.resolution),
        "--samples",
        str(args.samples),
        "--component-format",
        args.component_format,
        "--gpu-devices",
        "0",
        "--seed",
        str(args.seed),
        "--fixed-upper-half-white-grid",
        "--power-values",
        *[str(value) for value in args.power_values],
        "--ambient-subtracted-shadow-ratio",
        "--shadow-threshold",
        str(args.shadow_threshold),
        "--shadow-support-threshold",
        str(args.shadow_support_threshold),
        "--shadow-mask-mode",
        args.shadow_mask_mode,
        "--object-mask-erode-radius",
        str(args.object_mask_erode_radius),
        "--min-area",
        str(args.min_area),
        "--pad-radius",
        str(args.pad_radius),
    ]
    if item.get("scene_list_path"):
        command.extend(["--object-scene-list", str(item["scene_list_path"])])
    else:
        command.extend(["--object-offset", str(item["offset"]), "--object-limit", str(item["limit"])])
    if args.shadow_loss_debug_top_k > 0:
        command.extend(["--shadow-loss-debug-top-k", str(args.shadow_loss_debug_top_k)])
    if not args.no_resume:
        command.append("--skip-completed")
    if args.ambient_manifest:
        command.extend(["--ambient-manifest", str(resolve_path(args.ambient_manifest))])
    if args.receiver_texture_manifest:
        command.extend(["--receiver-texture-manifest", str(resolve_path(args.receiver_texture_manifest))])
    if args.no_receiver_textures:
        command.append("--no-receiver-textures")
    return command


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def aggregate_results(
    args: argparse.Namespace,
    output_root: Path,
    ranges: list[dict],
    return_codes: dict[int, int],
    manifest_sha256: str,
) -> tuple[dict, bool]:
    scenes = []
    failed = []
    missing = []
    workers = []

    for item in ranges:
        shard_id = int(item["shard_id"])
        shard_root = output_root / f"shard_{shard_id}"
        shard_manifest_path = shard_root / "dataset_manifest.json"
        shard_manifest = load_json(shard_manifest_path)
        workers.append(
            {
                **item,
                "return_code": return_codes.get(shard_id),
                "manifest": str(shard_manifest_path.relative_to(output_root)),
                "manifest_valid": shard_manifest is not None,
            }
        )
        scene_ids = item.get("scene_ids") or [
            f"scene_{index:06d}"
            for index in range(int(item["offset"]), int(item["offset"]) + int(item["limit"]))
        ]
        for scene_id in scene_ids:
            meta_path = shard_root / "scenes" / scene_id / "meta.json"
            error_path = shard_root / "failed_scenes" / f"{scene_id}_error.json"
            meta = load_json(meta_path)
            if meta is not None and meta.get("scene_id") == scene_id:
                scenes.append(
                    {
                        "scene_id": scene_id,
                        "shard_id": shard_id,
                        "meta": str(meta_path.relative_to(output_root)),
                    }
                )
                continue
            error = load_json(error_path)
            if error is not None:
                failed.append({**error, "shard_id": shard_id})
            else:
                missing.append({"scene_id": scene_id, "shard_id": shard_id})

    scenes.sort(key=lambda row: row["scene_id"])
    failed.sort(key=lambda row: row["scene_id"])
    missing.sort(key=lambda row: row["scene_id"])
    fatal_worker = any(not worker["manifest_valid"] for worker in workers)
    complete = not missing and not fatal_worker
    manifest = {
        "schema": "tokenlight_fixed_objaverse_multi_gpu_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object_manifest": str(resolve_path(args.object_manifest)),
        "object_manifest_sha256": manifest_sha256,
        "object_offset": None if args.object_scene_list else int(args.object_offset),
        "object_scene_list": str(resolve_path(args.object_scene_list)) if args.object_scene_list else None,
        "object_count_requested": sum(int(item["limit"]) for item in ranges),
        "scene_count_completed": len(scenes),
        "scene_count_failed": len(failed),
        "scene_count_missing": len(missing),
        "resume_enabled": not args.no_resume,
        "workers": workers,
        "scenes": scenes,
        "failed_scenes": failed,
        "missing_scenes": missing,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, complete


def terminate_workers(processes: list[tuple[dict, subprocess.Popen]]) -> None:
    for _item, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 10.0
    for _item, process in processes:
        timeout = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
    for _item, process in processes:
        process.wait()


def main() -> int:
    args = parse_args()
    manifest = resolve_path(args.object_manifest)
    rows = read_manifest_rows(manifest)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if args.expected_manifest_sha256 and args.expected_manifest_sha256.lower() != manifest_sha256:
        raise SystemExit(
            f"Object manifest SHA256 mismatch: expected {args.expected_manifest_sha256.lower()}, "
            f"got {manifest_sha256}"
        )
    output_root = resolve_path(args.output_root)
    if args.object_scene_list:
        scene_list_path = resolve_path(args.object_scene_list)
        selected_scene_ids = read_scene_ids(scene_list_path, len(rows))
        count = len(selected_scene_ids)
        ranges = split_scene_ids(selected_scene_ids, args.gpus)
        list_dir = output_root / "_scene_lists"
        list_dir.mkdir(parents=True, exist_ok=True)
        for item in ranges:
            worker_list = list_dir / f"shard_{int(item['shard_id'])}.txt"
            worker_list.write_text(
                "".join(f"{scene_id}\n" for scene_id in item["scene_ids"]),
                encoding="utf-8",
            )
            item["scene_list_path"] = str(worker_list)
    else:
        available = len(rows) - args.object_offset
        if available <= 0:
            raise SystemExit(f"--object-offset {args.object_offset} is outside manifest with {len(rows)} rows")
        count = available if args.object_count is None else args.object_count
        if count > available:
            raise SystemExit(
                f"Requested {count} objects from offset {args.object_offset}, but only {available} are available"
            )
        ranges = split_ranges(args.object_offset, count, args.gpus)
    commands = [(item, worker_command(args, item, manifest, output_root)) for item in ranges]

    print(
        f"[multi-gpu] objects={count} offset={args.object_offset} workers={len(commands)} "
        f"resume={not args.no_resume} output={output_root}\n"
        f"[multi-gpu] object_manifest_sha256={manifest_sha256}",
        flush=True,
    )
    for item, command in commands:
        if item.get("scene_ids"):
            selection = (
                f"scenes={len(item['scene_ids'])} "
                f"first={item['scene_ids'][0]} last={item['scene_ids'][-1]}"
            )
        else:
            selection = f"range=[{item['offset']}, {item['offset'] + item['limit']})"
        print(
            f"[multi-gpu] shard_{item['shard_id']} gpu={item['gpu']} "
            f"{selection}\n  {shlex.join(command)}",
            flush=True,
        )
    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[dict, subprocess.Popen]] = []
    try:
        for item, command in commands:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
            process = subprocess.Popen(command, cwd=ROOT, env=environment)
            processes.append((item, process))
            print(f"[multi-gpu] started shard_{item['shard_id']} pid={process.pid}", flush=True)
        return_codes = {}
        for item, process in processes:
            return_codes[int(item["shard_id"])] = process.wait()
            print(
                f"[multi-gpu] finished shard_{item['shard_id']} "
                f"return_code={return_codes[int(item['shard_id'])]}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n[multi-gpu] interrupted; stopping Blender workers...", file=sys.stderr, flush=True)
        terminate_workers(processes)
        return 130

    result, complete = aggregate_results(args, output_root, ranges, return_codes, manifest_sha256)
    print(
        f"[multi-gpu] summary completed={result['scene_count_completed']} "
        f"failed={result['scene_count_failed']} missing={result['scene_count_missing']}",
        flush=True,
    )
    print(f"[multi-gpu] wrote {output_root / 'dataset_manifest.json'}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
