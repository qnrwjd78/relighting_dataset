#!/usr/bin/env python3
"""Convert completed EXR dataset scenes to a structurally equivalent PNG dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LIGHTING_GAMMA = 2.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EXRs in completed scenes to PNG while preserving scene-relative paths. "
            "Only scene_* directories containing meta.json are processed."
        )
    )
    parser.add_argument("--input", required=True, help="Dataset root containing scenes/ or shard_*/scenes/.")
    parser.add_argument("--output", required=True, help="Output root. Converted scenes are written under scenes/.")
    parser.add_argument(
        "--gpu-shard",
        action="store_true",
        help="Read shard_*/scenes/ below --input and merge all completed scenes into OUTPUT/scenes/.",
    )
    parser.add_argument(
        "--with-white",
        action="store_true",
        help="Include position_*_white and mask_reference/ambient_white intermediate renders.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-limit", type=int, default=0, help="0 converts every completed scene.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy existing PNG/JSON/NPY files instead of hardlinking them when possible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_roots(input_root: Path, gpu_shard: bool) -> list[Path]:
    if gpu_shard:
        roots = sorted(path for path in input_root.glob("shard_*") if (path / "scenes").is_dir())
        if not roots:
            raise SystemExit(f"--gpu-shard was set, but no shard_*/scenes directories exist in {input_root}")
        return roots
    if not (input_root / "scenes").is_dir():
        raise SystemExit(f"Expected {input_root / 'scenes'}. Use --gpu-shard for shard_*/scenes input.")
    return [input_root]


def completed_scenes(roots: list[Path], offset: int, limit: int) -> list[tuple[Path, str]]:
    scenes: list[tuple[Path, str]] = []
    seen: dict[str, Path] = {}
    for root in roots:
        for meta_path in sorted((root / "scenes").glob("scene_*/meta.json")):
            scene_dir = meta_path.parent
            scene_id = scene_dir.name
            if scene_id in seen:
                raise SystemExit(f"Duplicate completed scene {scene_id}: {seen[scene_id]} and {scene_dir}")
            seen[scene_id] = scene_dir
            scenes.append((scene_dir, root.name))
    scenes.sort(key=lambda row: row[0].name)
    if offset > 0:
        scenes = scenes[offset:]
    if limit > 0:
        scenes = scenes[:limit]
    if not scenes:
        raise SystemExit("No completed scene_*/meta.json files were found.")
    return scenes


def is_white_intermediate(relative_path: Path) -> bool:
    return any(part == "white" or part.endswith("_white") for part in relative_path.parts)


def read_exr(path: Path) -> tuple[np.ndarray, list[str]]:
    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        available = set(header["channels"])
        if all(channel in available for channel in ("R", "G", "B")):
            names = ["R", "G", "B"]
        elif all(channel in available for channel in ("X", "Y", "Z")):
            names = ["X", "Y", "Z"]
        elif "Z" in available:
            names = ["Z"]
        elif "R" in available:
            names = ["R"]
        elif available:
            names = [sorted(available)[0]]
        else:
            raise RuntimeError(f"EXR has no channels: {path}")
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [
            np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width)
            for name in names
        ]
        image = np.stack(channels, axis=-1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image[:, :, :3], names
    finally:
        exr.close()


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_lighting(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    linear = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    linear = np.maximum(linear, 0.0)
    mapped = linear / (1.0 + linear)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / LIGHTING_GAMMA)
    return mapped, {"encoding": "reinhard_gamma", "gamma": LIGHTING_GAMMA}


def encode_depth(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    depth = image[:, :, 0].astype(np.float32)
    finite = np.isfinite(depth) & (depth > 0.0) & (depth < 1.0e6)
    if not np.any(finite):
        encoded = np.zeros_like(depth)
        stats = {"min": 0.0, "max": 1.0, "valid_pixel_ratio": 0.0}
    else:
        valid = depth[finite]
        depth_min = float(np.percentile(valid, 1.0))
        depth_max = float(np.percentile(valid, 99.0))
        if depth_max <= depth_min + 1.0e-6:
            depth_max = depth_min + 1.0
        encoded = 1.0 - (depth - depth_min) / (depth_max - depth_min)
        encoded = np.where(finite, encoded, 0.0)
        stats = {
            "min": depth_min,
            "max": depth_max,
            "valid_pixel_ratio": float(np.mean(finite)),
        }
    rgb = np.repeat(np.clip(encoded, 0.0, 1.0)[:, :, None], 3, axis=2)
    return rgb, {"encoding": "depth_percentile_1_99_near_white", **stats}


def encode_pbr(kind: str, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    clean = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if kind == "depth":
        return encode_depth(clean)
    if kind == "normal":
        if float(np.min(clean)) < -1.0e-4 or float(np.max(clean)) > 1.0 + 1.0e-4:
            clean = clean * 0.5 + 0.5
            encoding = "normal_xyz_minus1_1_to_0_1"
        else:
            encoding = "normal_rgb_0_1"
        return np.clip(clean, 0.0, 1.0), {"encoding": encoding}
    if kind == "roughness":
        gray = np.mean(clean[:, :, :3], axis=2, keepdims=True)
        return np.repeat(np.clip(gray, 0.0, 1.0), 3, axis=2), {"encoding": "roughness_grayscale_0_1"}
    return np.clip(clean[:, :, :3], 0.0, 1.0), {"encoding": f"{kind}_rgb_0_1"}


def exr_kind(relative_path: Path) -> str:
    if len(relative_path.parts) >= 2 and relative_path.parts[0] == "pbr":
        return relative_path.stem.lower()
    return "lighting"


def save_converted_exr(source: Path, dest: Path, relative_path: Path, overwrite: bool) -> dict[str, Any]:
    if dest.exists() and not overwrite:
        return {"image": relative_path.with_suffix(".png").as_posix(), "skipped": True}
    image, channels = read_exr(source)
    kind = exr_kind(relative_path)
    encoded, encoding = encode_lighting(image) if kind == "lighting" else encode_pbr(kind, image)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(encoded), mode="RGB").save(dest, compress_level=1)
    return {
        "image": relative_path.with_suffix(".png").as_posix(),
        "kind": kind,
        "channels": channels,
        **encoding,
    }


def link_or_copy(source: Path, dest: Path, copy_files: bool, overwrite: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return
        dest.unlink()
    if copy_files:
        shutil.copy2(source, dest)
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


DROP = object()


def rewrite_exr_references(value: Any, converted: set[str], excluded: set[str]) -> Any:
    if isinstance(value, list):
        result = [rewrite_exr_references(item, converted, excluded) for item in value]
        return [item for item in result if item is not DROP]
    if not isinstance(value, dict):
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            if normalized in converted:
                return str(Path(value).with_suffix(".png")).replace("\\", "/")
            if normalized in excluded:
                return DROP
        return value

    result: dict[str, Any] = {}
    for key, item in value.items():
        new_key = key
        if key == "exr" and isinstance(item, str) and item.replace("\\", "/") in converted:
            new_key = "png"
        elif key == "render_exr" and isinstance(item, str) and item.replace("\\", "/") in converted:
            new_key = "render_png"
        rewritten = rewrite_exr_references(item, converted, excluded)
        if rewritten is not DROP:
            result[new_key] = rewritten
    if value and not result:
        return DROP
    return result


def convert_scene(
    scene_dir_raw: str,
    output_root_raw: str,
    shard_name: str,
    with_white: bool,
    copy_files: bool,
    overwrite: bool,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir_raw)
    scene_dest = Path(output_root_raw) / "scenes" / scene_dir.name
    source_meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))

    if scene_dest.exists() and not with_white:
        white_dirs = [
            path
            for path in scene_dest.rglob("*")
            if path.is_dir() and is_white_intermediate(path.relative_to(scene_dest))
        ]
        for path in sorted(white_dirs, key=lambda item: len(item.parts), reverse=True):
            if path.exists():
                shutil.rmtree(path)

    files = [path for path in scene_dir.rglob("*") if path.is_file()]
    included = []
    for path in files:
        relative = path.relative_to(scene_dir)
        if not with_white and is_white_intermediate(relative):
            continue
        included.append((path, relative))

    exr_png_targets = {
        relative.with_suffix(".png") for source, relative in included if source.suffix.lower() == ".exr"
    }
    copied = 0
    for source, relative in included:
        if source.suffix.lower() == ".exr" or relative == Path("meta.json"):
            continue
        if relative in exr_png_targets:
            continue
        link_or_copy(source, scene_dest / relative, copy_files, overwrite)
        copied += 1

    conversions = []
    converted_paths: set[str] = set()
    excluded_paths = {
        path.relative_to(scene_dir).as_posix()
        for path in files
        if path.suffix.lower() == ".exr"
        and not with_white
        and is_white_intermediate(path.relative_to(scene_dir))
    }
    for source, relative in included:
        if source.suffix.lower() != ".exr":
            continue
        converted_paths.add(relative.as_posix())
        conversions.append(
            save_converted_exr(source, scene_dest / relative.with_suffix(".png"), relative, overwrite)
        )

    output_meta = rewrite_exr_references(source_meta, converted_paths, excluded_paths)
    if output_meta is DROP:
        raise RuntimeError(f"Metadata became empty after path rewriting: {scene_dir / 'meta.json'}")
    output_meta.setdefault("render", {})["component_format"] = "png"
    output_meta["png_conversion"] = {
        "source_scene": str(scene_dir),
        "source_shard": shard_name,
        "with_white": with_white,
        "converted_exr_count": len(conversions),
        "lighting_tonemap": "reinhard_gamma",
        "lighting_gamma": LIGHTING_GAMMA,
        "files": conversions,
    }
    scene_dest.mkdir(parents=True, exist_ok=True)
    (scene_dest / "meta.json").write_text(
        json.dumps(output_meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return {
        "scene_id": scene_dir.name,
        "source_scene": str(scene_dir),
        "source_shard": shard_name,
        "meta": f"scenes/{scene_dir.name}/meta.json",
        "converted_exr_count": len(conversions),
        "copied_file_count": copied,
    }


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise SystemExit("--output must be outside --input")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    roots = source_roots(input_root, args.gpu_shard)
    scenes = completed_scenes(roots, args.scene_offset, args.scene_limit)
    print(f"[INFO] input={input_root}")
    print(f"[INFO] output={output_root}")
    print(
        f"[INFO] gpu_shard={args.gpu_shard} roots={len(roots)} completed_scenes={len(scenes)} "
        f"with_white={args.with_white} workers={args.workers}"
    )
    for root in roots:
        count = sum(1 for scene, _ in scenes if scene.parent.parent == root)
        print(f"[INFO] source_root={root.name} selected_scenes={count}")
    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                convert_scene,
                str(scene_dir),
                str(output_root),
                shard_name,
                args.with_white,
                args.copy_files,
                args.overwrite,
            ): scene_dir
            for scene_dir, shard_name in scenes
        }
        for index, future in enumerate(as_completed(futures), 1):
            scene_dir = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = f"converted_exr={result['converted_exr_count']}"
            except Exception as exc:
                failures.append({"scene_id": scene_dir.name, "source_scene": str(scene_dir), "error": str(exc)})
                status = f"ERROR={type(exc).__name__}: {exc}"
            if index == 1 or index % 25 == 0 or index == len(futures):
                print(f"[PROGRESS] scenes={index}/{len(futures)} last={scene_dir.name} {status}", flush=True)

    results.sort(key=lambda row: row["scene_id"])
    manifest = {
        "schema": "relighting_exr_to_png_dataset_v1",
        "input": str(input_root),
        "output": str(output_root),
        "gpu_shard": args.gpu_shard,
        "source_roots": [str(root) for root in roots],
        "with_white": args.with_white,
        "scene_count_requested": len(scenes),
        "scene_count_written": len(results),
        "scene_count_failed": len(failures),
        "converted_exr_count": sum(row["converted_exr_count"] for row in results),
        "scenes": results,
        "failed_scenes": failures,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        f"[DONE] scenes={len(results)}/{len(scenes)} failed={len(failures)} "
        f"converted_exr={manifest['converted_exr_count']} output={output_root}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
