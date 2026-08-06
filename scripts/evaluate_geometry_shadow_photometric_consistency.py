#!/usr/bin/env python3
"""Measure rendered shadow loss inside geometry-ray shadow masks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


POSITION_RE = re.compile(r"position_(\d+)_no_object_shadow$")
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare shadow-on/off renders and measure how much real positive shadow loss "
            "falls inside each geometry-ray shadow mask."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Dataset roots containing scenes/ or shard_*/scenes/. Multiple roots are allowed.",
    )
    parser.add_argument("--output", required=True, help="Directory for JSONL, summary, and scene lists.")
    parser.add_argument(
        "--scene-list",
        default=None,
        help="Optional text file containing scene_XXXXXX ids to evaluate.",
    )
    parser.add_argument(
        "--relative-loss-threshold",
        type=float,
        default=0.02,
        help="Minimum (shadow_off - shadow_on) / shadow_off ratio for a shadow pixel.",
    )
    parser.add_argument(
        "--normalized-loss-threshold",
        type=float,
        default=0.005,
        help="Minimum positive loss divided by the shadow-off image p99 luminance.",
    )
    parser.add_argument(
        "--min-shadow-coverage",
        type=float,
        default=0.01,
        help="Minimum fraction of geometry-mask pixels passing both loss thresholds.",
    )
    parser.add_argument(
        "--min-geometry-pixels",
        type=int,
        default=64,
        help="Pairs with fewer geometry-mask pixels are inconclusive.",
    )
    parser.add_argument(
        "--png-encoding",
        choices=["auto", "reinhard-gamma", "srgb", "linear"],
        default="auto",
        help="How PNG lighting images should be decoded. Auto reads png_conversion from meta.json.",
    )
    args = parser.parse_args()
    for name in ("relative_loss_threshold", "normalized_loss_threshold", "min_shadow_coverage"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.min_geometry_pixels < 1:
        parser.error("--min-geometry-pixels must be positive")
    return args


def read_scene_filter(path: str | None) -> set[str] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"Scene list does not exist: {source}")
    return {
        line.strip().split()[0]
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def find_scene_dirs(roots: list[str]) -> dict[str, Path]:
    scenes: dict[str, Path] = {}
    for value in roots:
        root = Path(value)
        if not root.exists():
            raise SystemExit(f"Input root does not exist: {root}")
        for meta_path in root.glob("scenes/scene_*/meta.json"):
            scene_dir = meta_path.parent
            if scene_dir.name in scenes:
                raise SystemExit(f"Duplicate scene {scene_dir.name}: {scenes[scene_dir.name]} and {scene_dir}")
            scenes[scene_dir.name] = scene_dir
        for meta_path in root.glob("shard_*/scenes/scene_*/meta.json"):
            scene_dir = meta_path.parent
            if scene_dir.name in scenes:
                raise SystemExit(f"Duplicate scene {scene_dir.name}: {scenes[scene_dir.name]} and {scene_dir}")
            scenes[scene_dir.name] = scene_dir
    return dict(sorted(scenes.items()))


def read_exr(path: Path) -> np.ndarray:
    try:
        import Imath
        import OpenEXR
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading EXR requires the OpenEXR and Imath Python modules") from exc

    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        available = set(header["channels"])
        if not all(channel in available for channel in ("R", "G", "B")):
            raise RuntimeError(f"EXR does not contain RGB channels: {path}")
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [
            np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width)
            for name in ("R", "G", "B")
        ]
        return np.stack(channels, axis=-1)
    finally:
        exr.close()


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def png_encoding(meta: dict[str, Any], requested: str) -> tuple[str, float]:
    if requested != "auto":
        return requested, 2.2
    conversion = meta.get("png_conversion") or {}
    if conversion.get("lighting_tonemap") == "reinhard_gamma":
        return "reinhard-gamma", float(conversion.get("lighting_gamma", 2.2))
    return "srgb", 2.2


def read_rgb(path: Path, meta: dict[str, Any], requested_encoding: str) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".exr":
        image = read_exr(path)
        return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), "linear-exr"
    if path.suffix.lower() != ".png":
        raise RuntimeError(f"Unsupported image format: {path}")
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    encoding, gamma = png_encoding(meta, requested_encoding)
    if encoding == "reinhard-gamma":
        mapped = np.clip(image, 0.0, 1.0) ** gamma
        image = mapped / np.maximum(1.0 - mapped, 1.0e-6)
    elif encoding == "srgb":
        image = srgb_to_linear(image)
    elif encoding != "linear":
        raise RuntimeError(f"Unsupported PNG encoding: {encoding}")
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), encoding


def luminance(image: np.ndarray) -> np.ndarray:
    return np.maximum(image[..., :3] @ LUMA_WEIGHTS, 0.0)


def choose_existing(scene_dir: Path, relative: str | Path) -> Path | None:
    path = scene_dir / relative
    if path.is_file():
        return path
    for suffix in (".exr", ".png"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def discover_pairs(scene_dir: Path, meta: dict[str, Any]) -> list[tuple[int, Path, Path]]:
    pairs: dict[int, tuple[Path, Path]] = {}
    debug = meta.get("shadow_loss_debug")
    entries = debug if isinstance(debug, list) else [debug] if isinstance(debug, dict) else []
    for entry in entries:
        try:
            position_id = int(entry["position_id"])
        except (KeyError, TypeError, ValueError):
            continue
        on_path = choose_existing(scene_dir, entry.get("shadow_on", ""))
        off_path = choose_existing(scene_dir, entry.get("shadow_off", ""))
        if on_path and off_path:
            pairs[position_id] = (on_path, off_path)

    debug_dir = scene_dir / "debug_shadow_loss"
    for off_path in sorted(debug_dir.glob("position_*_no_object_shadow.*")):
        match = POSITION_RE.match(off_path.stem)
        if not match or off_path.suffix.lower() not in {".exr", ".png"}:
            continue
        position_id = int(match.group(1))
        on_path = choose_existing(scene_dir, f"samples/position/position_{position_id:03d}{off_path.suffix}")
        if on_path:
            pairs[position_id] = (on_path, off_path)
    return [(position_id, *pairs[position_id]) for position_id in sorted(pairs)]


def geometry_mask_path(scene_dir: Path, position_id: int) -> Path | None:
    mask_dir = scene_dir / f"samples/position/position_{position_id:03d}_masks"
    candidates = sorted(mask_dir.glob("object_shadow_geometry_ray_clean_minarea*.npy"))
    candidates = [path for path in candidates if "_pad" not in path.stem]
    if candidates:
        return candidates[0]
    candidates = sorted(mask_dir.glob("object_shadow_geometry_ray_clean_minarea*.png"))
    candidates = [path for path in candidates if "_pad" not in path.stem]
    return candidates[0] if candidates else None


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(bool)
    return np.asarray(Image.open(path).convert("L")) >= 128


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def measure_pair(
    scene_id: str,
    scene_dir: Path,
    position_id: int,
    on_path: Path,
    off_path: Path,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask_path = geometry_mask_path(scene_dir, position_id)
    if mask_path is None:
        return {"scene_id": scene_id, "position_id": position_id, "status": "missing_geometry_mask"}

    on_rgb, on_encoding = read_rgb(on_path, meta, args.png_encoding)
    off_rgb, off_encoding = read_rgb(off_path, meta, args.png_encoding)
    mask = read_mask(mask_path)
    if on_rgb.shape[:2] != off_rgb.shape[:2] or mask.shape != on_rgb.shape[:2]:
        raise RuntimeError(
            f"Shape mismatch for {scene_id} position_{position_id:03d}: "
            f"on={on_rgb.shape}, off={off_rgb.shape}, mask={mask.shape}"
        )

    geometry_pixels = int(mask.sum())
    on_y = luminance(on_rgb)
    off_y = luminance(off_rgb)
    loss = np.maximum(off_y - on_y, 0.0)
    image_scale = max(safe_percentile(off_y.reshape(-1), 99.0), 1.0e-8)
    relative_loss = loss / np.maximum(off_y, image_scale * 1.0e-6)
    normalized_loss = loss / image_scale

    if geometry_pixels:
        geometry_relative = relative_loss[mask]
        geometry_normalized = normalized_loss[mask]
        strong = (
            (geometry_relative >= float(args.relative_loss_threshold))
            & (geometry_normalized >= float(args.normalized_loss_threshold))
        )
        shadow_coverage = float(strong.mean())
    else:
        geometry_relative = np.empty(0, dtype=np.float32)
        geometry_normalized = np.empty(0, dtype=np.float32)
        shadow_coverage = 0.0

    eligible = geometry_pixels >= int(args.min_geometry_pixels)
    casts_shadow = eligible and shadow_coverage >= float(args.min_shadow_coverage)
    return {
        "scene_id": scene_id,
        "position_id": position_id,
        "status": "measured" if eligible else "geometry_too_small",
        "casts_shadow": bool(casts_shadow),
        "shadow_on": str(on_path),
        "shadow_off": str(off_path),
        "geometry_mask": str(mask_path),
        "image_encoding": on_encoding if on_encoding == off_encoding else [on_encoding, off_encoding],
        "geometry_pixels": geometry_pixels,
        "geometry_ratio": float(mask.mean()),
        "shadow_off_p99_luminance": image_scale,
        "relative_loss_mean": float(geometry_relative.mean()) if geometry_relative.size else 0.0,
        "relative_loss_p95": safe_percentile(geometry_relative, 95.0),
        "relative_loss_p99": safe_percentile(geometry_relative, 99.0),
        "normalized_loss_mean": float(geometry_normalized.mean()) if geometry_normalized.size else 0.0,
        "normalized_loss_p95": safe_percentile(geometry_normalized, 95.0),
        "normalized_loss_p99": safe_percentile(geometry_normalized, 99.0),
        "shadow_coverage": shadow_coverage,
        "thresholds": {
            "relative_loss": float(args.relative_loss_threshold),
            "normalized_loss": float(args.normalized_loss_threshold),
            "min_shadow_coverage": float(args.min_shadow_coverage),
            "min_geometry_pixels": int(args.min_geometry_pixels),
        },
    }


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected = read_scene_filter(args.scene_list)
    scenes = find_scene_dirs(args.input)
    requested_ids = sorted(selected if selected is not None else scenes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    measurements: list[dict[str, Any]] = []
    scene_results: list[dict[str, Any]] = []
    for index, scene_id in enumerate(requested_ids, start=1):
        scene_dir = scenes.get(scene_id)
        if scene_dir is None:
            scene_results.append({"scene_id": scene_id, "status": "inconclusive", "reason": "scene_not_found"})
            continue
        meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
        pairs = discover_pairs(scene_dir, meta)
        pair_results = [
            measure_pair(scene_id, scene_dir, position_id, on_path, off_path, meta, args)
            for position_id, on_path, off_path in pairs
        ]
        measurements.extend(pair_results)
        eligible = [row for row in pair_results if row.get("status") == "measured"]
        if not pairs:
            status, reason = "inconclusive", "no_shadow_on_off_pair"
        elif not eligible:
            status, reason = "inconclusive", "no_eligible_geometry_mask"
        elif any(row.get("casts_shadow", False) for row in eligible):
            status, reason = "pass", "actual_shadow_detected_inside_geometry_mask"
        else:
            status, reason = "fail", "no_actual_shadow_detected_inside_geometry_mask"
        scene_results.append(
            {
                "scene_id": scene_id,
                "status": status,
                "reason": reason,
                "pair_count": len(pair_results),
                "eligible_pair_count": len(eligible),
                "passing_pair_count": sum(bool(row.get("casts_shadow")) for row in eligible),
                "max_shadow_coverage": max((float(row["shadow_coverage"]) for row in eligible), default=0.0),
                "max_relative_loss_p99": max((float(row["relative_loss_p99"]) for row in eligible), default=0.0),
            }
        )
        if index % 25 == 0 or index == len(requested_ids):
            print(f"[{index}/{len(requested_ids)}] evaluated")

    with (output / "pair_measurements.jsonl").open("w", encoding="utf-8") as handle:
        for row in measurements:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    with (output / "scene_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in scene_results:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    by_status = {
        status: [row["scene_id"] for row in scene_results if row["status"] == status]
        for status in ("pass", "fail", "inconclusive")
    }
    for status, scene_ids in by_status.items():
        write_lines(output / f"{status}_scenes.txt", scene_ids)

    summary = {
        "input_roots": args.input,
        "scene_list": args.scene_list,
        "requested_scene_count": len(requested_ids),
        "found_scene_count": sum(scene_id in scenes for scene_id in requested_ids),
        "pair_measurement_count": len(measurements),
        "pass_scene_count": len(by_status["pass"]),
        "fail_scene_count": len(by_status["fail"]),
        "inconclusive_scene_count": len(by_status["inconclusive"]),
        "thresholds": {
            "relative_loss": float(args.relative_loss_threshold),
            "normalized_loss": float(args.normalized_loss_threshold),
            "min_shadow_coverage": float(args.min_shadow_coverage),
            "min_geometry_pixels": int(args.min_geometry_pixels),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
