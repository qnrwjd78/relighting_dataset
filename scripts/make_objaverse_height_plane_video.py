#!/usr/bin/env python3
"""Create a video from one scene's fixed-grid lights at one canonical height."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "outputs/objaverse_fixed_7x7x5_power06_png_0000_0999"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make an XY-plane lighting video for one scene and one canonical Z height."
    )
    parser.add_argument("--scene", required=True, help="Scene id, e.g. 1, 000001, or scene_000001.")
    parser.add_argument("--height", required=True, type=float, help="Canonical Z height, e.g. 0.7.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET, help="Converted PNG dataset root.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", default=None, help="Output MP4 path; defaults below DATASET_ROOT/videos/.")
    parser.add_argument(
        "--height-tolerance",
        type=float,
        default=1.0e-4,
        help="Maximum difference allowed between --height and an available grid height.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.height_tolerance < 0:
        parser.error("--height-tolerance must be nonnegative")
    return args


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def scene_token(value: str) -> str:
    text = str(value).strip()
    if text.startswith("scene_"):
        number = int(text.split("_", 1)[1])
    else:
        number = int(text)
    return f"scene_{number:06d}"


def available_heights(samples: list[dict]) -> list[float]:
    return sorted(
        {
            float(row["light"]["canonical_position"][2])
            for row in samples
            if row.get("task") == "position" and row.get("light", {}).get("canonical_position")
        }
    )


def selected_plane(samples: list[dict], height: float, tolerance: float) -> tuple[float, list[dict]]:
    heights = available_heights(samples)
    if not heights:
        raise RuntimeError("No position samples with canonical positions were found in meta.json")
    selected_height = min(heights, key=lambda value: abs(value - height))
    if abs(selected_height - height) > tolerance:
        choices = ", ".join(f"{value:g}" for value in heights)
        raise RuntimeError(f"Height {height:g} is unavailable; available heights: {choices}")

    rows = [
        row
        for row in samples
        if row.get("task") == "position"
        and abs(float(row["light"]["canonical_position"][2]) - selected_height) <= tolerance
    ]

    # X increases row-by-row; Y alternates direction for a continuous scan.
    x_values = sorted({int(row["light"]["grid_cell"][0]) for row in rows})
    x_order = {value: index for index, value in enumerate(x_values)}
    rows.sort(
        key=lambda row: (
            x_order[int(row["light"]["grid_cell"][0])],
            int(row["light"]["grid_cell"][1])
            if x_order[int(row["light"]["grid_cell"][0])] % 2 == 0
            else -int(row["light"]["grid_cell"][1]),
        )
    )
    return selected_height, rows


def ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("ffmpeg was not found and imageio-ffmpeg is unavailable") from exc


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    scene_id = scene_token(args.scene)
    scene_dir = dataset_root / "scenes" / scene_id
    meta_path = scene_dir / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"Scene meta.json does not exist: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    selected_height, rows = selected_plane(meta.get("samples", []), args.height, args.height_tolerance)
    if not rows:
        raise SystemExit(f"No position samples were found for {scene_id} at height {selected_height:g}")

    output = (
        resolve_path(args.output)
        if args.output
        else dataset_root / "videos" / f"{scene_id}_height_{selected_height:g}_{args.fps:g}fps.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="objaverse_height_video_") as temp_value:
        temp_dir = Path(temp_value)
        for index, row in enumerate(rows):
            image_value = row.get("image") or row.get("light", {}).get("render")
            if not image_value:
                raise RuntimeError(f"Sample has no image reference: {row.get('name')}")
            image_path = scene_dir / Path(str(image_value)).with_suffix(".png")
            if not image_path.is_file():
                raise FileNotFoundError(f"Position PNG does not exist: {image_path}")
            frame_path = temp_dir / f"frame_{index:04d}.png"
            frame_path.symlink_to(os.path.relpath(image_path, temp_dir))

        command = [
            ffmpeg_executable(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(args.fps),
            "-i",
            str(temp_dir / "frame_%04d.png"),
            "-frames:v",
            str(len(rows)),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)

    print(f"scene={scene_id}")
    print(f"height={selected_height:g}")
    print(f"frames={len(rows)}")
    print(f"fps={args.fps:g}")
    print(f"duration_seconds={len(rows) / args.fps:g}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
