#!/usr/bin/env python3
"""Run MoGe-3 on images and stream float32 point maps into a tar.zst archive."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED = ROOT / "weights" / "moge-3-vitg" / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["position", "source", "images"], required=True)
    parser.add_argument("--scene-list", default=None, help="Optional text file with one scene directory name per line.")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-limit", type=int, default=0, help="Zero processes all remaining scenes.")
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--resize", type=int, default=480)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--zstd-level", type=int, default=3)
    return parser.parse_args()


def image_paths(
    root: Path,
    mode: str,
    scene_list: str | None,
    scene_offset: int,
    scene_limit: int,
) -> list[Path]:
    if mode == "images":
        paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        paths = paths[max(0, scene_offset):]
        if scene_limit > 0:
            paths = paths[:scene_limit]
        return paths

    scenes_root = root / "scenes"
    if scene_list:
        names = [line.strip() for line in Path(scene_list).read_text().splitlines() if line.strip()]
        scene_dirs = [scenes_root / name for name in names]
    else:
        scene_dirs = sorted(scenes_root.glob("scene_*"))
    scene_dirs = scene_dirs[max(0, scene_offset):]
    if scene_limit > 0:
        scene_dirs = scene_dirs[:scene_limit]
    paths: list[Path] = []
    if mode == "source":
        return [scene_dir / "source.png" for scene_dir in scene_dirs if (scene_dir / "source.png").is_file()]
    for scene_dir in scene_dirs:
        paths.extend(sorted((scene_dir / "samples" / "position").glob("position_*.png")))
    return paths


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(payload))


def main() -> int:
    args = parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    paths = image_paths(root, args.mode, args.scene_list, args.scene_offset, args.scene_limit)
    if not paths:
        raise SystemExit("No input images found")

    from moge.model import import_model_class_by_version

    model = import_model_class_by_version("v3").from_pretrained(args.pretrained).cuda().eval()
    command = ["zstd", f"-{args.zstd_level}", "-T2", "-q", "-o", str(partial)]
    compressor = subprocess.Popen(command, stdin=subprocess.PIPE)
    if compressor.stdin is None:
        raise RuntimeError("Failed to open zstd stdin")

    started = time.time()
    manifest = {
        "schema": "moge3_pointmap_archive_v1",
        "model": args.pretrained,
        "dtype": "float32",
        "resize": args.resize,
        "resolution_level": args.resolution_level,
        "refine_steps": args.refine_steps,
        "input_root": str(root),
        "image_count": len(paths),
    }
    try:
        with tarfile.open(fileobj=compressor.stdin, mode="w|") as archive:
            add_bytes(archive, "pointmaps_manifest.json", json.dumps(manifest, indent=2).encode())
            for index, path in enumerate(paths, 1):
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Failed to read {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                if max(height, width) != args.resize:
                    scale = args.resize / max(height, width)
                    rgb = cv2.resize(
                        rgb,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float().div_(255.0)
                with torch.inference_mode():
                    result = model.infer(
                        tensor,
                        resolution_level=args.resolution_level,
                        refine_steps=args.refine_steps,
                        use_fp16=True,
                    )
                points = result["points"].float().cpu().numpy()
                if points.ndim != 3 or points.shape[-1] != 3:
                    raise RuntimeError(f"Invalid point map for {path}: shape={points.shape}")
                invalid_count = int(np.size(points) - np.count_nonzero(np.isfinite(points)))
                if invalid_count:
                    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
                buffer = io.BytesIO()
                np.save(buffer, points, allow_pickle=False)
                relative = path.relative_to(root).with_suffix(".npy")
                add_bytes(archive, relative.as_posix(), buffer.getvalue())
                if index == 1 or index % 10 == 0 or index == len(paths):
                    elapsed = time.time() - started
                    print(
                        f"[moge3] {index}/{len(paths)} elapsed={elapsed:.1f}s "
                        f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f} "
                        f"invalid_values_zeroed={invalid_count}",
                        flush=True,
                    )
                del tensor, result, points
        compressor.stdin.close()
        return_code = compressor.wait()
        if return_code != 0:
            raise RuntimeError(f"zstd exited with {return_code}")
        os.replace(partial, output)
    except BaseException:
        if compressor.stdin and not compressor.stdin.closed:
            compressor.stdin.close()
        compressor.wait()
        raise
    print(f"[moge3] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
