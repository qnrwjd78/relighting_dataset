#!/usr/bin/env python3
"""Extract MoGe-3 source-image point maps into a scene-ID directory tree."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED = ROOT / "weights" / "moge-3-vitg" / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list of {scene_id, image} objects")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--resize", type=int, default=480)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = json.loads(Path(args.manifest).read_text())
    entries = entries[max(args.offset, 0):]
    if args.limit > 0:
        entries = entries[:args.limit]
    if not entries:
        raise SystemExit("No inputs selected")

    output_root = Path(args.output_root)
    from moge.model import import_model_class_by_version
    model = import_model_class_by_version("v3").from_pretrained(args.pretrained).cuda().eval()
    started = time.time()

    for index, entry in enumerate(entries, 1):
        scene_id = entry["scene_id"]
        image_path = Path(entry["image"])
        output = output_root / scene_id / "source.npy"
        if output.is_file():
            continue
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if max(height, width) != args.resize:
            scale = args.resize / max(height, width)
            rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
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
            raise RuntimeError(f"Invalid point map for {image_path}: {points.shape}")
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".npy.partial")
        with partial.open("wb") as handle:
            np.save(handle, points, allow_pickle=False)
        partial.replace(output)
        if index == 1 or index % 10 == 0 or index == len(entries):
            print(f"[source] {index}/{len(entries)} scene={scene_id} elapsed={time.time()-started:.1f}s", flush=True)
        del tensor, result, points
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
