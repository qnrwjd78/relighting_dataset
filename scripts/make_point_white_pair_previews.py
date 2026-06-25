from __future__ import annotations

import argparse
import json
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw, ImageFont


FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create point | white_shading_optical diagnostic pairs.")
    parser.add_argument(
        "--scenes-root",
        default="outputs/objaverse_ratio3p5_cube1p6_full_scene4000_4039_640/scenes",
    )
    parser.add_argument(
        "--selection-manifest",
        default="outputs/relighting_aux_previews_ambient_point_white_depth_normal/manifest.json",
    )
    parser.add_argument("--output-dir", default="outputs/point_white_shading_optical_pair_diagnostics")
    parser.add_argument(
        "--point-rel-template",
        default="spatial/point_lights/light_{point_id:03d}.exr",
    )
    parser.add_argument("--left-panel-name", default=None)
    parser.add_argument("--left-label-template", default="point {point_id:03d}")
    parser.add_argument(
        "--white-rel-template",
        default="pbr/white_shading_optical/point_light_{point_id:03d}.exr",
    )
    parser.add_argument("--right-panel-name", default="white_shading_optical")
    parser.add_argument("--right-label-template", default="white shading {point_id:03d}")
    parser.add_argument(
        "--point-color-mode",
        choices=["raw", "colored"],
        default="colored",
        help="Use raw white point-light EXR, or multiply it by sampled_color_rgb.",
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.25, 0.1, 0.05])
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--panel-size", type=int, default=640)
    parser.add_argument("--label-height", type=int, default=39)
    return parser.parse_args()


def read_exr(path: Path, channels: list[str] | None = None) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    available = list(header["channels"].keys())
    channels = channels or ["R", "G", "B"]
    planes = []
    for channel in channels:
        if channel in available:
            plane = np.frombuffer(exr.channel(channel, FLOAT), dtype=np.float32).reshape(height, width)
        else:
            plane = np.zeros((height, width), dtype=np.float32)
        planes.append(plane)
    return np.stack(planes, axis=-1).astype(np.float32, copy=False)


def tonemap(image: np.ndarray, scale: float, gamma: float) -> np.ndarray:
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image * float(scale), 0.0)
    image = image / (1.0 + image)
    image = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def resize_panel(image: np.ndarray, panel_size: int) -> Image.Image:
    panel = Image.fromarray(image, "RGB")
    if panel.size != (panel_size, panel_size):
        panel = panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    return panel


def load_font(size: int = 20) -> ImageFont.ImageFont:
    for path in [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_pair(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str,
    right_label: str,
    panel_size: int,
    label_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (panel_size * 2, panel_size + label_height), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate([(left, left_label), (right, right_label)]):
        canvas.paste(resize_panel(panel, panel_size), (index * panel_size, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = index * panel_size + (panel_size - text_width) // 2
        y = panel_size + (label_height - text_height) // 2 - 1
        draw.text((x, y), label, fill=(245, 245, 245), font=font)
    return canvas


def image_stats(image: np.ndarray) -> dict:
    lum = np.mean(np.maximum(image, 0.0), axis=-1)
    return {
        "mean": float(np.mean(lum)),
        "p50": float(np.percentile(lum, 50)),
        "p90": float(np.percentile(lum, 90)),
        "p99": float(np.percentile(lum, 99)),
        "max": float(np.max(lum)),
    }


def load_selection_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items", payload if isinstance(payload, list) else [])
    return sorted(rows, key=lambda row: (int(row["scene_id"]), int(row.get("variant", 0))))


def scale_name(scale: float) -> str:
    return f"{scale:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> int:
    args = parse_args()
    scenes_root = Path(args.scenes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font()
    rows = load_selection_manifest(Path(args.selection_manifest))
    manifest = {
        "source_root": str(scenes_root),
        "selection_manifest": args.selection_manifest,
        "output_root": str(output_dir),
        "panels": [args.left_panel_name or f"{args.point_color_mode}_point", args.right_panel_name],
        "tonemap": "clip(((image * scale) / (1 + image * scale)) ** (1/gamma), 0, 1)",
        "scales": args.scales,
        "items": [],
    }

    for row in rows:
        scene_id = int(row["scene_id"])
        point_id = int(row["point_light_id"])
        variant = int(row.get("variant", 0))
        color = np.array(row.get("sampled_color_rgb", [1.0, 1.0, 1.0]), dtype=np.float32)
        scene_dir = scenes_root / f"scene_{scene_id:06d}"
        point_rel = args.point_rel_template.format(scene_id=scene_id, point_id=point_id)
        point_path = scene_dir / point_rel
        right_rel = args.white_rel_template.format(scene_id=scene_id, point_id=point_id)
        right_path = scene_dir / right_rel
        point = read_exr(point_path, ["R", "G", "B"])
        if args.point_color_mode == "colored":
            point = point * color.reshape(1, 1, 3)
        right = read_exr(right_path, ["R", "G", "B"])
        item = {
            "scene_id": scene_id,
            "point_light_id": point_id,
            "variant": variant,
            "sampled_color_rgb": [round(float(c), 6) for c in color.tolist()],
            "inputs": {args.left_panel_name or "point": str(point_path), args.right_panel_name: str(right_path)},
            "stats": {args.left_panel_name or "point": image_stats(point), args.right_panel_name: image_stats(right)},
            "previews": [],
        }
        for scale in args.scales:
            point_preview = tonemap(point, scale, args.gamma)
            right_preview = tonemap(right, scale, args.gamma)
            preview = make_pair(
                point_preview,
                right_preview,
                f"{args.left_label_template.format(scene_id=scene_id, point_id=point_id)} x{scale:g}",
                f"{args.right_label_template.format(scene_id=scene_id, point_id=point_id)} x{scale:g}",
                args.panel_size,
                args.label_height,
                font,
            )
            out_path = output_dir / (
                f"scene_{scene_id:06d}_point_{point_id:03d}_preview_{variant:02d}_scale_{scale_name(scale)}.png"
            )
            preview.save(out_path)
            item["previews"].append({"scale": float(scale), "path": str(out_path)})
            print("wrote", out_path)
        manifest["items"].append(item)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
