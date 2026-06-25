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
    parser = argparse.ArgumentParser(description="Create direct/soft light transport comparison previews.")
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--soft-root", required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--point-id", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/light_transport_six_panel_previews")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.25])
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--label-height", type=int, default=34)
    return parser.parse_args()


def read_exr(path: Path) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    available = header["channels"].keys()
    planes = []
    for channel in ["R", "G", "B"]:
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


def load_font(size: int = 18) -> ImageFont.ImageFont:
    for path in [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def resize_panel(image: np.ndarray, panel_size: int) -> Image.Image:
    panel = Image.fromarray(image, "RGB")
    if panel.size != (panel_size, panel_size):
        panel = panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    return panel


def draw_label(canvas: Image.Image, text: str, x0: int, panel_size: int, label_height: int, font) -> None:
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = x0 + (panel_size - text_width) // 2
    y = panel_size + (label_height - text_height) // 2 - 1
    draw.text((x, y), text, fill=(245, 245, 245), font=font)


def scale_token(scale: float) -> str:
    return f"{scale:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def image_stats(image: np.ndarray) -> dict[str, float]:
    lum = image.mean(axis=-1)
    return {
        "mean": float(lum.mean()),
        "p50": float(np.percentile(lum, 50)),
        "p90": float(np.percentile(lum, 90)),
        "p99": float(np.percentile(lum, 99)),
        "max": float(lum.max()),
    }


def scene_dir(root: Path, scene_id: int) -> Path:
    return root / "scenes" / f"scene_{scene_id:06d}"


def main() -> int:
    args = parse_args()
    direct_scene = scene_dir(Path(args.direct_root), args.scene_id)
    soft_scene = scene_dir(Path(args.soft_root), args.scene_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    point_name = f"light_{args.point_id:03d}.exr"

    direct_ambient = read_exr(direct_scene / "spatial" / "ambient.exr")
    soft_ambient = read_exr(soft_scene / "spatial" / "ambient.exr")
    direct_point = read_exr(direct_scene / "spatial" / "point_lights" / point_name)
    soft_point = read_exr(soft_scene / "spatial" / "point_lights" / point_name)
    panels = [
        ("ambient", direct_ambient),
        ("ambient_soft", soft_ambient),
        ("ambient+point", direct_ambient + direct_point),
        ("ambient+point_soft", soft_ambient + soft_point),
        ("point", direct_point),
        ("point_soft", soft_point),
    ]

    font = load_font()
    manifest = {
        "direct_root": args.direct_root,
        "soft_root": args.soft_root,
        "scene_id": args.scene_id,
        "point_id": args.point_id,
        "panels": [label for label, _ in panels],
        "stats": {label: image_stats(image) for label, image in panels},
        "previews": [],
        "individual_panels": [],
    }

    for scale in args.scales:
        token = scale_token(scale)
        canvas = Image.new("RGB", (args.panel_size * len(panels), args.panel_size + args.label_height), (8, 8, 8))
        for index, (label, image) in enumerate(panels):
            preview = tonemap(image, scale, args.gamma)
            x = index * args.panel_size
            canvas.paste(resize_panel(preview, args.panel_size), (x, 0))
            draw_label(canvas, f"{label} x{scale:g}", x, args.panel_size, args.label_height, font)
            individual_path = output_dir / f"scene_{args.scene_id:06d}_point_{args.point_id:03d}_{label.replace('+', '_plus_')}_scale_{token}.png"
            resize_panel(preview, args.panel_size).save(individual_path)
            manifest["individual_panels"].append({"label": label, "scale": scale, "path": str(individual_path)})
        out_path = output_dir / f"scene_{args.scene_id:06d}_point_{args.point_id:03d}_light_transport_6panel_scale_{token}.png"
        canvas.save(out_path)
        manifest["previews"].append({"scale": scale, "path": str(out_path)})
        print("wrote", out_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
