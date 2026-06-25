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
    parser = argparse.ArgumentParser(description="Create contact sheets for shading candidate comparisons.")
    parser.add_argument(
        "--scenes-root",
        default="outputs/objaverse_ratio3p5_cube1p6_full_scene4000_4039_640/scenes",
    )
    parser.add_argument(
        "--selection-manifest",
        default="outputs/relighting_aux_previews_ambient_point_white_depth_normal/manifest.json",
    )
    parser.add_argument("--output-dir", default="outputs/shading_candidate_contact_sheets_round1")
    parser.add_argument("--scenes", nargs="*", type=int, default=None)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "white_shading_optical",
            "matte_direct_a0p6",
            "optical_direct_a0p6",
            "optical_soft_a0p6",
            "neutral_direct_a0p5",
            "neutral_soft_a0p5",
        ],
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.25, 0.1])
    parser.add_argument("--point-color-mode", choices=["raw", "colored"], default="raw")
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--panel-size", type=int, default=360)
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
    image = np.maximum(image * scale, 0.0)
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


def draw_label(canvas: Image.Image, text: str, x0: int, y0: int, panel_size: int, label_height: int, font) -> None:
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = x0 + (panel_size - text_width) // 2
    y = y0 + panel_size + (label_height - text_height) // 2 - 1
    draw.text((x, y), text, fill=(245, 245, 245), font=font)


def candidate_path(scene_dir: Path, candidate: str, point_id: int) -> Path:
    if candidate == "white_shading_optical":
        return scene_dir / "pbr" / "white_shading_optical" / f"point_light_{point_id:03d}.exr"
    return scene_dir / "pbr" / "shading_candidates" / candidate / f"point_light_{point_id:03d}.exr"


def load_selection(path: Path, scenes: set[int] | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items", payload if isinstance(payload, list) else [])
    result = []
    for row in rows:
        scene_id = int(row["scene_id"])
        if scenes is not None and scene_id not in scenes:
            continue
        result.append(row)
    return sorted(result, key=lambda row: (int(row["scene_id"]), int(row.get("variant", 0))))


def main() -> int:
    args = parse_args()
    scenes_root = Path(args.scenes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = set(args.scenes) if args.scenes else None
    rows = load_selection(Path(args.selection_manifest), scenes)
    font = load_font()
    manifest = {
        "source_root": str(scenes_root),
        "selection_manifest": args.selection_manifest,
        "output_root": str(output_dir),
        "point_color_mode": args.point_color_mode,
        "columns": ["point"] + args.candidates,
        "scales": args.scales,
        "items": [],
    }

    for row in rows:
        scene_id = int(row["scene_id"])
        point_id = int(row["point_light_id"])
        variant = int(row.get("variant", 0))
        color = np.array(row.get("sampled_color_rgb", [1.0, 1.0, 1.0]), dtype=np.float32)
        scene_dir = scenes_root / f"scene_{scene_id:06d}"
        point = read_exr(scene_dir / "spatial" / "point_lights" / f"light_{point_id:03d}.exr")
        if args.point_color_mode == "colored":
            point = point * color.reshape(1, 1, 3)
        candidate_images = [(candidate, read_exr(candidate_path(scene_dir, candidate, point_id))) for candidate in args.candidates]

        for scale in args.scales:
            columns = [("point", point)] + candidate_images
            width = args.panel_size * len(columns)
            height = args.panel_size + args.label_height
            canvas = Image.new("RGB", (width, height), (8, 8, 8))
            for index, (label, image) in enumerate(columns):
                preview = resize_panel(tonemap(image, scale, args.gamma), args.panel_size)
                x = index * args.panel_size
                canvas.paste(preview, (x, 0))
                draw_label(canvas, f"{label} x{scale:g}", x, 0, args.panel_size, args.label_height, font)
            scale_token = f"{scale:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            out_path = output_dir / f"scene_{scene_id:06d}_point_{point_id:03d}_preview_{variant:02d}_scale_{scale_token}.png"
            canvas.save(out_path)
            manifest["items"].append(
                {
                    "scene_id": scene_id,
                    "point_light_id": point_id,
                    "variant": variant,
                    "scale": float(scale),
                    "preview": str(out_path),
                }
            )
            print("wrote", out_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
