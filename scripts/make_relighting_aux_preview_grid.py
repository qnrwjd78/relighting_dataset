from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw, ImageFont


FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create relighting auxiliary preview grids.")
    parser.add_argument(
        "--scenes-root",
        default="outputs/objaverse_ratio3p5_cube1p6_full_scene4000_4039_640/scenes",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/relighting_aux_previews_ambient_point_white_depth_normal",
    )
    parser.add_argument("--scenes", nargs="+", type=int, default=None)
    parser.add_argument(
        "--selection-manifest",
        default=None,
        help="Optional manifest with scene_id, point_light_id, and sampled_color_rgb items.",
    )
    parser.add_argument(
        "--shading-rel-template",
        default="pbr/white_shading_optical/point_light_{point_id:03d}.exr",
        help="Scene-relative EXR path for the third panel.",
    )
    parser.add_argument(
        "--shading-label-template",
        default="white shading {point_id:03d}",
        help="Label template for the third panel.",
    )
    parser.add_argument(
        "--shading-panel-name",
        default="white_shading_optical",
        help="Manifest name for the third panel.",
    )
    parser.add_argument(
        "--include-point-panel",
        action="store_true",
        help="Insert a colored point-light-only panel between ambient+point and the shading panel.",
    )
    parser.add_argument("--variants-per-scene", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--panel-size", type=int, default=640)
    parser.add_argument("--label-height", type=int, default=39)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--min-light-mean", type=float, default=0.01)
    parser.add_argument("--min-light-max", type=float, default=0.05)
    return parser.parse_args()


def read_exr(path: Path, channels: list[str] | None = None) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    available = list(header["channels"].keys())

    if channels is None:
        if all(c in available for c in ["R", "G", "B"]):
            channels = ["R", "G", "B"]
        elif all(c in available for c in ["X", "Y", "Z"]):
            channels = ["X", "Y", "Z"]
        else:
            channels = [available[0]]

    planes = []
    for channel in channels:
        if channel in available:
            plane = np.frombuffer(exr.channel(channel, FLOAT), dtype=np.float32).reshape(height, width)
        else:
            plane = np.zeros((height, width), dtype=np.float32)
        planes.append(plane)

    image = np.stack(planes, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image.astype(np.float32, copy=False)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0 + 0.5).astype(np.uint8)


def tonemap_lighting(image: np.ndarray, gamma: float) -> np.ndarray:
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0.0)
    image = image / (1.0 + image)
    image = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
    return to_uint8(image)


def make_depth_preview(depth_rgb: np.ndarray) -> np.ndarray:
    depth = depth_rgb[..., 0].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    values = depth[valid]
    near = float(np.percentile(values, 1.0))
    far = float(np.percentile(values, 99.0))
    if far <= near + 1e-6:
        near = float(values.min())
        far = float(values.max())

    if far <= near + 1e-6:
        normalized = np.zeros_like(depth)
    else:
        normalized = (depth - near) / (far - near)

    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = 1.0 - normalized
    normalized[~valid] = 0.0
    return to_uint8(np.repeat(normalized[..., None], 3, axis=-1))


def make_normal_preview(normal: np.ndarray) -> np.ndarray:
    normal = np.nan_to_num(normal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmin(normal)) < -0.05 or float(np.nanmax(normal)) > 1.05:
        normal = normal * 0.5 + 0.5
    return to_uint8(normal)


def resize_panel(image: np.ndarray, panel_size: int) -> Image.Image:
    panel = Image.fromarray(image, "RGB")
    if panel.size != (panel_size, panel_size):
        panel = panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    return panel


def load_font(size: int = 20) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_labeled_row(
    panels: list[np.ndarray],
    labels: list[str],
    panel_size: int,
    label_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (panel_size * len(panels), panel_size + label_height), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(resize_panel(panel, panel_size), (index * panel_size, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = index * panel_size + (panel_size - text_width) // 2
        y = panel_size + (label_height - text_height) // 2 - 1
        draw.text((x, y), label, fill=(245, 245, 245), font=font)
    return canvas


def light_stats(scene_dir: Path, point_id: int) -> tuple[float, float]:
    point = read_exr(scene_dir / "spatial" / "point_lights" / f"light_{point_id:03d}.exr", ["R", "G", "B"])
    return float(np.mean(point)), float(np.max(point))


def available_point_lights(
    scene_dir: Path,
    min_light_mean: float,
    min_light_max: float,
) -> list[tuple[int, float, float]]:
    candidates = []
    fallback = []
    for point_id in range(64):
        point_path = scene_dir / "spatial" / "point_lights" / f"light_{point_id:03d}.exr"
        white_path = scene_dir / "pbr" / "white_shading_optical" / f"point_light_{point_id:03d}.exr"
        if not point_path.exists() or not white_path.exists():
            continue
        mean, maximum = light_stats(scene_dir, point_id)
        item = (point_id, mean, maximum)
        fallback.append(item)
        if mean > min_light_mean and maximum > min_light_max:
            candidates.append(item)
    return candidates if len(candidates) >= 2 else fallback


def sample_color(rng: random.Random) -> np.ndarray:
    color = [rng.uniform(0.18, 1.0) for _ in range(3)]
    color[rng.randrange(3)] = rng.uniform(0.75, 1.0)
    return np.array(color, dtype=np.float32)


def load_selection_manifest(path: Path) -> dict[int, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items", payload if isinstance(payload, list) else [])
    selections: dict[int, list[dict]] = {}
    for row in rows:
        scene_id = int(row["scene_id"])
        selections.setdefault(scene_id, []).append(
            {
                "scene_id": scene_id,
                "variant": int(row.get("variant", len(selections.get(scene_id, [])))),
                "point_light_id": int(row["point_light_id"]),
                "sampled_color_rgb": row.get("sampled_color_rgb"),
            }
        )
    for rows_for_scene in selections.values():
        rows_for_scene.sort(key=lambda item: item["variant"])
    return selections


def main() -> int:
    args = parse_args()
    scenes_root = Path(args.scenes_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    font = load_font()

    panel_names = ["ambient", "ambient_plus_colored_point", args.shading_panel_name, "depth", "normal"]
    if args.include_point_panel:
        panel_names.insert(2, "colored_point")

    manifest = {
        "source_root": str(scenes_root),
        "output_root": str(output_dir),
        "seed": args.seed,
        "panels": panel_names,
        "tonemap": {
            "lighting": "clip((x / (1 + x)) ** (1/gamma), 0, 1)",
            "ambient_plus_point": "tonemap(ambient_exr + point_light_exr * sampled_rgb)",
            "point": "tonemap(point_light_exr * sampled_rgb)",
            "depth": "1 - percentile_normalize(depth, p1, p99)",
            "normal": "normal * 0.5 + 0.5 when stored range is [-1, 1]",
        },
        "items": [],
    }

    selections = load_selection_manifest(Path(args.selection_manifest)) if args.selection_manifest else None
    scene_ids = sorted(selections) if selections is not None else args.scenes
    if not scene_ids:
        raise SystemExit("Provide --scenes or --selection-manifest.")

    for scene_id in scene_ids:
        scene_dir = scenes_root / f"scene_{scene_id:06d}"
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)

        if selections is None:
            choices = available_point_lights(scene_dir, args.min_light_mean, args.min_light_max)
            if len(choices) < args.variants_per_scene:
                raise RuntimeError(f"Not enough point lights for scene {scene_id}: {len(choices)}")
            selected = [
                {
                    "variant": variant,
                    "point_light_id": point_id,
                    "sampled_color_rgb": None,
                    "point_light_mean": mean,
                    "point_light_max": maximum,
                }
                for variant, (point_id, mean, maximum) in enumerate(rng.sample(choices, args.variants_per_scene))
            ]
        else:
            selected = []
            for item in selections.get(scene_id, []):
                point_id = int(item["point_light_id"])
                mean, maximum = light_stats(scene_dir, point_id)
                selected.append(
                    {
                        "variant": int(item.get("variant", len(selected))),
                        "point_light_id": point_id,
                        "sampled_color_rgb": item.get("sampled_color_rgb"),
                        "point_light_mean": mean,
                        "point_light_max": maximum,
                    }
                )

        ambient = read_exr(scene_dir / "spatial" / "ambient.exr", ["R", "G", "B"])
        depth = read_exr(scene_dir / "pbr" / "depth.exr", ["R", "G", "B"])
        normal = read_exr(scene_dir / "pbr" / "normal.exr", ["X", "Y", "Z"])

        for item in selected:
            point_id = int(item["point_light_id"])
            variant_index = int(item["variant"])
            mean = float(item["point_light_mean"])
            maximum = float(item["point_light_max"])
            color_value = item.get("sampled_color_rgb")
            color = np.array(color_value, dtype=np.float32) if color_value is not None else sample_color(rng)
            point = read_exr(scene_dir / "spatial" / "point_lights" / f"light_{point_id:03d}.exr", ["R", "G", "B"])
            shading_rel = args.shading_rel_template.format(scene_id=scene_id, point_id=point_id)
            shading = read_exr(scene_dir / shading_rel, ["R", "G", "B"])
            colored_point = point * color.reshape(1, 1, 3)
            combined = ambient + colored_point

            panels = [
                tonemap_lighting(ambient, args.gamma),
                tonemap_lighting(combined, args.gamma),
                tonemap_lighting(shading, args.gamma),
                make_depth_preview(depth),
                make_normal_preview(normal),
            ]
            labels = [
                "ambient",
                f"ambient + point {point_id:03d}",
                args.shading_label_template.format(scene_id=scene_id, point_id=point_id),
                "depth",
                "normal",
            ]
            if args.include_point_panel:
                panels.insert(2, tonemap_lighting(colored_point, args.gamma))
                labels.insert(2, f"point {point_id:03d}")
            preview = make_labeled_row(panels, labels, args.panel_size, args.label_height, font)
            out_path = output_dir / f"scene_{scene_id:06d}_point_{point_id:03d}_preview_{variant_index:02d}.png"
            preview.save(out_path)

            item = {
                "scene_id": scene_id,
                "variant": variant_index,
                "point_light_id": point_id,
                "sampled_color_rgb": [round(float(c), 6) for c in color.tolist()],
                "point_light_mean": mean,
                "point_light_max": maximum,
                "preview": str(out_path),
                "inputs": {
                    "ambient": str(scene_dir / "spatial" / "ambient.exr"),
                    "point_light": str(scene_dir / "spatial" / "point_lights" / f"light_{point_id:03d}.exr"),
                    args.shading_panel_name: str(scene_dir / shading_rel),
                    "depth": str(scene_dir / "pbr" / "depth.exr"),
                    "normal": str(scene_dir / "pbr" / "normal.exr"),
                },
            }
            manifest["items"].append(item)
            print(
                "wrote",
                out_path,
                "point=",
                f"{point_id:03d}",
                "color=",
                item["sampled_color_rgb"],
                "mean=",
                f"{mean:.4f}",
                "max=",
                f"{maximum:.4f}",
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
