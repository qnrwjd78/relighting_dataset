from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenlight_dataset.exr_io import read_exr  # noqa: E402
from tokenlight_dataset.tonemap import reinhard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create global/per-light TokenLight control preview sheets.")
    parser.add_argument("--root", default="outputs/test_one_scene")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--out-dir", default="outputs/test_one_scene_loader_preview")
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--label-height", type=int, default=56)
    parser.add_argument("--font-size", type=int, default=24)
    parser.add_argument("--ambient-scale", type=float, default=0.65)
    parser.add_argument("--diffuse-intensity", type=float, default=1.0)
    parser.add_argument("--light-index", type=int, default=None)
    parser.add_argument("--color", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--ambient-values", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--intensity-values", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument(
        "--color-values",
        nargs="+",
        default=["1,1,1", "1,0.86,0.68", "0.68,0.82,1", "1,0.34,0.24", "0.35,1,0.55", "0.25,0.5,1"],
        help="Six RGB colors for the color sequence, formatted as r,g,b values in 0..1.",
    )
    parser.add_argument("--write-rows", action="store_true")
    parser.add_argument("--write-panels", action="store_true")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_scene(root: Path, scene_id: str | None) -> tuple[Path, dict]:
    scenes_root = root / "scenes"
    if scene_id:
        scene_dir = scenes_root / scene_id
    else:
        candidates = sorted(path for path in scenes_root.iterdir() if path.is_dir() and (path / "meta.json").exists())
        if not candidates:
            raise FileNotFoundError(f"No scene meta.json files under {scenes_root}")
        scene_dir = candidates[0]
    with (scene_dir / "meta.json").open("r", encoding="utf-8") as f:
        return scene_dir, json.load(f)


def component_path(scene_dir: Path, entry: dict | str) -> Path:
    if isinstance(entry, dict):
        value = entry.get("render_exr") or entry.get("exr") or entry.get("render")
    else:
        value = entry
    path = scene_dir / str(value)
    if path.suffix.lower() != ".exr":
        path = path.with_suffix(".exr")
    return path


def read_component(scene_dir: Path, entry: dict | str) -> np.ndarray:
    return read_exr(component_path(scene_dir, entry))


def spatial_component(scene_dir: Path, spatial: dict, light: dict, ambient: np.ndarray, entry: dict | None = None) -> np.ndarray:
    component = read_component(scene_dir, entry or light)
    if spatial.get("point_light_output_semantics") == "ambient_plus_point_light_target":
        component = component - ambient
    source_color = np.asarray(light.get("component_color", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
    component = component / np.maximum(source_color, 1e-4)
    return np.maximum(component, 0.0)


def valid_spatial_lights(scene_dir: Path, spatial: dict) -> list[dict]:
    lights = []
    for light in spatial["point_lights"]:
        if not light.get("valid", True) or not light.get("render"):
            continue
        if component_path(scene_dir, light).exists():
            lights.append(light)
    if not lights:
        raise RuntimeError("No valid spatial light component EXRs found.")
    return lights


def select_light(scene_dir: Path, spatial: dict, light_index: int | None) -> dict:
    lights = valid_spatial_lights(scene_dir, spatial)
    if light_index is None:
        return lights[0]
    by_id = {int(light["id"]): light for light in lights}
    if light_index not in by_id:
        raise ValueError(f"light-index {light_index} is not a valid rendered light id.")
    return by_id[light_index]


def to_image(x: np.ndarray, panel_size: int) -> Image.Image:
    img = np.nan_to_num(reinhard(np.maximum(x, 0.0)), nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    arr = (img * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    return pil.resize((panel_size, panel_size), Image.Resampling.LANCZOS)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=max(int(size), 1))
        except OSError:
            continue
    return ImageFont.load_default()


def fitted_font(label: str, width: int, font_size: int) -> ImageFont.ImageFont:
    size = max(int(font_size), 1)
    while size > 8:
        font = load_font(size)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)
        if bbox[2] - bbox[0] <= width - 8:
            return font
        size -= 1
    return load_font(size)


def label_panel(img: Image.Image, label: str, label_height: int, font_size: int) -> Image.Image:
    font = fitted_font(label, img.width, font_size)
    out = Image.new("RGB", (img.width, img.height + label_height), (18, 18, 18))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    bbox = draw.textbbox((0, 0), label, font=font)
    x = max((img.width - (bbox[2] - bbox[0])) // 2, 4)
    y = img.height + max((label_height - (bbox[3] - bbox[1])) // 2, 2)
    draw.text((x, y), label, fill=(240, 240, 240), font=font)
    return out


def make_row(images: list[Image.Image], labels: list[str], label_height: int, font_size: int) -> Image.Image:
    panels = [label_panel(img, label, label_height, font_size) for img, label in zip(images, labels)]
    row = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), (0, 0, 0))
    x = 0
    for panel in panels:
        row.paste(panel, (x, 0))
        x += panel.width
    return row


def make_ambient_row(scene_dir: Path, meta: dict, args: argparse.Namespace) -> tuple[Image.Image, list[Image.Image], list[str]]:
    spatial = meta["spatial"]
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    values = list(args.ambient_values[:5])
    if len(values) < 1:
        raise ValueError("--ambient-values needs at least one target value.")
    images = [to_image(ambient, args.panel_size)]
    labels = ["gt"]
    for value in values:
        images.append(to_image(float(value) * ambient, args.panel_size))
        labels.append(f"a={float(value):.2f}")
    return make_row(images, labels, args.label_height, args.font_size), images, labels


def global_diffuse_complete_targets(diffuse: dict) -> bool:
    return bool(diffuse.get("complete_target_variants", True))


def compose_global_diffuse_variant(scene_dir: Path, diffuse: dict, row: dict, args: argparse.Namespace) -> np.ndarray:
    if global_diffuse_complete_targets(diffuse):
        return read_component(scene_dir, row)
    ambient_entry = diffuse.get("ambient_output", diffuse.get("ambient_render"))
    if ambient_entry is None:
        raise RuntimeError("Component global_diffuse metadata needs ambient_output or ambient_render.")
    ambient = read_component(scene_dir, ambient_entry)
    component = read_component(scene_dir, row)
    color = np.asarray(diffuse.get("light", {}).get("color", args.color), dtype=np.float32).reshape(1, 1, 3)
    ambient_scale = float(diffuse.get("preview_ambient_scale", 1.0))
    intensity = float(diffuse.get("preview_intensity", args.diffuse_intensity))
    return ambient_scale * ambient + intensity * component * color


def make_global_diffuse_row(scene_dir: Path, meta: dict, args: argparse.Namespace) -> tuple[Image.Image, list[Image.Image], list[str]]:
    spatial = meta["spatial"]
    diffuse = global_diffuse_meta(meta)
    if diffuse and diffuse.get("variants"):
        variants = sorted(diffuse["variants"], key=lambda row: float(row.get("dg", row.get("normalized_diffuse", 0.0))))
        if global_diffuse_complete_targets(diffuse):
            base = read_component(scene_dir, diffuse.get("base_output", diffuse.get("base_render", spatial["ambient_render"])))
        else:
            base_index = int(diffuse.get("base_variant_id", 0))
            base = compose_global_diffuse_variant(scene_dir, diffuse, variants[min(base_index, len(variants) - 1)], args)
        targets = [row for row in variants if float(row.get("dg", row.get("normalized_diffuse", 0.0))) > 1e-6]
        if len(targets) < 5:
            targets = variants[1:6] if len(variants) >= 6 else variants[:5]
        targets = targets[:5]
        if len(targets) < 1:
            raise RuntimeError("Need at least one global_diffuse target variant.")
        images = [to_image(base, args.panel_size)]
        labels = ["gt"]
        for row in targets:
            value = float(row.get("dg", row.get("normalized_diffuse", 0.0)))
            images.append(to_image(compose_global_diffuse_variant(scene_dir, diffuse, row, args), args.panel_size))
            labels.append(f"dg={value:.2f}")
        return make_row(images, labels, args.label_height, args.font_size), images, labels

    if "diffuse" not in meta:
        raise RuntimeError("No global_diffuse metadata and no legacy diffuse fallback found.")
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    spreads = sorted(meta["diffuse"]["spreads"], key=lambda row: float(row.get("normalized_spread", row.get("spread_degrees", 0.0))))
    targets = spreads[:5]
    values = normalized_values(targets)
    color = np.asarray(args.color, dtype=np.float32).reshape(1, 1, 3)
    images = [to_image(ambient, args.panel_size)]
    labels = ["gt"]
    for spread, value in zip(targets, values):
        component = read_component(scene_dir, spread)
        linear = args.ambient_scale * ambient + args.diffuse_intensity * component * color
        images.append(to_image(linear, args.panel_size))
        labels.append(f"dg~{value:.2f}")
    return make_row(images, labels, args.label_height, args.font_size), images, labels


def make_intensity_row(scene_dir: Path, meta: dict, args: argparse.Namespace) -> tuple[Image.Image, list[Image.Image], list[str]]:
    spatial = meta["spatial"]
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    light = select_light(scene_dir, spatial, args.light_index)
    component = spatial_component(scene_dir, spatial, light, ambient)
    color = np.asarray(args.color, dtype=np.float32).reshape(1, 1, 3)
    values = list(args.intensity_values[:5])
    if len(values) < 1:
        raise ValueError("--intensity-values needs at least one target value.")

    images = [to_image(args.ambient_scale * ambient + component * color, args.panel_size)]
    labels = ["gt"]
    for value in values:
        linear = args.ambient_scale * ambient + float(value) * component * color
        images.append(to_image(linear, args.panel_size))
        labels.append(f"intensity={float(value):.2f}")
    return make_row(images, labels, args.label_height, args.font_size), images, labels


def make_color_row(scene_dir: Path, meta: dict, args: argparse.Namespace) -> tuple[Image.Image, list[Image.Image], list[str]]:
    spatial = meta["spatial"]
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    light = select_light(scene_dir, spatial, args.light_index)
    component = spatial_component(scene_dir, spatial, light, ambient)
    colors = parse_color_values(args.color_values)[:6]
    if len(colors) < 2:
        raise ValueError("--color-values needs at least two RGB colors.")

    images = []
    labels = []
    for i, color in enumerate(colors):
        linear = args.ambient_scale * ambient + component * color.reshape(1, 1, 3)
        images.append(to_image(linear, args.panel_size))
        labels.append("gt" if i == 0 else f"color={color_label(color)}")
    return make_row(images, labels, args.label_height, args.font_size), images, labels


def make_per_light_diffuse_row(scene_dir: Path, meta: dict, args: argparse.Namespace) -> tuple[Image.Image, list[Image.Image], list[str]]:
    spatial = meta["spatial"]
    ambient = read_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
    light = select_light(scene_dir, spatial, args.light_index)
    variants = [row for row in light.get("diffuse_variants", []) if row.get("render")]
    variants = sorted(variants, key=lambda row: float(row.get("d", row.get("normalized_diffuse", 0.0))))
    if len(variants) < 2:
        raise RuntimeError(
            "No per-light diffuse variants found. Re-render with --per-light-diffuse or PER_LIGHT_DIFFUSE=1."
        )
    selected = variants[:6]
    color = np.asarray(args.color, dtype=np.float32).reshape(1, 1, 3)
    images = []
    labels = []
    for i, variant in enumerate(selected):
        component = spatial_component(scene_dir, spatial, light, ambient, variant)
        linear = args.ambient_scale * ambient + args.diffuse_intensity * component * color
        images.append(to_image(linear, args.panel_size))
        value = float(variant.get("d", variant.get("normalized_diffuse", 0.0)))
        labels.append("gt" if i == 0 else f"d={value:.2f}")
    return make_row(images, labels, args.label_height, args.font_size), images, labels


def normalized_values(rows: list[dict]) -> list[float]:
    raw = []
    for row in rows:
        if row.get("normalized_spread") is not None:
            raw.append(float(row["normalized_spread"]))
        elif row.get("spread_degrees") is not None:
            raw.append(float(row["spread_degrees"]))
        else:
            raw.append(float(len(raw)))
    if not raw:
        return []
    lo, hi = min(raw), max(raw)
    if hi <= lo + 1e-8:
        return [0.0 for _ in raw]
    return [(value - lo) / (hi - lo) for value in raw]


def parse_color_values(values: list[str]) -> list[np.ndarray]:
    colors = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Color must be formatted as r,g,b: {value!r}")
        color = np.asarray([float(part) for part in parts], dtype=np.float32)
        colors.append(np.clip(color, 0.0, 1.0))
    return colors


def color_label(color: np.ndarray) -> str:
    return ",".join(f"{float(channel):.2f}" for channel in color)


def global_diffuse_meta(meta: dict) -> dict | None:
    return meta.get("global_diffuse") or meta.get("spatial", {}).get("global_diffuse")


def stack_rows(rows: list[Image.Image], gap: int = 16) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (width, height), (28, 28, 28))
    y = 0
    for row in rows:
        out.paste(row, ((width - row.width) // 2, y))
        y += row.height + gap
    return out


def save_individual_panels(out_dir: Path, prefix: str, images: list[Image.Image], labels: list[str]) -> None:
    safe_labels = [label.replace("=", "_").replace(".", "_").replace(",", "_").replace("~", "_") for label in labels]
    for i, (img, label) in enumerate(zip(images, safe_labels)):
        img.save(out_dir / f"{prefix}_{i:02d}_{label}.png")


def main() -> int:
    args = parse_args()
    root = resolve_repo_path(args.root)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_dir, meta = load_scene(root, args.scene_id)

    ambient_row, ambient_images, ambient_labels = make_ambient_row(scene_dir, meta, args)
    global_diffuse_row, global_diffuse_images, global_diffuse_labels = make_global_diffuse_row(scene_dir, meta, args)
    color_row, color_images, color_labels = make_color_row(scene_dir, meta, args)
    intensity_row, intensity_images, intensity_labels = make_intensity_row(scene_dir, meta, args)
    per_light_diffuse_row, per_light_diffuse_images, per_light_diffuse_labels = make_per_light_diffuse_row(scene_dir, meta, args)

    global_controls = stack_rows([ambient_row, global_diffuse_row])
    per_light_controls = stack_rows([color_row, intensity_row, per_light_diffuse_row])
    global_controls.save(out_dir / "global_controls_sequence.png")
    per_light_controls.save(out_dir / "per_light_controls_sequence.png")

    if args.write_rows:
        ambient_row.save(out_dir / "ambient_sequence.png")
        global_diffuse_row.save(out_dir / "global_diffuse_sequence.png")
        color_row.save(out_dir / "color_sequence.png")
        intensity_row.save(out_dir / "intensity_sequence.png")
        per_light_diffuse_row.save(out_dir / "per_light_diffuse_sequence.png")
    if args.write_panels:
        save_individual_panels(out_dir, "ambient", ambient_images, ambient_labels)
        save_individual_panels(out_dir, "global_diffuse", global_diffuse_images, global_diffuse_labels)
        save_individual_panels(out_dir, "color", color_images, color_labels)
        save_individual_panels(out_dir, "intensity", intensity_images, intensity_labels)
        save_individual_panels(out_dir, "per_light_diffuse", per_light_diffuse_images, per_light_diffuse_labels)

    print(f"Wrote {out_dir / 'global_controls_sequence.png'}")
    print(f"Wrote {out_dir / 'per_light_controls_sequence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
