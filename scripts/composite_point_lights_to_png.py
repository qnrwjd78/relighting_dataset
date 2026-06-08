from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from array import array
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError as exc:  # pragma: no cover - must run inside Blender
    raise SystemExit("scripts/composite_point_lights_to_png.py must be run by Blender Python.") from exc


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Convert point-light EXR components to tone-mapped PNG composites.")
    parser.add_argument("--input", required=True, help="Input dataset root or a single scene directory.")
    parser.add_argument("--output", default=None, help="Output root. Defaults to INPUT_NAME_png next to the input.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.2)
    return parser.parse_args(argv)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_png")


def is_point_light_exr(path: Path) -> bool:
    return path.suffix.lower() == ".exr" and path.parent.name == "point_lights" and path.name.startswith("light_")


def copy_tree_except_point_light_exrs(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists: {output_root}. Use --overwrite to replace/update it.")
        if output_root == input_root:
            raise SystemExit("--output must be different from --input when using --overwrite.")
        shutil.rmtree(output_root)
    for src in input_root.rglob("*"):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if is_point_light_exr(src):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and overwrite:
            dst.unlink()
        if not dst.exists():
            shutil.copy2(src, dst)


def load_image_pixels(path: Path) -> tuple[bpy.types.Image, array]:
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = array("f", [0.0]) * (image.size[0] * image.size[1] * 4)
    image.pixels.foreach_get(pixels)
    return image, pixels


def tone_map_channel(value: float, gamma: float) -> float:
    linear = max(0.0, float(value))
    mapped = linear / (1.0 + linear)
    if gamma > 0.0:
        mapped = mapped ** (1.0 / gamma)
    return max(0.0, min(1.0, mapped))


def composite_to_png(ambient_exr: Path, light_exr: Path, png_path: Path, gamma: float) -> None:
    ambient_image, ambient_pixels = load_image_pixels(ambient_exr)
    light_image, light_pixels = load_image_pixels(light_exr)
    try:
        width, height = ambient_image.size
        if tuple(light_image.size) != (width, height):
            raise RuntimeError(f"Image size mismatch: {ambient_exr} vs {light_exr}")

        output_pixels = array("f", [0.0]) * len(ambient_pixels)
        for i in range(0, len(output_pixels), 4):
            output_pixels[i] = tone_map_channel(ambient_pixels[i] + light_pixels[i], gamma)
            output_pixels[i + 1] = tone_map_channel(ambient_pixels[i + 1] + light_pixels[i + 1], gamma)
            output_pixels[i + 2] = tone_map_channel(ambient_pixels[i + 2] + light_pixels[i + 2], gamma)
            output_pixels[i + 3] = 1.0

        png_path.parent.mkdir(parents=True, exist_ok=True)
        image = bpy.data.images.new(png_path.stem, width=width, height=height, alpha=True, float_buffer=False)
        try:
            image.pixels.foreach_set(output_pixels)
            image.filepath_raw = str(png_path)
            image.file_format = "PNG"
            image.save()
        finally:
            bpy.data.images.remove(image)
    finally:
        bpy.data.images.remove(ambient_image)
        bpy.data.images.remove(light_image)


def find_scene_dirs(input_root: Path) -> list[Path]:
    if (input_root / "spatial" / "ambient.exr").exists():
        return [input_root]
    scenes_root = input_root / "scenes"
    if scenes_root.exists():
        return sorted(path for path in scenes_root.iterdir() if path.is_dir() and (path / "spatial" / "ambient.exr").exists())
    return sorted(path for path in input_root.rglob("scene_*") if path.is_dir() and (path / "spatial" / "ambient.exr").exists())


def update_scene_meta(output_scene_dir: Path, gamma: float) -> None:
    meta_path = output_scene_dir / "meta.json"
    if not meta_path.exists():
        return
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    spatial = meta.get("spatial")
    if not isinstance(spatial, dict):
        return
    spatial["point_light_composite_png"] = {
        "formula": "png = ((ambient + light_i) / (1 + ambient + light_i)) ** (1 / gamma)",
        "ambient": spatial.get("ambient_render", "spatial/ambient.exr"),
        "gamma": gamma,
        "light_intensity": 1.0,
        "light_color": [1.0, 1.0, 1.0],
    }
    for light in spatial.get("point_lights", []):
        render = light.get("render")
        if isinstance(render, str) and render.endswith(".exr"):
            original_render = render
            light["source_component_exr"] = original_render
            light["render"] = original_render[:-4] + ".png"
        copied_from = light.get("copied_from")
        if isinstance(copied_from, str) and copied_from.endswith(".exr"):
            light["copied_from"] = copied_from[:-4] + ".png"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def update_dataset_manifest(output_root: Path) -> None:
    manifest_path = output_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["point_light_components_converted_to_png"] = True
    manifest["point_light_png_formula"] = "png = ((ambient + light_i) / (1 + ambient + light_i)) ** (1 / gamma)"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).resolve()
    if not input_root.exists():
        raise SystemExit(f"Input does not exist: {input_root}")
    output_root = Path(args.output).resolve() if args.output else default_output_path(input_root).resolve()

    print(f"[CompositePNG] Copying {input_root} -> {output_root}", flush=True)
    copy_tree_except_point_light_exrs(input_root, output_root, args.overwrite)

    scene_dirs = find_scene_dirs(input_root)
    if not scene_dirs:
        raise SystemExit(f"No scene directories with spatial/ambient.exr found under {input_root}")

    total = 0
    for scene_dir in scene_dirs:
        rel_scene = scene_dir.relative_to(input_root)
        output_scene_dir = output_root / rel_scene
        ambient = scene_dir / "spatial" / "ambient.exr"
        lights = sorted((scene_dir / "spatial" / "point_lights").glob("light_*.exr"))
        print(f"[CompositePNG] {scene_dir.name}: {len(lights)} lights", flush=True)
        for light in lights:
            rel_light = light.relative_to(scene_dir)
            png_path = (output_scene_dir / rel_light).with_suffix(".png")
            if png_path.exists() and not args.overwrite:
                continue
            composite_to_png(ambient, light, png_path, args.gamma)
            total += 1
        update_scene_meta(output_scene_dir, args.gamma)

    update_dataset_manifest(output_root)
    print(f"[CompositePNG] Wrote {total} PNG composites under {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
