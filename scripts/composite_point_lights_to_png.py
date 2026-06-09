from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tarfile
import zipfile

import numpy as np
from tqdm import tqdm

try:
    import OpenImageIO as oiio
except ModuleNotFoundError as exc:
    raise SystemExit(
        "OpenImageIO is required. Install with:\n"
        "  pip install OpenImageIO\n"
        "or:\n"
        "  conda install -c conda-forge openimageio"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a minimal dataset from EXR components:\n"
            "- spatial/ambient.exr -> spatial/ambient.png\n"
            "- first N point_lights/light_*.exr -> point_lights/light_*.png\n"
            "- copy meta.json\n"
            "- copy preview files/directories only\n"
        )
    )
    parser.add_argument("--input", required=True, help="Input dataset root or a single scene directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output root. Defaults to INPUT_NAME_minimal_png next to the input.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output root if it exists.")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma after Reinhard tone mapping.")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create archive after export.",
    )
    parser.add_argument(
        "--archive-format",
        choices=["tar.gz", "zip"],
        default="tar.gz",
        help="Archive format. Default: tar.gz.",
    )
    parser.add_argument(
        "--archive-output",
        default=None,
        help="Archive output path. Defaults to OUTPUT.tar.gz or OUTPUT.zip.",
    )
    parser.add_argument(
        "--light-limit",
        type=int,
        default=64,
        help="Number of point lights to export per scene.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. 1 = sequential with per-light progress.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_minimal_png")


def find_scene_dirs(input_root: Path) -> list[Path]:
    if (input_root / "spatial" / "ambient.exr").exists():
        return [input_root]

    scenes_root = input_root / "scenes"
    if scenes_root.exists():
        return sorted(
            p for p in scenes_root.iterdir()
            if p.is_dir() and (p / "spatial" / "ambient.exr").exists()
        )

    return sorted(
        p for p in input_root.rglob("scene_*")
        if p.is_dir() and (p / "spatial" / "ambient.exr").exists()
    )


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def read_image_float(path: Path) -> np.ndarray:
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"Failed to open image: {path}")

    try:
        spec = inp.spec()
        width = spec.width
        height = spec.height
        nchannels = spec.nchannels

        pixels = inp.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError(f"Failed to read image data: {path}")

        arr = np.asarray(pixels, dtype=np.float32).reshape(height, width, nchannels)
        return arr
    finally:
        inp.close()


def tone_map_rgb(rgb: np.ndarray, gamma: float) -> np.ndarray:
    rgb = np.maximum(rgb, 0.0)
    mapped = rgb / (1.0 + rgb)  # Reinhard
    if gamma > 0.0:
        mapped = np.power(mapped, 1.0 / gamma)
    return np.clip(mapped, 0.0, 1.0)


def exr_to_png(exr_path: Path, png_path: Path, gamma: float) -> None:
    image = read_image_float(exr_path)
    if image.shape[2] < 3:
        raise RuntimeError(f"Image must have at least 3 channels: {exr_path}")

    rgb = image[..., :3]
    rgb = tone_map_rgb(rgb, gamma)
    alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=2)
    write_png(png_path, rgba)


def write_png(path: Path, rgba_float01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rgba_uint8 = np.clip(rgba_float01 * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba_uint8 = np.ascontiguousarray(rgba_uint8)

    height, width, channels = rgba_uint8.shape
    if channels != 4:
        raise ValueError(f"Expected RGBA image with 4 channels, got {channels}")

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"Failed to create output image: {path}")

    try:
        spec = oiio.ImageSpec(width, height, channels, oiio.UINT8)
        ok = out.open(str(path), spec)
        if not ok:
            raise RuntimeError(f"Failed to open output for writing: {path}")
        ok = out.write_image(rgba_uint8)
        if not ok:
            raise RuntimeError(f"Failed to write PNG: {path}")
    finally:
        out.close()


def copy_preview_only(scene_dir: Path, output_scene_dir: Path) -> None:
    """
    Copy only preview-related files/folders.
    Supports:
      - scene_dir/preview/...
      - scene_dir/preview.png
      - scene_dir/preview.jpg/jpeg/webp
    """
    preview_dir = scene_dir / "preview"
    if preview_dir.exists() and preview_dir.is_dir():
        dst = output_scene_dir / "preview"
        shutil.copytree(preview_dir, dst, dirs_exist_ok=True)

    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        preview_file = scene_dir / f"preview{ext}"
        if preview_file.exists() and preview_file.is_file():
            dst = output_scene_dir / preview_file.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview_file, dst)


def update_and_copy_meta(
    scene_dir: Path,
    output_scene_dir: Path,
    selected_light_names: set[str],
    gamma: float,
    light_limit: int,
) -> None:
    meta_src = scene_dir / "meta.json"
    if not meta_src.exists():
        return

    with meta_src.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    spatial = meta.get("spatial")
    if isinstance(spatial, dict):
        ambient_render = spatial.get("ambient_render")
        if isinstance(ambient_render, str) and ambient_render.endswith(".exr"):
            spatial["ambient_render"] = ambient_render[:-4] + ".png"
        else:
            spatial["ambient_render"] = "spatial/ambient.png"

        point_lights = spatial.get("point_lights")
        if isinstance(point_lights, list):
            new_point_lights = []
            for light in point_lights:
                if not isinstance(light, dict):
                    continue

                render = light.get("render")
                if not (isinstance(render, str) and render.endswith(".exr")):
                    continue

                light_name = Path(render).name
                if light_name not in selected_light_names:
                    continue

                original_render = render
                light["source_component_exr"] = original_render
                light["render"] = original_render[:-4] + ".png"

                copied_from = light.get("copied_from")
                if isinstance(copied_from, str) and copied_from.endswith(".exr"):
                    copied_from_name = Path(copied_from).name
                    if copied_from_name in selected_light_names:
                        light["copied_from"] = copied_from[:-4] + ".png"

                new_point_lights.append(light)

            spatial["point_lights"] = new_point_lights

        spatial["minimal_png_export"] = {
            "mode": "component_png",
            "gamma": gamma,
            "light_limit": light_limit,
            "ambient": "spatial/ambient.png",
            "point_lights_dir": "spatial/point_lights",
        }

    output_scene_dir.mkdir(parents=True, exist_ok=True)
    meta_dst = output_scene_dir / "meta.json"
    with meta_dst.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def archive_output_dir(
    output_root: Path,
    archive_format: str,
    archive_output: str | None,
) -> Path:
    if archive_output is not None:
        archive_path = Path(archive_output).resolve()
    else:
        if archive_format == "tar.gz":
            archive_path = output_root.with_suffix(output_root.suffix + ".tar.gz")
        elif archive_format == "zip":
            archive_path = output_root.with_suffix(output_root.suffix + ".zip")
        else:
            raise ValueError(f"Unsupported archive format: {archive_format}")

    if archive_path.exists():
        archive_path.unlink()

    print(f"[MinimalPNG] Archiving {output_root} -> {archive_path}", flush=True)

    files = [p for p in output_root.rglob("*") if p.is_file()]

    if archive_format == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as tar:
            for file_path in tqdm(
                files,
                desc="Archiving",
                unit="file",
                dynamic_ncols=True,
            ):
                arcname = file_path.relative_to(output_root.parent)
                tar.add(file_path, arcname=arcname)

    elif archive_format == "zip":
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as zf:
            for file_path in tqdm(
                files,
                desc="Archiving",
                unit="file",
                dynamic_ncols=True,
            ):
                arcname = file_path.relative_to(output_root.parent)
                zf.write(file_path, arcname=arcname)

    else:
        raise ValueError(f"Unsupported archive format: {archive_format}")

    print(f"[MinimalPNG] Archive written: {archive_path}", flush=True)
    return archive_path

def process_scene(
    scene_dir: Path,
    input_root: Path,
    output_root: Path,
    gamma: float,
    light_limit: int,
    show_light_progress: bool,
) -> tuple[str, int, int]:
    rel_scene = scene_dir.relative_to(input_root)
    output_scene_dir = output_root / rel_scene

    ambient_exr = scene_dir / "spatial" / "ambient.exr"
    if not ambient_exr.exists():
        raise RuntimeError(f"Missing ambient EXR: {ambient_exr}")

    point_lights_dir = scene_dir / "spatial" / "point_lights"
    lights = sorted(point_lights_dir.glob("light_*.exr"))[:light_limit]
    selected_light_names = {p.name for p in lights}

    # create minimal folder structure
    (output_scene_dir / "spatial" / "point_lights").mkdir(parents=True, exist_ok=True)

    # 1) copy preview only
    copy_preview_only(scene_dir, output_scene_dir)

    # 2) ambient.exr -> ambient.png
    ambient_png = output_scene_dir / "spatial" / "ambient.png"
    exr_to_png(ambient_exr, ambient_png, gamma)

    # 3) first N point lights -> png
    light_iter = lights
    if show_light_progress:
        light_iter = tqdm(
            lights,
            desc=f"{scene_dir.name}",
            unit="light",
            dynamic_ncols=True,
            leave=False,
        )

    written = 0
    for light_exr in light_iter:
        png_path = (output_scene_dir / "spatial" / "point_lights" / light_exr.name).with_suffix(".png")
        exr_to_png(light_exr, png_path, gamma)
        written += 1

    # 4) meta.json copy + patch
    update_and_copy_meta(
        scene_dir=scene_dir,
        output_scene_dir=output_scene_dir,
        selected_light_names=selected_light_names,
        gamma=gamma,
        light_limit=light_limit,
    )

    return scene_dir.name, len(lights), written


def main() -> int:
    args = parse_args()

    input_root = Path(args.input).resolve()
    if not input_root.exists():
        raise SystemExit(f"Input does not exist: {input_root}")

    output_root = (
        Path(args.output).resolve()
        if args.output
        else default_output_path(input_root).resolve()
    )

    prepare_output_root(output_root, args.overwrite)

    scene_dirs = find_scene_dirs(input_root)
    if not scene_dirs:
        raise SystemExit(f"No scene directories with spatial/ambient.exr found under {input_root}")

    total_available_lights = sum(
        len(list((scene / "spatial" / "point_lights").glob("light_*.exr")))
        for scene in scene_dirs
    )

    print(f"[MinimalPNG] input       : {input_root}", flush=True)
    print(f"[MinimalPNG] output      : {output_root}", flush=True)
    print(f"[MinimalPNG] scenes      : {len(scene_dirs)}", flush=True)
    print(f"[MinimalPNG] total lights: {total_available_lights}", flush=True)
    print(f"[MinimalPNG] light limit : {args.light_limit}", flush=True)
    print(f"[MinimalPNG] workers     : {args.workers}", flush=True)
    print(f"[MinimalPNG] gamma       : {args.gamma}", flush=True)

    total_written = 0

    if args.workers <= 1:
        for scene_dir in tqdm(scene_dirs, desc="Scenes", unit="scene", dynamic_ncols=True):
            scene_name, num_lights, written = process_scene(
                scene_dir=scene_dir,
                input_root=input_root,
                output_root=output_root,
                gamma=args.gamma,
                light_limit=args.light_limit,
                show_light_progress=True,
            )
            total_written += written
            tqdm.write(f"[MinimalPNG] {scene_name}: selected={num_lights}, written={written}")

    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    process_scene,
                    scene_dir,
                    input_root,
                    output_root,
                    args.gamma,
                    args.light_limit,
                    False,
                )
                for scene_dir in scene_dirs
            ]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Scenes",
                unit="scene",
                dynamic_ncols=True,
            ):
                scene_name, num_lights, written = fut.result()
                total_written += written
                tqdm.write(f"[MinimalPNG] {scene_name}: selected={num_lights}, written={written}")

    print(f"[MinimalPNG] Done. Wrote {total_written} point-light PNGs under {output_root}", flush=True)

    if args.archive:
        archive_output_dir(
            output_root=output_root,
            archive_format=args.archive_format,
            archive_output=args.archive_output,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())