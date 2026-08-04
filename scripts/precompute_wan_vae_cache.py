#!/usr/bin/env python3
"""Precompute Wan2.2 VAE latents for rendered relighting PNG datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
WAN_VAE_MODULE_PATH = Path(__file__).resolve().with_name("wan_vae2_2.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Wan2.2 VAE latents from source and sample PNG images."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset containing scenes/scene_*/meta.json.")
    parser.add_argument("--ckpt-dir", required=True, help="Directory containing Wan2.2_VAE.pth.")
    parser.add_argument("--vae-path", default=None, help="Explicit Wan2.2_VAE.pth path.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--image-transform", choices=("rgb", "luminance"), default="luminance")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("fp32", "float32", "fp16", "float16", "bf16", "bfloat16"),
        default="bf16",
    )
    parser.add_argument(
        "--save-dtype",
        choices=("auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--scene-ids", nargs="+", default=[])
    parser.add_argument("--sample-task-filter", default=None, help="Comma-separated tasks, e.g. position,color.")
    parser.add_argument("--no-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return aliases[name]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_wan_vae(vae_path: Path, device: torch.device, dtype: torch.dtype):
    module_path = WAN_VAE_MODULE_PATH
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing bundled Wan2.2 VAE code: {module_path}")
    spec = importlib.util.spec_from_file_location("wan_vae2_2_direct", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Wan VAE module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE(vae_pth=str(vae_path), dtype=dtype, device=device)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_scene_id(value: str) -> str:
    value = str(value).strip()
    suffix = value.removeprefix("scene_")
    if not suffix.isdigit():
        raise ValueError(f"Invalid scene id: {value}")
    return f"scene_{int(suffix):06d}"


def requested_scene_ids(args: argparse.Namespace) -> list[str]:
    result = []
    for raw in [*(args.scene_id or []), *(args.scene_ids or [])]:
        for value in str(raw).split(","):
            if value.strip():
                scene_id = normalize_scene_id(value)
                if scene_id not in result:
                    result.append(scene_id)
    return result


def scene_dirs(dataset_root: Path, selected_ids: list[str]) -> list[Path]:
    scenes_root = dataset_root / "scenes"
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"Missing scenes directory: {scenes_root}")
    scenes = sorted(
        path
        for path in scenes_root.iterdir()
        if path.is_dir() and ((path / "meta.json").is_file() or (path / "samples_manifest.json").is_file())
    )
    if selected_ids:
        by_name = {path.name: path for path in scenes}
        missing = [scene_id for scene_id in selected_ids if scene_id not in by_name]
        if missing:
            raise FileNotFoundError(f"Requested scenes not found: {', '.join(missing)}")
        scenes = [by_name[scene_id] for scene_id in selected_ids]
    if not scenes:
        raise FileNotFoundError(f"No completed scenes found under {scenes_root}")
    return scenes


def comma_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {part.strip() for part in value.split(",") if part.strip()}
    return result or None


def source_from_meta(meta: dict[str, Any]) -> str:
    if meta.get("source_image"):
        return str(meta["source_image"])
    ambient = meta.get("source", {}).get("ambient_only", {})
    for key in ("render_png", "png", "render", "primary"):
        value = ambient.get(key)
        if value and str(value).lower().endswith(".png"):
            return str(value)
    return "source.png"


def load_scene_inputs(scene_dir: Path, task_filter: set[str] | None) -> tuple[dict[str, Any], str, list[dict]]:
    manifest_path = scene_dir / "samples_manifest.json"
    if not manifest_path.is_file():
        manifest_path = scene_dir / "meta.json"
    manifest = read_json(manifest_path)
    samples = []
    for row in manifest.get("samples", []):
        if task_filter is not None and row.get("task") not in task_filter:
            continue
        image = row.get("image")
        if not image:
            light = row.get("light", {})
            image = light.get("render_png") or light.get("png") or light.get("render") or light.get("primary")
        if image and str(image).lower().endswith(".png"):
            normalized = dict(row)
            normalized["image"] = str(image)
            samples.append(normalized)
    if not samples:
        raise ValueError(f"No PNG samples found in {manifest_path}")
    return manifest, source_from_meta(manifest), samples


def image_to_video_tensor(path: Path, resolution: int, image_transform: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if image_transform == "luminance":
        luminance = sum(weight * array[..., index] for index, weight in enumerate(LUMA_WEIGHTS))
        array = np.repeat(luminance[..., None], 3, axis=2)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().mul(2.0).sub(1.0)
    return tensor.unsqueeze(1)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def encode_paths(
    vae,
    paths: list[Path],
    resolution: int,
    image_transform: str,
    batch_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), batch_size):
        videos = [
            image_to_video_tensor(path, resolution, image_transform).to(device=device, dtype=compute_dtype)
            for path in paths[start : start + batch_size]
        ]
        with torch.no_grad(), autocast_context(device, compute_dtype):
            latents = vae.encode(videos)
        if latents is None:
            raise RuntimeError("Wan VAE encode returned None")
        chunks.append(torch.stack([latent.detach().to(device="cpu", dtype=save_dtype) for latent in latents]))
    return torch.cat(chunks, dim=0)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def process_scene(
    vae,
    scene_dir: Path,
    out_path: Path,
    args: argparse.Namespace,
    task_filter: set[str] | None,
    device: torch.device,
    compute_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> None:
    manifest, source_rel, samples = load_scene_inputs(scene_dir, task_filter)
    sample_paths = [scene_dir / row["image"] for row in samples]
    source_path = scene_dir / source_rel
    required = [*sample_paths] if args.no_source else [source_path, *sample_paths]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} PNG files; first: {missing[0]}")

    sample_latents = encode_paths(
        vae,
        sample_paths,
        args.resolution,
        args.image_transform,
        args.batch_size,
        device,
        compute_dtype,
        save_dtype,
    )
    payload = {
        "schema": "wan_vae_scene_latent_cache_v2",
        "scene_id": manifest.get("scene_id", scene_dir.name),
        "resolution": args.resolution,
        "image_transform": args.image_transform,
        "sample_task_filter": sorted(task_filter or []),
        "source_image": source_rel,
        "sample_images": [row["image"] for row in samples],
        "samples": samples,
        "sample_latents": sample_latents,
        "latent_shape": list(sample_latents.shape[1:]),
        "latent_dtype": str(sample_latents.dtype).removeprefix("torch."),
    }
    if not args.no_source:
        payload["source_latent"] = encode_paths(
            vae,
            [source_path],
            args.resolution,
            args.image_transform,
            1,
            device,
            compute_dtype,
            save_dtype,
        )[0]
    atomic_torch_save(payload, out_path)


def write_cache_manifest(
    out_dir: Path,
    dataset_root: Path,
    vae_path: Path,
    scenes: list[Path],
    args: argparse.Namespace,
    task_filter: set[str] | None,
    compute_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> None:
    payload = {
        "schema": "wan_vae_latent_cache_v2",
        "dataset_root": str(dataset_root),
        "vae_path": str(vae_path),
        "scene_count": len(scenes),
        "scene_ids": [scene.name for scene in scenes],
        "resolution": args.resolution,
        "image_transform": args.image_transform,
        "sample_task_filter": sorted(task_filter or []),
        "include_source": not args.no_source,
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "save_dtype": str(save_dtype).removeprefix("torch."),
        "layout": "scenes/{scene_id}.pt",
        "latent_channels": 48,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f"cache_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, out_dir / "cache_manifest.json")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require --num-shards >= 1 and 0 <= --shard-id < --num-shards")
    if args.batch_size < 1 or args.resolution < 16:
        raise ValueError("--batch-size must be positive and --resolution must be >= 16")

    root = Path(__file__).resolve().parents[1]
    dataset_root = resolve_path(root, args.dataset_root)
    ckpt_dir = resolve_path(root, args.ckpt_dir)
    vae_path = resolve_path(root, args.vae_path) if args.vae_path else ckpt_dir / "Wan2.2_VAE.pth"
    out_dir = resolve_path(root, args.out_dir)
    if not vae_path.is_file():
        raise FileNotFoundError(f"Missing Wan VAE checkpoint: {vae_path}")

    compute_dtype = dtype_from_name(args.dtype)
    save_dtype = compute_dtype if args.save_dtype == "auto" else dtype_from_name(args.save_dtype)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    task_filter = comma_set(args.sample_task_filter)
    scenes = scene_dirs(dataset_root, requested_scene_ids(args))
    if args.shard_id == 0:
        write_cache_manifest(out_dir, dataset_root, vae_path, scenes, args, task_filter, compute_dtype, save_dtype)

    shard_scenes = [scene for index, scene in enumerate(scenes) if index % args.num_shards == args.shard_id]
    vae = load_wan_vae(vae_path, device, compute_dtype)
    failed_path = out_dir / f"failed_scenes_shard_{args.shard_id:03d}.jsonl"
    failures = 0
    for scene_dir in tqdm(shard_scenes, desc=f"Wan VAE shard {args.shard_id}/{args.num_shards}", dynamic_ncols=True):
        out_path = out_dir / "scenes" / f"{scene_dir.name}.pt"
        if out_path.is_file() and not args.overwrite:
            continue
        try:
            process_scene(vae, scene_dir, out_path, args, task_filter, device, compute_dtype, save_dtype)
        except Exception as exc:
            if not args.skip_errors:
                raise
            failures += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"scene": scene_dir.name, "error": repr(exc)}) + "\n")
    if failures:
        raise SystemExit(f"Completed with {failures} failures; see {failed_path}")


if __name__ == "__main__":
    main()
