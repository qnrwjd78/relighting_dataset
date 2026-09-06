#!/usr/bin/env python3
"""Unified cache, point-map, and asset packaging workflows for objaverse_245."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from pipeline_utils import archive_directory, archive_paths, run_gpu_shards, upload_dataset_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPO_ID = "JaehoChae/blender_relight"
REPO_DIR = "objaverse_245"
MOGE_PYTHON = ROOT / "miniconda3" / "envs" / "moge3" / "bin" / "python"
MOGE_EXTRACTOR = ROOT / "scripts" / "extract_moge3_source_pointmaps.py"
VAE_SCRIPT = ROOT / "scripts" / "precompute_wan_vae_cache.py"
VAE_CHECKPOINT = ROOT / "weights" / "Wan2.2-TI2V-5B"

CACHE_SPLITS = {
    "train": (
        OUTPUTS / "objaverse_fixed_7x7x5_power06_png_0000_0999",
        OUTPUTS / "objaverse_245_train_cache_0000_0999",
        "objaverse_245_train_cache_0000_0999.tar.zst",
    ),
    "eval": (
        OUTPUTS / "unseen_7x7x5_power06_png_exclude_requested",
        OUTPUTS / "objaverse_245_eval_cache_2500_2999",
        "objaverse_245_eval_cache_2500_2999.tar.zst",
    ),
}

POINTMAP_SPLITS = {
    "train": (OUTPUTS / "moge3_vitg_source_pointmaps" / "objaverse_fixed_7x7x5_power06_source", "objaverse_245_train_point_map_0000_1999.tar.zst"),
    "eval": (OUTPUTS / "moge3_vitg_source_pointmaps" / "unseen_7x7x5_power06_source", "objaverse_245_eval_point_map_2500_2999.tar.zst"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--repo-dir", default=REPO_DIR)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--upload", action="store_true", help="Upload verified archives to Hugging Face.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="Build RGB Wan VAE caches from PNG datasets.")
    cache.add_argument("--split", choices=["train", "eval", "all"], default="all")
    cache.add_argument("--image-transform", choices=["rgb", "luminance"], default="rgb")
    cache.add_argument("--overwrite", action="store_true", help="Recompute existing scene cache files.")

    pointmaps = subparsers.add_parser("pointmaps", help="Build source-only MoGe point maps.")
    pointmaps.add_argument("--split", choices=["train", "eval", "all"], default="all")

    subparsers.add_parser("pbr-masks", help="Archive PBR and mask PNG files without NPY arrays.")
    return parser.parse_args()


def selected_splits(value: str) -> list[str]:
    return list(CACHE_SPLITS) if value == "all" else [value]


def maybe_upload(args: argparse.Namespace, archive: Path) -> None:
    if args.upload:
        upload_dataset_file(archive, args.repo, args.repo_dir)


def build_caches(args: argparse.Namespace) -> None:
    for split in selected_splits(args.split):
        dataset, output, archive_name = CACHE_SPLITS[split]
        command = [
            "/usr/bin/python3", str(VAE_SCRIPT),
            "--dataset-root", str(dataset), "--ckpt-dir", str(VAE_CHECKPOINT),
            "--out-dir", str(output), "--resolution", "480",
            "--image-transform", args.image_transform, "--batch-size", "4",
            "--dtype", "bf16", "--save-dtype", "bf16",
        ]
        if args.overwrite:
            command.append("--overwrite")
        run_gpu_shards(command, args.gpus, OUTPUTS / "objaverse_245_logs", f"cache_{split}")
        expected = len(list((dataset / "scenes").glob("scene_*")))
        actual = len(list((output / "scenes").glob("scene_*.pt")))
        manifest = json.loads((output / "cache_manifest.json").read_text())
        if actual != expected or manifest.get("image_transform") != args.image_transform:
            raise RuntimeError(f"Invalid {split} cache: expected={expected} actual={actual} transform={manifest.get('image_transform')}")
        archive = OUTPUTS / archive_name
        archive_directory(output, archive)
        maybe_upload(args, archive)
        print(f"[cache] split={split} scenes={actual} archive={archive}", flush=True)


def pointmap_entries(split: str) -> list[dict[str, str]]:
    entries: dict[str, str] = {}
    if split == "train":
        for path in (OUTPUTS / "objaverse_fixed_7x7x5_power06_png_0000_0999" / "scenes").glob("scene_*/source.png"):
            entries[path.parent.name] = str(path.resolve())
        for path in (ROOT / "source_images").glob("scene_*.png"):
            entries[path.stem] = str(path.resolve())
    else:
        dataset = OUTPUTS / "unseen_7x7x5_power06_png_exclude_requested"
        for path in [*dataset.glob("scenes/scene_*/source.png"), *dataset.glob("shard_*/scenes/scene_*/source.png")]:
            entries[path.parent.name] = str(path.resolve())
    return [{"scene_id": key, "image": entries[key]} for key in sorted(entries)]


def build_pointmaps(args: argparse.Namespace) -> None:
    manifest_root = OUTPUTS / "moge3_vitg_source_pointmaps"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in selected_splits(args.split):
        output, archive_name = POINTMAP_SPLITS[split]
        entries = pointmap_entries(split)
        manifest = manifest_root / f"{split}_source_inputs.json"
        manifest.write_text(json.dumps(entries, indent=2) + "\n")
        if len(list(output.glob("scene_*/source.npy"))) != len(entries):
            workers = []
            for shard_id, gpu in enumerate(args.gpus):
                start = len(entries) * shard_id // len(args.gpus)
                end = len(entries) * (shard_id + 1) // len(args.gpus)
                log = (manifest_root / f"{split}_source_gpu{gpu}.log").open("a")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                command = [str(MOGE_PYTHON), str(MOGE_EXTRACTOR), "--manifest", str(manifest), "--output-root", str(output), "--offset", str(start), "--limit", str(end - start)]
                workers.append((gpu, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            failures = []
            for gpu, process, log in workers:
                code = process.wait()
                log.close()
                if code:
                    failures.append((gpu, code))
            if failures:
                raise RuntimeError(f"Point-map workers failed: {failures}")
        actual = len(list(output.glob("scene_*/source.npy")))
        if actual != len(entries):
            raise RuntimeError(f"Invalid {split} point maps: expected={len(entries)} actual={actual}")
        archive = manifest_root / archive_name
        archive_directory(output, archive)
        maybe_upload(args, archive)
        print(f"[pointmaps] split={split} scenes={actual} archive={archive}", flush=True)


def archive_pbr_masks(args: argparse.Namespace) -> None:
    dataset = OUTPUTS / "objaverse_fixed_7x7x5_power06_png_0000_0999"
    patterns = ("scenes/scene_*/pbr/*.png", "scenes/scene_*/masks/*.png", "scenes/scene_*/samples/position/*_masks/*.png")
    paths = (path for pattern in patterns for path in dataset.glob(pattern))
    archive = OUTPUTS / "objaverse_245_train_pbr_masks_0000_0999.tar.zst"
    count = archive_paths(paths, dataset, archive)
    maybe_upload(args, archive)
    print(f"[pbr-masks] files={count} archive={archive}", flush=True)


def main() -> None:
    args = parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("--gpus must not contain duplicates")
    if args.command == "cache":
        build_caches(args)
    elif args.command == "pointmaps":
        build_pointmaps(args)
    else:
        archive_pbr_masks(args)


if __name__ == "__main__":
    main()
