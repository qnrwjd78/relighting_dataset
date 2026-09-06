#!/usr/bin/env python3
"""Run resumable multi-GPU MoGe-3 archive batches and upload each verified shard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "extract_moge3_pointmaps_to_tar.py"
PYTHON = ROOT / "miniconda3" / "envs" / "moge3" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--mode", choices=["position", "source", "images"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repo", default="JaehoChae/blender_relight")
    parser.add_argument("--repo-path", default="objaverse_32/moge3_pointmaps")
    parser.add_argument("--keep-local-archives", action="store_true")
    return parser.parse_args()


def total_units(root: Path, mode: str) -> int:
    if mode in {"position", "source"}:
        return len(sorted((root / "scenes").glob("scene_*")))
    return len(sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ))


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{args.name}_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": []}
    completed = set(state.get("completed", []))
    count = total_units(input_root, args.mode)
    batches = [(offset, min(args.batch_size, count - offset)) for offset in range(0, count, args.batch_size)]

    for wave_start in range(0, len(batches), len(args.gpus)):
        wave = batches[wave_start:wave_start + len(args.gpus)]
        jobs = []
        for gpu, (offset, limit) in zip(args.gpus, wave):
            batch_id = f"{offset:06d}_{offset + limit - 1:06d}"
            if batch_id in completed:
                continue
            archive = output_dir / f"{args.name}_part{batch_id}.tar.zst"
            log = output_dir / f"{args.name}_part{batch_id}.log"
            command = [
                str(PYTHON), str(WORKER),
                "--input-root", str(input_root),
                "--output", str(archive),
                "--mode", args.mode,
                "--scene-offset", str(offset),
                "--scene-limit", str(limit),
            ]
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = log.open("w")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            jobs.append((batch_id, offset, limit, archive, log, handle, process))

        for batch_id, offset, limit, archive, log, handle, process in jobs:
            return_code = process.wait()
            handle.close()
            if return_code != 0:
                raise RuntimeError(f"MoGe batch {batch_id} failed; see {log}")
            subprocess.run(["zstd", "-tq", str(archive)], check=True)
            path_in_repo = f"{args.repo_path.rstrip('/')}/{archive.name}"
            subprocess.run(
                ["hf", "upload", args.repo, str(archive), path_in_repo, "--repo-type", "dataset"],
                check=True,
            )
            if not args.keep_local_archives:
                archive.unlink()
            completed.add(batch_id)
            state = {
                "name": args.name,
                "input_root": str(input_root),
                "mode": args.mode,
                "unit_count": count,
                "batch_size": args.batch_size,
                "completed": sorted(completed),
                "updated_at": time.time(),
            }
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"[batch] complete {batch_id} ({len(completed)}/{len(batches)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
