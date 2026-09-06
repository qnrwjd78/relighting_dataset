#!/usr/bin/env python3
"""Shared process, archive, checksum, and Hugging Face helpers."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


def run_gpu_shards(
    command_prefix: Sequence[str],
    gpus: Sequence[str],
    log_dir: Path,
    log_prefix: str,
) -> None:
    """Run one command per GPU, appending ``--num-shards`` and ``--shard-id``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for shard_id, gpu in enumerate(gpus):
        log = (log_dir / f"{log_prefix}_gpu{gpu}.log").open("a")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [*command_prefix, "--num-shards", str(len(gpus)), "--shard-id", str(shard_id)]
        workers.append((gpu, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failures = []
    for gpu, process, log in workers:
        code = process.wait()
        log.close()
        if code:
            failures.append((gpu, code))
    if failures:
        raise RuntimeError(f"GPU workers failed: {failures}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_paths(paths: Iterable[Path], root: Path, archive: Path) -> int:
    """Create a verified tar.zst containing paths relative to ``root``."""
    selected = sorted({path.resolve() for path in paths if path.is_file()})
    if not selected:
        raise RuntimeError("No files selected for archive")
    relative = [str(path.relative_to(root.resolve())) for path in selected]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb") as file_list:
        file_list.write(b"\0".join(path.encode() for path in relative) + b"\0")
        file_list.flush()
        subprocess.run(
            ["tar", "-I", "zstd -3 -T0", "-cf", str(archive), "-C", str(root), "--null", "-T", file_list.name],
            check=True,
        )
    verify_archive(archive)
    return len(selected)


def archive_directory(directory: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-I", "zstd -3 -T0", "-cf", str(archive), "-C", str(directory.parent), directory.name],
        check=True,
    )
    verify_archive(archive)


def verify_archive(archive: Path) -> None:
    subprocess.run(["zstd", "-tq", str(archive)], check=True)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n")


def upload_dataset_file(
    archive: Path,
    repo_id: str,
    repo_dir: str,
    commit_message: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=f"{repo_dir.rstrip('/')}/{archive.name}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or f"Upload {archive.name}",
    )
