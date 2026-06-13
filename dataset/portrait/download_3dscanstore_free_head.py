from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

DATASET_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_progress import progress_bar, progress_write


SOURCE_PAGE = "https://www.3dscanstore.com/blog/Free-3D-Head-Model"
DOWNLOAD_URL = "https://samplescan.s3.us-west-2.amazonaws.com/3D_ScanStore_Free+Head.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "3dscanstore_free_head"
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "previews" / "3dscanstore_free_head"
USER_AGENT = "relighting-dataset-3dscanstore-free-head/1.0"
SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the 3D Scan Store free HD head sample.")
    parser.add_argument("--url", default=DOWNLOAD_URL)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_PREVIEW_ROOT / "3dscanstore_free_head_objects.txt"),
    )
    parser.add_argument(
        "--metadata-out",
        default=str(DEFAULT_PREVIEW_ROOT / "3dscanstore_free_head_download_meta.json"),
    )
    args = parser.parse_args()
    if args.delete_zip_after_extract and not (args.extract or args.extract_only):
        parser.error("--delete-zip-after-extract requires --extract or --extract-only")
    return args


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000.0 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1000.0
    return f"{size:.1f}TB"


def request_head(session: requests.Session, url: str, timeout: float) -> int | None:
    try:
        response = session.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
    except Exception:
        return None
    value = response.headers.get("content-length")
    return int(value) if value and value.isdigit() else None


def download_file(
    session: requests.Session,
    url: str,
    target: Path,
    *,
    size_bytes: int | None,
    chunk_size: int,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        if size_bytes is None or target.stat().st_size == size_bytes:
            progress_write(f"[3DScanStore] exists: {target}")
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if overwrite:
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total = size_bytes or int(response.headers.get("content-length") or 0) or None
                with part.open("wb") as handle:
                    with progress_bar(total=total, desc=target.name, unit="B", leave=False, unit_scale=True, unit_divisor=1024) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                handle.write(chunk)
                                pbar.update(len(chunk))
            part.replace(target)
            return
        except Exception as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        progress_write(f"[3DScanStore] extracted exists: {extract_dir}")
        return
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def find_asset_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS)


def main() -> int:
    args = parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    out_dir = Path(args.out_dir).resolve()
    zip_path = out_dir / "zips" / "3D_ScanStore_Free_Head.zip"
    extract_dir = out_dir / "extracted"
    manifest = Path(args.manifest).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    chunk_size = args.chunk_size_mb * 1024 * 1024
    size_bytes = request_head(session, args.url, args.timeout)

    progress_write(f"[3DScanStore] Source: {SOURCE_PAGE}")
    progress_write(f"[3DScanStore] Download: {args.url}")
    progress_write(f"[3DScanStore] Size: {format_bytes(size_bytes)}")
    progress_write(f"[3DScanStore] Output: {out_dir}")
    if args.dry_run:
        return 0

    if not args.extract_only:
        download_file(
            session,
            args.url,
            zip_path,
            size_bytes=size_bytes,
            chunk_size=chunk_size,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )
    if args.extract or args.extract_only:
        if not zip_path.exists():
            raise SystemExit(f"Zip not found: {zip_path}")
        extract_zip(zip_path, extract_dir, args.overwrite)
        if args.delete_zip_after_extract:
            zip_path.unlink(missing_ok=True)
            progress_write(f"[3DScanStore] Deleted zip: {zip_path}")

    assets = find_asset_files(extract_dir)
    metadata = {
        "source": "3dscanstore_free_head",
        "source_page": SOURCE_PAGE,
        "download_url": args.url,
        "license_note": "Check the source page for current 3D Scan Store sample license terms before redistribution.",
        "size_bytes": size_bytes,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir) if extract_dir.exists() else None,
        "assets": [str(path) for path in assets],
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if extract_dir.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(str(path) for path in assets) + ("\n" if assets else ""), encoding="utf-8")
        progress_write(f"[3DScanStore] wrote manifest: {manifest}")
    progress_write(f"[3DScanStore] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
