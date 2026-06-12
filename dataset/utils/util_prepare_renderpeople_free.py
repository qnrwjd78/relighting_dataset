from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

DATASET_ROOT = Path(__file__).resolve().parents[1]
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from utils.util_progress import progress_bar, progress_write


SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare officially free RenderPeople packages from local zips or direct zip URLs."
    )
    parser.add_argument("--zip-dir", default=None, help="Folder containing RenderPeople free zip files.")
    parser.add_argument("--urls-file", default=None, help="Text file with direct RenderPeople free zip URLs, one per line.")
    parser.add_argument("--out-dir", default="data/renderpeople_free", help="Extraction root.")
    parser.add_argument("--manifest", default="outputs/previews/renderpeople_free/renderpeople_free_objects.txt")
    parser.add_argument("--metadata-out", default="outputs/previews/renderpeople_free/renderpeople_free_prepare_meta.json")
    parser.add_argument("--keep-zip", action="store_true", help="Keep URL-downloaded zip files under out-dir/zips.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def download_url(url: str, dst: Path, timeout: float, retries: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        progress_write(f"[RenderPeople] exists: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    total = int(response.headers.get("content-length") or 0) or None
                    with progress_bar(total=total, desc=dst.name, unit="B", leave=False, unit_scale=True, unit_divisor=1024) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                pbar.update(len(chunk))
            tmp.replace(dst)
            return
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def safe_stem(value: str) -> str:
    stem = Path(urlparse(value).path).stem or Path(value).stem
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        progress_write(f"[RenderPeople] extracted exists: {extract_dir}")
        return
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def candidate_score(path: Path) -> tuple[int, int, str]:
    preferred = [".blend", ".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"]
    ext_rank = preferred.index(path.suffix.lower()) if path.suffix.lower() in preferred else len(preferred)
    return (ext_rank, len(path.parts), str(path))


def find_asset_files(root: Path) -> list[Path]:
    grouped: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            grouped.setdefault(path.parent, []).append(path)
    return sorted((sorted(paths, key=candidate_score)[0] for paths in grouped.values()), key=str)


def read_urls(path: Path) -> list[str]:
    urls = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def main() -> int:
    args = parse_args()
    if not args.zip_dir and not args.urls_file:
        raise SystemExit("Pass --zip-dir for local zips or --urls-file for direct free zip URLs.")
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    manifest = (repo_root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    metadata_out = (repo_root / args.metadata_out).resolve() if not Path(args.metadata_out).is_absolute() else Path(args.metadata_out)

    zip_paths: list[Path] = []
    metadata: list[dict] = []
    if args.zip_dir:
        zip_dir = Path(args.zip_dir).resolve()
        if not zip_dir.exists():
            raise SystemExit(f"zip-dir does not exist: {zip_dir}")
        zip_paths.extend(sorted(zip_dir.glob("*.zip")))
    if args.urls_file:
        urls = read_urls(Path(args.urls_file).resolve())
        for url in urls:
            zip_path = out_dir / "zips" / f"{safe_stem(url)}.zip"
            progress_write(f"[RenderPeople] download {url}")
            download_url(url, zip_path, args.timeout, args.retries, args.overwrite)
            zip_paths.append(zip_path)

    if not zip_paths:
        raise SystemExit("No zip files found.")

    with progress_bar(zip_paths, total=len(zip_paths), desc="RenderPeople extract", unit="zip") as pbar:
        for zip_path in pbar:
            pbar.set_postfix(file=zip_path.name[:32])
            extract_dir = out_dir / "extracted" / safe_stem(str(zip_path))
            progress_write(f"[RenderPeople] extract {zip_path}")
            extract_zip(zip_path, extract_dir, args.overwrite)
            metadata.append({"zip_path": str(zip_path), "extract_dir": str(extract_dir)})
            if args.urls_file and not args.keep_zip and str(zip_path).startswith(str(out_dir / "zips")):
                zip_path.unlink(missing_ok=True)

    assets = find_asset_files(out_dir / "extracted")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in assets) + ("\n" if assets else ""), encoding="utf-8")
    metadata_out.write_text(
        json.dumps({"packages": metadata, "assets": [str(path) for path in assets]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    progress_write(f"[RenderPeople] wrote manifest: {manifest}")
    progress_write(f"[RenderPeople] wrote metadata: {metadata_out}")
    progress_write(f"[RenderPeople] found assets: {len(assets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
