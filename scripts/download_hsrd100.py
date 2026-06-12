from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests


REPO_ID = "digitalrealitylab/HSRD-100"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract HSRD-100 scan LOD zip files.")
    parser.add_argument("--lod", choices=["LOD0", "LOD1", "LOD2"], default="LOD1")
    parser.add_argument("--out-dir", default="assets/hsrd100", help="Output asset root.")
    parser.add_argument("--manifest", default=None, help="Manifest text path. Defaults to manifests/hsrd100_<lod>_objects.txt.")
    parser.add_argument("--metadata-out", default=None, help="Metadata JSON path. Defaults to manifests/hsrd100_<lod>_meta.json.")
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N scans.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zips.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip files after extraction.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and re-extract existing assets.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def request_json(url: str, timeout: float, retries: int) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def list_dir(path: str, timeout: float, retries: int) -> list[dict]:
    return request_json(f"{API_ROOT}/{quote(path)}?expand=1", timeout, retries)


def discover_lod_files(lod: str, timeout: float, retries: int) -> list[dict]:
    files: list[dict] = []
    people = [item for item in list_dir("data", timeout, retries) if item.get("type") == "directory"]
    for person in people:
        poses = [item for item in list_dir(person["path"], timeout, retries) if item.get("type") == "directory"]
        for pose in poses:
            scans_path = f"{pose['path']}/scans"
            scan_files = [item for item in list_dir(scans_path, timeout, retries) if item.get("type") == "file"]
            suffix = f"-Scan-{lod}.zip"
            match = next((item for item in scan_files if item["path"].endswith(suffix)), None)
            if match is not None:
                files.append(match)
    return sorted(files, key=lambda item: item["path"])


def download_file(remote_path: str, dst: Path, expected_size: int | None, timeout: float, retries: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        if expected_size is None or dst.stat().st_size == expected_size:
            print(f"[HSRD] exists: {dst}", flush=True)
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = f"{RESOLVE_ROOT}/{quote(remote_path)}?download=true"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(dst)
            return
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to download {remote_path}: {last_error}")


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        print(f"[HSRD] extracted exists: {extract_dir}", flush=True)
        return
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def find_primary_obj(extract_dir: Path) -> Path | None:
    obj_files = sorted(extract_dir.rglob("*.obj"), key=lambda path: (len(path.parts), str(path)))
    if not obj_files:
        return None
    scan_objs = [path for path in obj_files if "scan" in path.stem.lower()]
    return scan_objs[0] if scan_objs else obj_files[0]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    manifest = Path(args.manifest) if args.manifest else repo_root / "manifests" / f"hsrd100_{args.lod.lower()}_objects.txt"
    metadata_out = Path(args.metadata_out) if args.metadata_out else repo_root / "manifests" / f"hsrd100_{args.lod.lower()}_meta.json"

    print(f"[HSRD] discovering {args.lod} files from {REPO_ID}", flush=True)
    files = discover_lod_files(args.lod, args.timeout, args.retries)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No HSRD-100 {args.lod} scan files found.")

    metadata: list[dict] = []
    manifest_lines: list[str] = []
    zip_root = out_dir / "zips" / args.lod
    extract_root = out_dir / args.lod

    for index, item in enumerate(files):
        remote_path = item["path"]
        pose_name = Path(remote_path).stem.replace("-Scan-" + args.lod, "")
        person_name = remote_path.split("/")[1]
        zip_path = zip_root / person_name / Path(remote_path).name
        extract_dir = extract_root / person_name / pose_name
        print(f"[HSRD] {index + 1}/{len(files)} {remote_path}", flush=True)
        download_file(remote_path, zip_path, item.get("size"), args.timeout, args.retries, args.overwrite)
        primary_obj = None
        if args.extract:
            extract_zip(zip_path, extract_dir, args.overwrite)
            primary_obj = find_primary_obj(extract_dir)
            if primary_obj is not None:
                manifest_lines.append(str(primary_obj))
            if not args.keep_zip:
                zip_path.unlink(missing_ok=True)
        metadata.append(
            {
                "remote_path": remote_path,
                "size_bytes": item.get("size"),
                "zip_path": str(zip_path),
                "extract_dir": str(extract_dir) if args.extract else None,
                "primary_obj": str(primary_obj) if primary_obj else None,
            }
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    if args.extract:
        manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
        print(f"[HSRD] wrote manifest: {manifest}", flush=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[HSRD] wrote metadata: {metadata_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
