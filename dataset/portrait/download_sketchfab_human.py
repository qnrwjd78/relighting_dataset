from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import requests

DATASET_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_progress import progress_bar, progress_write


API_ROOT = "https://api.sketchfab.com/v3"
DEFAULT_QUERIES = ["human head", "human bust", "portrait head", "face scan", "realistic human head", "person bust"]
DEFAULT_LICENSES = "cc0,by"
SUPPORTED_EXTS = {".glb", ".gltf", ".blend", ".fbx", ".obj", ".dae", ".ply", ".stl"}
USER_AGENT = "relighting-dataset-sketchfab-human/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search/download Sketchfab human portrait model candidates.")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--licenses", default=DEFAULT_LICENSES, help="Comma-separated Sketchfab license slugs.")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--out-dir", default="data/sketchfab_human")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN", ""))
    parser.add_argument("--token-file", default=None, help="Read Sketchfab API token from a local file.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--search-out", default="outputs/previews/sketchfab_human/search_results.json")
    parser.add_argument("--manifest", default="outputs/previews/sketchfab_human/sketchfab_human_objects.txt")
    parser.add_argument("--metadata-out", default="outputs/previews/sketchfab_human/sketchfab_human_download_meta.json")
    args = parser.parse_args()
    if args.delete_zip_after_extract and not (args.extract or args.extract_only):
        parser.error("--delete-zip-after-extract requires --extract or --extract-only")
    return args


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token.strip()
    if args.token_file:
        return resolve_repo_path(args.token_file).read_text(encoding="utf-8").strip()
    return ""


def headers(token: str = "") -> dict[str, str]:
    result = {"User-Agent": USER_AGENT}
    if token:
        result["Authorization"] = f"Token {token}"
    return result


def request_json(session: requests.Session, url: str, args: argparse.Namespace) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            response = session.get(url, timeout=args.timeout, headers=headers(args.token))
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def search_url(query: str, args: argparse.Namespace) -> str:
    params = {
        "type": "models",
        "downloadable": "true",
        "archives_flavours": "false",
        "q": query,
        "sort_by": "-likeCount",
        "count": str(args.page_size),
    }
    if args.licenses:
        params["licenses"] = args.licenses
    return f"{API_ROOT}/search?{urlencode(params)}"


def model_uid(row: dict) -> str | None:
    return row.get("uid") or row.get("id")


def normalize_license(row: dict) -> str | None:
    value = row.get("license")
    if isinstance(value, dict):
        return value.get("slug") or value.get("label") or value.get("name")
    return str(value) if value else None


def discover(session: requests.Session, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for query in args.queries:
        url = search_url(query, args)
        while url and len(rows) < args.max_results:
            data = request_json(session, url, args)
            for row in data.get("results", []):
                uid = model_uid(row)
                if not uid or uid in seen:
                    continue
                seen.add(uid)
                row["_query"] = query
                rows.append(row)
                if len(rows) >= args.max_results:
                    break
            url = data.get("next")
    return rows


def safe_name(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)[:120] or "sketchfab_model"


def download_info(session: requests.Session, uid: str, args: argparse.Namespace) -> dict:
    return request_json(session, f"{API_ROOT}/models/{uid}/download", args)


def download_zip(
    session: requests.Session,
    url: str,
    target: Path,
    *,
    chunk_size: int,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        progress_write(f"[Sketchfab] exists: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if overwrite:
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0) or None
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
    raise RuntimeError(f"Failed to download {target.name}: {last_error}")


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        progress_write(f"[Sketchfab] extracted exists: {extract_dir}")
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


def author_name(row: dict) -> str | None:
    user = row.get("user")
    if isinstance(user, dict):
        return user.get("displayName") or user.get("username")
    return None


def model_url(row: dict) -> str | None:
    return row.get("viewerUrl") or row.get("embedUrl") or row.get("url")


def main() -> int:
    args = parse_args()
    args.token = load_token(args)
    session = requests.Session()
    out_dir = Path(args.out_dir).resolve()
    zip_dir = out_dir / "zips"
    extract_root = out_dir / "extracted"
    search_out = Path(args.search_out).resolve()
    manifest = Path(args.manifest).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    chunk_size = args.chunk_size_mb * 1024 * 1024

    if args.extract_only:
        rows = []
    else:
        rows = discover(session, args)
        progress_write(f"[Sketchfab] candidates: {len(rows)}")
        for row in rows:
            progress_write(f"  - {row.get('name')} uid={model_uid(row)} license={normalize_license(row)}")
        if args.dry_run:
            return 0
        search_out.parent.mkdir(parents=True, exist_ok=True)
        search_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.token:
            raise SystemExit("Sketchfab downloads require SKETCHFAB_API_TOKEN or --token.")

    metadata: list[dict] = []
    if args.extract_only:
        zip_paths = sorted(zip_dir.glob("*.zip"))
        for zip_path in zip_paths:
            extract_dir = extract_root / zip_path.stem
            extract_zip(zip_path, extract_dir, args.overwrite)
            if args.delete_zip_after_extract:
                zip_path.unlink(missing_ok=True)
            metadata.append({"zip_path": str(zip_path), "extract_dir": str(extract_dir)})
    else:
        with progress_bar(rows, total=len(rows), desc="Sketchfab downloads", unit="model") as pbar:
            for row in pbar:
                uid = model_uid(row)
                if not uid:
                    continue
                name = safe_name(row.get("name") or uid)
                pbar.set_postfix(model=name[:24])
                zip_path = zip_dir / f"{uid}_{name}.zip"
                extract_dir = extract_root / f"{uid}_{name}"
                record = {
                    "source": "sketchfab",
                    "uid": uid,
                    "name": row.get("name"),
                    "author": author_name(row),
                    "license": normalize_license(row),
                    "model_url": model_url(row),
                    "query": row.get("_query"),
                    "zip_path": str(zip_path),
                    "extract_dir": str(extract_dir) if args.extract else None,
                }
                try:
                    info = download_info(session, uid, args)
                    gltf = info.get("gltf")
                    if not isinstance(gltf, dict) or not gltf.get("url"):
                        raise RuntimeError("Sketchfab download response did not include glTF archive URL.")
                    record["download_size_bytes"] = gltf.get("size")
                    download_zip(
                        session,
                        gltf["url"],
                        zip_path,
                        chunk_size=chunk_size,
                        timeout=args.timeout,
                        retries=args.retries,
                        overwrite=args.overwrite,
                    )
                    if args.extract:
                        extract_zip(zip_path, extract_dir, args.overwrite)
                        if args.delete_zip_after_extract:
                            zip_path.unlink(missing_ok=True)
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    progress_write(f"[Sketchfab] Failed: {uid} {row.get('name')}: {exc}")
                    if not args.skip_failed:
                        raise
                else:
                    record["status"] = "ok"
                metadata.append(record)

    assets = find_asset_files(extract_root)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps({"items": metadata, "assets": [str(path) for path in assets]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if extract_root.exists():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(str(path) for path in assets) + ("\n" if assets else ""), encoding="utf-8")
        progress_write(f"[Sketchfab] wrote manifest: {manifest}")
    progress_write(f"[Sketchfab] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
