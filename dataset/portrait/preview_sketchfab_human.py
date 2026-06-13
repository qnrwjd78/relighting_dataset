from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode

DATASET_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_progress import progress_bar, progress_write


API_ROOT = "https://api.sketchfab.com/v3"
DATASET_NAME = "sketchfab_human"
DEFAULT_QUERIES_FILE = "dataset/portrait/queries_sketchfab_human.txt"
DEFAULT_QUERIES = [
    "human head",
    "human bust",
    "portrait head",
    "face scan",
    "realistic human head",
    "person bust",
]
DEFAULT_LICENSES = "cc0,by"
SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".dae", ".ply", ".stl"}
PREFERRED_EXTS = [".glb", ".gltf", ".blend", ".fbx", ".obj", ".dae", ".ply", ".stl"]
USER_AGENT = "relighting-dataset-sketchfab-human-preview/1.0"


def blender_argv() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def running_in_blender() -> bool:
    try:
        import bpy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def repo_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_name(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)[:120] or "sketchfab_model"


def parse_single_asset_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one temporary Sketchfab asset preview inside Blender.")
    parser.add_argument("--single-asset-preview", action="store_true")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--record-json", default=None)
    return parser.parse_args(argv)


def run_single_asset_preview(argv: list[str]) -> int:
    args = parse_single_asset_args(argv)
    from mathutils import Vector
    from utils import util_preview_assets as preview_assets

    asset = Path(args.asset).resolve()
    preview_path = Path(args.preview).resolve()
    metadata_path = Path(args.metadata).resolve()
    record = {}
    if args.record_json:
        record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))

    bbox_min = Vector((0.0, 0.0, 0.0))
    bbox_max = Vector((0.0, 0.0, 0.0))
    status = "ok"
    error = None
    try:
        preview_assets.clear_scene()
        objects = preview_assets.import_asset(asset)
        bbox_min, bbox_max = preview_assets.normalize_for_preview(objects, upright=True)
        preview_assets.setup_camera_and_lights(bbox_min, bbox_max, args.resolution)
        preview_assets.render_preview(preview_path)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        progress_write(f"[SketchfabPreview] Failed: {asset}: {exc}")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": DATASET_NAME,
        "asset": str(asset),
        "source_path": str(asset),
        "preview": str(preview_path),
        "asset_type": "portrait_asset",
        "status": status,
        "error": error,
        "bbox_min_preview_space": preview_assets.vec_to_list(bbox_min),
        "bbox_max_preview_space": preview_assets.vec_to_list(bbox_max),
        "record": record,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if status == "ok" else 2


def maybe_run_legacy_root_preview() -> int | None:
    argv = blender_argv()
    if "--root" not in argv:
        return None
    from utils.util_preview_assets import main as preview_main

    return preview_main(DATASET_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search Sketchfab human portrait candidates, temporarily download one model at a time, "
            "render preview PNG/metadata, then delete the downloaded archive and extracted files."
        )
    )
    parser.add_argument("--target-count", type=int, default=None, help="Successful preview target. Defaults to --max-results.")
    parser.add_argument("--max-results", type=int, default=50, help="Maximum Sketchfab candidates to inspect.")
    parser.add_argument("--queries", nargs="+", default=None)
    parser.add_argument("--queries-file", default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--licenses", default=DEFAULT_LICENSES, help="Comma-separated Sketchfab license slugs.")
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN", ""))
    parser.add_argument("--token-file", default=None, help="Read Sketchfab API token from a local file.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing-index", action="store_true", default=True)
    parser.add_argument("--no-skip-existing-index", dest="skip_existing_index", action="store_false")
    parser.add_argument("--temp-dir", default="data/sketchfab_human/_tmp_preview")
    parser.add_argument("--index-json", default="outputs/previews/sketchfab_human/sketchfab_human_index.json")
    parser.add_argument("--preview-dir", default="outputs/previews/sketchfab_human")
    parser.add_argument("--manifest-out", default="outputs/previews/sketchfab_human/downloads.jsonl")
    parser.add_argument("--search-out", default="outputs/previews/sketchfab_human/search_results.json")
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--preview-width", type=int, default=768)
    parser.add_argument("--preview-height", type=int, default=768, help="Accepted for BlenderKit parity; previews are square.")
    parser.add_argument("--preview-samples", type=int, default=64, help="Accepted for BlenderKit parity; shared preview uses its own samples.")
    parser.add_argument("--show-subprocess-output", action="store_true")
    parser.add_argument("--batch-log", default="outputs/previews/sketchfab_human/sketchfab_human_batches.log")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    if args.target_count is None:
        args.target_count = args.max_results
    if args.target_count > args.max_results:
        args.max_results = args.target_count
    return args


def load_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token.strip()
    if args.token_file:
        return resolve_repo_path(args.token_file).read_text(encoding="utf-8").strip()
    return ""


def read_queries(args: argparse.Namespace) -> list[str]:
    if args.queries:
        queries = args.queries
    elif args.queries_file and resolve_repo_path(args.queries_file).exists():
        queries = [
            line.strip()
            for line in resolve_repo_path(args.queries_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        queries = DEFAULT_QUERIES
    return list(dict.fromkeys(queries))


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


def author_name(row: dict) -> str | None:
    user = row.get("user")
    if isinstance(user, dict):
        return user.get("displayName") or user.get("username")
    return None


def model_url(row: dict) -> str | None:
    return row.get("viewerUrl") or row.get("embedUrl") or row.get("url")


def download_info(session: requests.Session, uid: str, args: argparse.Namespace) -> dict:
    return request_json(session, f"{API_ROOT}/models/{uid}/download", args)


def iter_candidates(session: requests.Session, args: argparse.Namespace, queries: list[str], seen_search: set[str]):
    emitted = 0
    for query in queries:
        url = search_url(query, args)
        while url and emitted < args.max_results:
            data = request_json(session, url, args)
            for row in data.get("results", []):
                uid = model_uid(row)
                if not uid or uid in seen_search:
                    continue
                seen_search.add(uid)
                row["_query"] = query
                emitted += 1
                yield row
                if emitted >= args.max_results:
                    break
            url = data.get("next")


def download_zip(
    session: requests.Session,
    url: str,
    target: Path,
    *,
    chunk_size: int,
    timeout: float,
    retries: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
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


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def find_asset_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS)


def asset_score(path: Path) -> tuple[int, int, str]:
    ext = path.suffix.lower()
    ext_rank = PREFERRED_EXTS.index(ext) if ext in PREFERRED_EXTS else len(PREFERRED_EXTS)
    scene_bonus = 0 if path.name.lower() in {"scene.gltf", "scene.glb"} else 1
    return (ext_rank, scene_bonus, str(path))


def select_preview_asset(extract_dir: Path) -> Path:
    assets = find_asset_files(extract_dir)
    if not assets:
        raise RuntimeError(f"No supported asset files found under {extract_dir}")
    return sorted(assets, key=asset_score)[0]


def load_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("items", []))
    return list(data)


def write_index(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "portrait_preview_harvest_index_v1",
        "dataset": DATASET_NAME,
        "count": len(items),
        "ok_count": sum(1 for item in items if item.get("status", "ok") == "ok"),
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def successful_count(items: list[dict]) -> int:
    return sum(1 for item in items if item.get("status", "ok") == "ok")


def existing_uids(items: list[dict]) -> set[str]:
    ids = set()
    for item in items:
        for key in ("uid", "asset_id"):
            value = item.get(key)
            if value:
                ids.add(str(value))
        record = item.get("record")
        if isinstance(record, dict):
            value = record.get("uid") or record.get("asset_id")
            if value:
                ids.add(str(value))
    return ids


def next_item_number(items: list[dict]) -> int:
    values = []
    for item in items:
        value = str(item.get("id", ""))
        match = re.search(r"(\d+)$", value)
        if match:
            values.append(int(match.group(1)))
    return max(values, default=0) + 1


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_preview_subprocess(asset: Path, preview_path: Path, metadata_path: Path, record_path: Path, args: argparse.Namespace) -> None:
    cmd = shlex.split(args.blender_cmd) + [
        "-b",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--single-asset-preview",
        "--asset",
        str(asset),
        "--preview",
        str(preview_path),
        "--metadata",
        str(metadata_path),
        "--resolution",
        str(args.preview_width),
        "--record-json",
        str(record_path),
    ]
    if args.show_subprocess_output:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return
    batch_log = resolve_repo_path(args.batch_log)
    batch_log.parent.mkdir(parents=True, exist_ok=True)
    with batch_log.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def row_record(row: dict, uid: str, zip_path: Path | None, extract_dir: Path | None) -> dict:
    return {
        "source": "sketchfab",
        "uid": uid,
        "name": row.get("name"),
        "author": author_name(row),
        "license": normalize_license(row),
        "model_url": model_url(row),
        "query": row.get("_query"),
        "zip_path": repo_relative(zip_path) if zip_path else None,
        "extract_dir": repo_relative(extract_dir) if extract_dir else None,
    }


def process_candidate(
    session: requests.Session,
    row: dict,
    args: argparse.Namespace,
    item_id: str,
    preview_dir: Path,
    temp_dir: Path,
) -> dict:
    uid = model_uid(row)
    if not uid:
        raise RuntimeError("Sketchfab row has no uid.")
    name = safe_name(row.get("name") or uid)
    work_dir = temp_dir / f"{uid}_{name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    zip_path = work_dir / f"{uid}_{name}.zip"
    extract_dir = work_dir / "extracted"
    record = row_record(row, uid, zip_path, extract_dir)
    preview_path = preview_dir / "img" / f"{DATASET_NAME}_{item_id}.png"
    metadata_path = preview_dir / "metadata" / f"{DATASET_NAME}_{item_id}.json"
    record_path = work_dir / "record.json"
    try:
        info = download_info(session, uid, args)
        gltf = info.get("gltf")
        if not isinstance(gltf, dict) or not gltf.get("url"):
            raise RuntimeError("Sketchfab download response did not include glTF archive URL.")
        record["download_size_bytes"] = gltf.get("size")
        if args.dry_run:
            record["status"] = "dry_run"
            return record
        download_zip(
            session,
            gltf["url"],
            zip_path,
            chunk_size=args.chunk_size_mb * 1024 * 1024,
            timeout=args.timeout,
            retries=args.retries,
        )
        extract_zip(zip_path, extract_dir)
        record["zip_deleted"] = True
        record["extract_deleted"] = True
        asset = select_preview_asset(extract_dir)
        record["selected_asset"] = repo_relative(asset)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        run_preview_subprocess(asset, preview_path, metadata_path, record_path, args)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        record["status"] = metadata.get("status", "ok")
        record["error"] = metadata.get("error")
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
    return {
        "id": item_id,
        "source": "sketchfab",
        "uid": uid,
        "name": row.get("name"),
        "author": author_name(row),
        "license": normalize_license(row),
        "model_url": model_url(row),
        "query": row.get("_query"),
        "preview_png": repo_relative(preview_path),
        "metadata_json": repo_relative(metadata_path),
        "asset_deleted": True,
        "zip_deleted": True,
        "status": record.get("status", "ok"),
        "error": record.get("error"),
        "record": record,
    }


def main() -> int:
    if running_in_blender():
        legacy_result = maybe_run_legacy_root_preview()
        if legacy_result is not None:
            return legacy_result
        argv = blender_argv()
        if "--single-asset-preview" in argv:
            return run_single_asset_preview(argv)
        raise SystemExit("Sketchfab preview Blender mode requires --root or --single-asset-preview.")

    args = parse_args()
    args.token = load_token(args)
    if not args.token and not args.dry_run:
        raise SystemExit("Sketchfab previews require SKETCHFAB_API_TOKEN or --token-file.")

    queries = read_queries(args)
    if not queries:
        raise SystemExit("No Sketchfab queries provided.")

    import requests

    session = requests.Session()
    index_json = resolve_repo_path(args.index_json)
    preview_dir = resolve_repo_path(args.preview_dir)
    manifest_out = resolve_repo_path(args.manifest_out)
    search_out = resolve_repo_path(args.search_out)
    temp_dir = resolve_repo_path(args.temp_dir)
    items = load_index(index_json)
    seen = existing_uids(items) if args.skip_existing_index else set()
    seen_search = set(seen)
    inspected: list[dict] = []
    item_number = next_item_number(items)

    progress_write(
        f"[SketchfabPreview] target={args.target_count} current_ok={successful_count(items)} "
        f"max_results={args.max_results} queries={len(queries)}"
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for row in iter_candidates(session, args, queries, seen_search):
            inspected.append(row)
            progress_write(
                f"[SketchfabPreview] DRY {row.get('name')} uid={model_uid(row)} "
                f"license={normalize_license(row)} query={row.get('_query')}"
            )
        search_out.parent.mkdir(parents=True, exist_ok=True)
        search_out.write_text(json.dumps(inspected, indent=2, ensure_ascii=False), encoding="utf-8")
        progress_write(f"[SketchfabPreview] Dry-run candidates: {len(inspected)}")
        return 0

    with progress_bar(total=args.target_count, initial=min(successful_count(items), args.target_count), desc="Sketchfab human previews", unit="asset") as pbar:
        for row in iter_candidates(session, args, queries, seen_search):
            if successful_count(items) >= args.target_count:
                break
            uid = model_uid(row)
            if args.skip_existing_index and uid in seen:
                continue
            inspected.append(row)
            item_id = f"{item_number:05d}"
            item_number += 1
            pbar.set_postfix(model=safe_name(row.get("name") or uid or "model")[:24], ok=successful_count(items))
            try:
                item = process_candidate(session, row, args, item_id, preview_dir, temp_dir)
            except subprocess.CalledProcessError as exc:
                item = {
                    "id": item_id,
                    "source": "sketchfab",
                    "uid": uid,
                    "name": row.get("name"),
                    "author": author_name(row),
                    "license": normalize_license(row),
                    "model_url": model_url(row),
                    "query": row.get("_query"),
                    "preview_png": repo_relative(preview_dir / "img" / f"{DATASET_NAME}_{item_id}.png"),
                    "metadata_json": repo_relative(preview_dir / "metadata" / f"{DATASET_NAME}_{item_id}.json"),
                    "asset_deleted": True,
                    "zip_deleted": True,
                    "status": "failed",
                    "error": f"Blender preview failed: {exc}",
                }
                progress_write(f"[SketchfabPreview] Failed preview: {row.get('name')} uid={uid}: {exc}")
                if not args.skip_failed:
                    raise
            except Exception as exc:
                item = {
                    "id": item_id,
                    "source": "sketchfab",
                    "uid": uid,
                    "name": row.get("name"),
                    "author": author_name(row),
                    "license": normalize_license(row),
                    "model_url": model_url(row),
                    "query": row.get("_query"),
                    "status": "failed",
                    "error": str(exc),
                    "asset_deleted": True,
                    "zip_deleted": True,
                }
                progress_write(f"[SketchfabPreview] Failed: {row.get('name')} uid={uid}: {exc}")
                if not args.skip_failed:
                    raise
            items.append(item)
            seen.add(str(uid))
            write_index(index_json, items)
            write_jsonl(manifest_out, item)
            if item.get("status") == "ok":
                pbar.update(1)
            if args.sleep > 0:
                time.sleep(args.sleep)

    search_out.parent.mkdir(parents=True, exist_ok=True)
    search_out.write_text(json.dumps(inspected, indent=2, ensure_ascii=False), encoding="utf-8")
    write_index(index_json, items)
    progress_write(f"[SketchfabPreview] Wrote index: {index_json}")
    progress_write(f"[SketchfabPreview] Wrote manifest: {manifest_out}")
    progress_write(f"[SketchfabPreview] Wrote search results: {search_out}")
    progress_write(f"[SketchfabPreview] ok={successful_count(items)}/{args.target_count}")
    return 0 if successful_count(items) >= min(args.target_count, args.max_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
