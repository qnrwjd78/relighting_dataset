from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "configs").exists() and (path / "tokenlight_dataset").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root()
API_ROOT = "https://www.blenderkit.com/api/v1"
DEFAULT_USER_AGENT = "relighting-dataset-blenderkit-scene-downloader/0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search BlenderKit scenes/models and download .blend assets using a BlenderKit API key."
    )
    parser.add_argument("--query", default="interior room")
    parser.add_argument("--asset-type", choices=["scene", "model"], default="scene")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--out-dir", default="data/blenderkit")
    parser.add_argument("--manifest-out", default="outputs/previews/blenderkit/downloads.jsonl")
    parser.add_argument("--search-out", default="outputs/previews/blenderkit/search_results.json")
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--download-paid", action="store_true", help="Allow downloading paid assets if your account can access them.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-and-delete", action="store_true", help="Render review PNG/metadata and delete each .blend right after download.")
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument("--preview-dir", default="outputs/previews/blenderkit")
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=704)
    parser.add_argument("--preview-samples", type=int, default=32)
    parser.add_argument("--preview-engine", choices=["current", "eevee", "cycles", "workbench"], default="current")
    parser.add_argument("--skip-existing-index", action="store_true", help="Skip assets already present in --index-json.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        return resolve_repo_path(args.api_key_file).read_text(encoding="utf-8").strip()
    return ""


def headers(api_key: str, user_agent: str) -> dict[str, str]:
    result = {"User-Agent": user_agent}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def request_json(url: str, api_key: str, user_agent: str) -> dict:
    req = urllib.request.Request(url, headers=headers(api_key, user_agent))
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def urlopen_follow(url: str, api_key: str, user_agent: str):
    req = urllib.request.Request(url, headers=headers(api_key, user_agent))
    return urllib.request.urlopen(req, timeout=180)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_search_url(query: str, asset_type: str, page_size: int) -> str:
    search = query.strip()
    terms = []
    if search:
        terms.append(search)
    terms.append(f"asset_type:{asset_type}")
    terms.append("order:-score,_score")
    query_string = "+".join(urllib.parse.quote_plus(term) for term in terms)
    return (
        f"{API_ROOT}/search/?query={query_string}"
        f"&page_size={page_size}&dict_parameters=1&addon_version=3.19.0&blender_version=4.4.3"
    )


def search_assets(args: argparse.Namespace, api_key: str, max_results: int | None = None) -> list[dict]:
    url = build_search_url(args.query, args.asset_type, args.page_size)
    target_count = max_results if max_results is not None else args.max_results
    results = []
    while url and len(results) < target_count:
        data = request_json(url, api_key, args.user_agent)
        for row in data.get("results", []):
            if args.free_only and not row.get("isFree", False):
                continue
            if not args.download_paid and not row.get("isFree", False):
                continue
            results.append(row)
            if len(results) >= target_count:
                break
        url = data.get("next")
    return results


def usable_assets(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    assets = []
    for row in rows:
        if args.free_only and not row.get("isFree", False):
            continue
        if not args.download_paid and not row.get("isFree", False):
            continue
        assets.append(row)
    return assets


def iter_search_pages(args: argparse.Namespace, api_key: str):
    url = build_search_url(args.query, args.asset_type, args.page_size)
    page_index = 1
    while url:
        data = request_json(url, api_key, args.user_agent)
        rows = data.get("results", [])
        assets = usable_assets(rows, args)
        print(f"[BlenderKit] Search page {page_index}: {len(rows)} candidate(s), {len(assets)} usable")
        yield page_index, assets
        url = data.get("next")
        page_index += 1


def existing_asset_ids(index_json: str) -> set[str]:
    path = resolve_repo_path(index_json)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data if isinstance(data, list) else [])
    ids = set()
    for item in items:
        for key in ("asset_id", "asset_base_id"):
            value = item.get(key)
            if value:
                ids.add(str(value))
    return ids


def blend_file(asset: dict) -> dict | None:
    for file_meta in asset.get("files", []):
        if file_meta.get("fileType") == "blend":
            return file_meta
    return None


def filename_for(asset: dict, file_meta: dict) -> str:
    raw = file_meta.get("filename") or f"{asset.get('assetBaseId', asset.get('id'))}.blend"
    name = Path(raw).name
    if not name.endswith(".blend"):
        name += ".blend"
    safe = []
    for ch in f"{asset.get('assetBaseId', asset.get('id'))}_{name}":
        safe.append(ch if ch.isalnum() or ch in ".-_" else "_")
    return "".join(safe)


def resolve_download_url(download_url: str, api_key: str, user_agent: str, scene_uuid: str) -> str:
    sep = "&" if "?" in download_url else "?"
    data = request_json(f"{download_url}{sep}scene_uuid={scene_uuid}", api_key, user_agent)
    for key in ("filePath", "file_path", "url", "downloadUrl", "download_url"):
        if data.get(key):
            return data[key]
    files = data.get("files")
    if isinstance(files, list):
        for item in files:
            for key in ("filePath", "url", "downloadUrl"):
                if item.get(key):
                    return item[key]
    raise RuntimeError(f"Download response did not contain a file URL: {data}")


def download_file(url: str, path: Path, api_key: str, user_agent: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"[BlenderKit] Exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urlopen_follow(url, api_key, user_agent) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)
    print(f"[BlenderKit] Downloaded: {path}")


def curate_one(record: dict, args: argparse.Namespace) -> None:
    tmp_manifest = resolve_repo_path("outputs/previews/blenderkit/tmp_last_download.jsonl")
    tmp_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest.write_text(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "dataset" / "utils" / "util_curate_downloaded_scenes.py"),
        "--download-manifest",
        str(tmp_manifest),
        "--index-json",
        args.index_json,
        "--preview-dir",
        args.preview_dir,
        "--blender-cmd",
        args.blender_cmd,
        "--width",
        str(args.preview_width),
        "--height",
        str(args.preview_height),
        "--samples",
        str(args.preview_samples),
        "--engine",
        args.preview_engine,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    api_key = load_api_key(args)
    if not api_key and not args.dry_run:
        raise SystemExit("Missing BlenderKit API key. Set BLENDERKIT_API_KEY or pass --api-key-file.")

    search_out = resolve_repo_path(args.search_out)
    out_dir = resolve_repo_path(args.out_dir)
    manifest_out = resolve_repo_path(args.manifest_out)
    seen = existing_asset_ids(args.index_json) if args.skip_existing_index else set()
    inspected_assets = []
    records = []
    successful = 0
    skipped_existing = 0
    inspected_count = 0
    scene_uuid = str(uuid.uuid4())
    for page_index, page_assets in iter_search_pages(args, api_key):
        if successful >= args.max_results:
            break
        if not page_assets:
            print(f"[BlenderKit] Page {page_index} has no usable assets; checking next page.")
            continue
        page_success_before = successful
        page_inspected_before = inspected_count
        for asset in page_assets:
            if successful >= args.max_results:
                break
            inspected_count += 1
            inspected_assets.append(asset)
            if args.skip_existing_index and (
                str(asset.get("id")) in seen or str(asset.get("assetBaseId")) in seen
            ):
                skipped_existing += 1
                continue
            file_meta = blend_file(asset)
            if not file_meta:
                print(f"[BlenderKit] Skip no .blend: {asset.get('name')}")
                continue
            can_download = bool(asset.get("canDownload", False))
            if not can_download and not api_key:
                print(f"[BlenderKit] Skip not downloadable without login: {asset.get('name')}")
                continue
            dst = out_dir / filename_for(asset, file_meta)
            print(f"[BlenderKit] candidate={inspected_count:04d} success={successful}/{args.max_results} {asset.get('name')} -> {dst}")
            resolved_url = None
            if not args.dry_run:
                try:
                    resolved_url = resolve_download_url(file_meta["downloadUrl"], api_key, args.user_agent, scene_uuid)
                    download_file(resolved_url, dst, api_key, args.user_agent, args.overwrite)
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    print(f"[BlenderKit] Skip download HTTP {exc.code} for {asset.get('name')}: {detail}", file=sys.stderr)
                    continue
                except urllib.error.URLError as exc:
                    print(f"[BlenderKit] Skip download URL error for {asset.get('name')}: {exc}", file=sys.stderr)
                    continue
                except RuntimeError as exc:
                    print(f"[BlenderKit] Skip download runtime error for {asset.get('name')}: {exc}", file=sys.stderr)
                    continue
            record = {
                "source": "blenderkit",
                "license": asset.get("license"),
                "asset_type": asset.get("assetType"),
                "asset_id": asset.get("id"),
                "asset_base_id": asset.get("assetBaseId"),
                "name": asset.get("name"),
                "is_free": asset.get("isFree"),
                "can_download": asset.get("canDownload"),
                "download_path": repo_relative(dst),
                "blend_paths": [repo_relative(dst)],
                "download_api_url": file_meta.get("downloadUrl"),
                "resolved_url": resolved_url,
                "thumbnail": asset.get("thumbnailLargeUrl") or asset.get("thumbnailMiddleUrl"),
            }
            records.append(record)
            if args.preview_and_delete and not args.dry_run:
                curate_one(record, args)
            if asset.get("id"):
                seen.add(str(asset.get("id")))
            if asset.get("assetBaseId"):
                seen.add(str(asset.get("assetBaseId")))
            successful += 1
            time.sleep(0.2)
        page_success = successful - page_success_before
        page_inspected = inspected_count - page_inspected_before
        print(f"[BlenderKit] Page {page_index} processed: inspected={page_inspected}, successful={page_success}")
        if page_success > 0:
            break
        print(f"[BlenderKit] Page {page_index} produced no new previews; checking next page.")

    print(
        f"[BlenderKit] Inspected {inspected_count} candidate(s), "
        f"skipped_existing={skipped_existing}, successful={successful}/{args.max_results}."
    )

    if not args.dry_run:
        search_out.parent.mkdir(parents=True, exist_ok=True)
        search_out.write_text(json.dumps(inspected_assets, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with manifest_out.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        print(f"[BlenderKit] Wrote manifest: {manifest_out}")
        print(f"[BlenderKit] Wrote search results: {search_out}")
    else:
        for asset in inspected_assets[:10]:
            print(
                f"[BlenderKit] DRY {asset.get('name')} free={asset.get('isFree')} "
                f"canDownload={asset.get('canDownload')} license={asset.get('license')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
