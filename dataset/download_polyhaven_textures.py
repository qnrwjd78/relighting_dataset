from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from utils.util_progress import progress_bar, progress_write


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.polyhaven.com"
DEFAULT_CATEGORIES = ["concrete", "plaster-concrete", "wood", "tiles", "brick"]
DEFAULT_MAPS = {
    "albedo": "Diffuse",
    "roughness": "Rough",
    "normal": "nor_gl",
}
DEFAULT_USER_AGENT = "relighting-dataset-polyhaven-texture-downloader/0.1 research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Poly Haven PBR textures and write outputs/previews/polyhaven_textures manifest."
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--resolution", default="2k")
    parser.add_argument("--format", choices=["jpg", "png", "exr"], default="jpg")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--sort", choices=["random", "downloads", "name"], default="random")
    parser.add_argument("--out-dir", default="data/polyhaven_textures")
    parser.add_argument("--metadata-out", default="outputs/previews/polyhaven_textures/polyhaven_textures.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Poly Haven asks API clients to send a unique User-Agent.",
    )
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def repo_relative_or_abs(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def request_json(url: str, user_agent: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def asset_list(category: str, user_agent: str) -> dict:
    query = urllib.parse.urlencode({"t": "textures", "categories": category})
    return request_json(f"{API_ROOT}/assets?{query}", user_agent)


def asset_files(asset_id: str, user_agent: str) -> dict:
    return request_json(f"{API_ROOT}/files/{asset_id}", user_agent)


def order_assets(assets: dict, sort_mode: str, rng: random.Random) -> list[tuple[str, dict]]:
    items = list(assets.items())
    if sort_mode == "downloads":
        items.sort(key=lambda item: int(item[1].get("download_count", 0)), reverse=True)
    elif sort_mode == "name":
        items.sort(key=lambda item: item[1].get("name", item[0]).lower())
    else:
        rng.shuffle(items)
    return items


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, path: Path, expected_md5: str | None, overwrite: bool, user_agent: str) -> None:
    if path.exists() and not overwrite:
        if expected_md5 and md5sum(path) != expected_md5:
            progress_write(f"[PolyHavenTex] Existing file has wrong md5, re-downloading: {path}")
        else:
            progress_write(f"[PolyHavenTex] Exists: {path}")
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("content-length") or 0) or None
        with progress_bar(total=total, desc=path.name, unit="B", leave=False, unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(path)

    if expected_md5:
        actual = md5sum(path)
        if actual != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {path}: expected {expected_md5}, got {actual}")
    progress_write(f"[PolyHavenTex] Downloaded: {path}")


def select_texture_file(files: dict, polyhaven_key: str, resolution: str, preferred_format: str) -> tuple[str, dict] | None:
    by_resolution = files.get(polyhaven_key, {})
    by_format = by_resolution.get(resolution)
    if not isinstance(by_format, dict):
        return None
    for fmt in [preferred_format, "jpg", "png", "exr"]:
        file_meta = by_format.get(fmt)
        if isinstance(file_meta, dict) and file_meta.get("url"):
            return fmt, file_meta
    return None


def build_texture_entry(
    asset_id: str,
    category: str,
    meta: dict,
    files: dict,
    out_dir: Path,
    resolution: str,
    preferred_format: str,
) -> dict | None:
    maps: dict[str, dict] = {}
    for map_name, polyhaven_key in DEFAULT_MAPS.items():
        selected = select_texture_file(files, polyhaven_key, resolution, preferred_format)
        if selected is None:
            if map_name == "albedo":
                return None
            continue
        fmt, file_meta = selected
        dest = out_dir / category / asset_id / f"{asset_id}_{map_name}_{resolution}.{fmt}"
        maps[map_name] = {
            "path": repo_relative_or_abs(dest),
            "url": file_meta["url"],
            "size": file_meta.get("size"),
            "md5": file_meta.get("md5"),
            "resolution": resolution,
            "format": fmt,
            "polyhaven_key": polyhaven_key,
        }
    return {
        "id": asset_id,
        "name": meta.get("name", asset_id),
        "download_category": category,
        "asset_categories": meta.get("categories", []),
        "maps": maps,
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = resolve_repo_path(args.out_dir)
    metadata_path = resolve_repo_path(args.metadata_out)

    selected: list[dict] = []
    seen_ids: set[str] = set()
    with progress_bar(args.categories, total=len(args.categories), desc="PolyHaven texture categories", unit="category") as pbar:
        for category in pbar:
            pbar.set_postfix(category=category)
            assets = asset_list(category, args.user_agent)
            chosen = []
            for asset_id, meta in order_assets(assets, args.sort, rng):
                if asset_id in seen_ids:
                    continue
                chosen.append((asset_id, meta))
                if len(chosen) >= args.per_category:
                    break
            progress_write(f"[PolyHavenTex] {category}: selected {len(chosen)} / {len(assets)} textures")
            for asset_id, meta in chosen:
                seen_ids.add(asset_id)
                selected.append({"id": asset_id, "category": category, "meta": meta})

    manifest_entries: list[dict] = []
    with progress_bar(selected, total=len(selected), desc="PolyHaven texture downloads", unit="texture") as pbar:
        for i, item in enumerate(pbar, 1):
            asset_id = item["id"]
            pbar.set_postfix(category=item["category"], asset=asset_id[:24])
            try:
                files = asset_files(asset_id, args.user_agent)
                entry = build_texture_entry(
                    asset_id,
                    item["category"],
                    item["meta"],
                    files,
                    out_dir,
                    args.resolution,
                    args.format,
                )
            except KeyError as exc:
                progress_write(f"[PolyHavenTex] Skipping {asset_id}: malformed file metadata ({exc})")
                continue
            if entry is None:
                progress_write(f"[PolyHavenTex] Skipping {asset_id}: missing {args.resolution} albedo map")
                continue

            manifest_entries.append(entry)
            progress_write(f"[PolyHavenTex] {i:03d}/{len(selected):03d} {item['category']} {asset_id}")
            if not args.dry_run:
                for map_info in entry["maps"].values():
                    try:
                        download_file(
                            map_info["url"],
                            resolve_repo_path(map_info["path"]),
                            map_info.get("md5"),
                            args.overwrite,
                            args.user_agent,
                        )
                    except urllib.error.URLError as exc:
                        raise RuntimeError(f"Failed to download {asset_id}: {exc}") from exc
                time.sleep(0.1)

    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "polyhaven_texture_manifest_v1",
            "resolution": args.resolution,
            "format_preference": args.format,
            "textures": manifest_entries,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        progress_write(f"[PolyHavenTex] Wrote metadata: {metadata_path}")
    else:
        progress_write("[PolyHavenTex] Dry run; no files or manifest were downloaded/written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
