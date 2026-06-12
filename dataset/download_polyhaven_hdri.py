from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from utils.util_progress import progress_bar, progress_write


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.polyhaven.com"
DEFAULT_CATEGORIES = ["studio", "indoor", "outdoor", "urban", "nature"]
DEFAULT_USER_AGENT = "relighting-dataset-polyhaven-downloader/0.1 research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Poly Haven HDRIs and write outputs/previews/polyhaven_hdri manifests."
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--resolution", default="2k")
    parser.add_argument("--format", choices=["hdr", "exr"], default="hdr")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--sort", choices=["random", "downloads", "name"], default="random")
    parser.add_argument("--out-dir", default="data/polyhaven_hdri")
    parser.add_argument("--manifest", default="outputs/previews/polyhaven_hdri/polyhaven_hdri_hdris.txt")
    parser.add_argument("--metadata-out", default="outputs/previews/polyhaven_hdri/polyhaven_hdri_index.json")
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
    return request_json(f"{API_ROOT}/assets?t=hdris&categories={category}", user_agent)


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


def download_file(url: str, path: Path, expected_md5: str | None, overwrite: bool, user_agent: str) -> None:
    if path.exists() and not overwrite:
        if expected_md5 and md5sum(path) != expected_md5:
            progress_write(f"[PolyHaven] Existing file has wrong md5, re-downloading: {path}")
        else:
            progress_write(f"[PolyHaven] Exists: {path}")
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
    progress_write(f"[PolyHaven] Downloaded: {path}")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = resolve_repo_path(args.out_dir)
    manifest_path = resolve_repo_path(args.manifest)
    metadata_path = resolve_repo_path(args.metadata_out)

    selected: list[dict] = []
    seen_ids: set[str] = set()
    with progress_bar(args.categories, total=len(args.categories), desc="PolyHaven categories", unit="category") as pbar:
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
            progress_write(f"[PolyHaven] {category}: selected {len(chosen)} / {len(assets)} HDRIs")
            for asset_id, meta in chosen:
                seen_ids.add(asset_id)
                selected.append({"id": asset_id, "category": category, "name": meta.get("name", asset_id), "meta": meta})

    manifest_entries: list[str] = []
    with progress_bar(selected, total=len(selected), desc="PolyHaven downloads", unit="hdri") as pbar:
        for i, item in enumerate(pbar, 1):
            asset_id = item["id"]
            pbar.set_postfix(category=item["category"], asset=asset_id[:24])
            try:
                files = asset_files(asset_id, args.user_agent)
                file_meta = files["hdri"][args.resolution][args.format]
            except KeyError as exc:
                progress_write(f"[PolyHaven] Skipping {asset_id}: missing {args.resolution} {args.format}")
                continue

            dest = out_dir / item["category"] / f"{asset_id}_{args.resolution}.{args.format}"
            item["file"] = {
                "path": repo_relative_or_abs(dest),
                "url": file_meta["url"],
                "size": file_meta.get("size"),
                "md5": file_meta.get("md5"),
                "resolution": args.resolution,
                "format": args.format,
            }
            manifest_entries.append(repo_relative_or_abs(dest))

            progress_write(f"[PolyHaven] {i:03d}/{len(selected):03d} {item['category']} {asset_id}")
            if not args.dry_run:
                try:
                    download_file(file_meta["url"], dest, file_meta.get("md5"), args.overwrite, args.user_agent)
                except urllib.error.URLError as exc:
                    raise RuntimeError(f"Failed to download {asset_id}: {exc}") from exc
                time.sleep(0.1)

    if not args.dry_run:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")
        progress_write(f"[PolyHaven] Wrote manifest: {manifest_path}")
        progress_write(f"[PolyHaven] Wrote metadata: {metadata_path}")
    else:
        progress_write("[PolyHaven] Dry run; no files or manifest were downloaded/written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
