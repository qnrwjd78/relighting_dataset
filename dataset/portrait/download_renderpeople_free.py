from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from utils.util_progress import progress_bar, progress_write


PAGE_URL = "https://renderpeople.com/free-3d-people/"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "renderpeople_free"
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "previews" / "renderpeople_free"
USER_AGENT = "Mozilla/5.0 RenderPeopleFreeDownloader/1.0"
SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".ply", ".stl"}
FORMAT_LABELS = {
    "3DM": "Rhino",
    "Alembic": "Alembic",
    "BLD": "Blender",
    "C4D": "Cinema 4D",
    "FBX": "FBX",
    "GLB": "GLB",
    "MAX": "3ds Max",
    "MAYA": "Maya",
    "OBJ": "OBJ",
    "PSD": "Photoshop",
    "SKP": "SketchUp",
    "U3D": "Unity",
    "UE": "Unreal Engine",
    "UE4": "Unreal Engine 4",
    "USD": "USD",
}
KNOWN_FORMAT_CODES = sorted(FORMAT_LABELS, key=len, reverse=True)
PREVIEW_FORMAT_ORDER = ["BLD", "OBJ", "FBX", "GLB", "Alembic", "USD", "C4D", "MAYA", "MAX", "3DM", "SKP", "U3D", "UE", "UE4", "PSD"]


@dataclass(frozen=True)
class Package:
    group: str
    format_code: str
    format_label: str
    url: str
    filename: str
    size_bytes: int | None = None


class ZipLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        href = attr_map.get("href")
        if href and ".zip" in href.lower():
            self.links.append(urljoin(self.base_url, html.unescape(href)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official free RenderPeople zip packages.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Output root. Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--page-url", default=PAGE_URL, help=f"Free models page. Default: {PAGE_URL}")
    parser.add_argument(
        "--formats",
        default="preview",
        help=(
            "Comma-separated formats: preview, all, Blender/BLD, OBJ, FBX, GLB, USD, Alembic, "
            "MAX, MAYA, C4D, SKP, 3DM, U3D, UE, UE4, PSD. Default: preview."
        ),
    )
    parser.add_argument("--group", action="append", help="Only download package groups containing this text. Can be repeated.")
    parser.add_argument("--url", action="append", help="Download only an exact discovered zip URL. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected packages after filtering.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected packages and sizes without downloading.")
    parser.add_argument("--extract", action="store_true", help="Extract each zip after download.")
    parser.add_argument("--extract-only", action="store_true", help="Extract existing selected zips from out-dir/zips; skip download.")
    parser.add_argument(
        "--delete-zip-after-extract",
        "--remove-zip-after-extract",
        dest="delete_zip_after_extract",
        action="store_true",
        help="Delete each zip only after successful extraction.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed zips and extracted folders.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume .part files.")
    parser.add_argument("--skip-failed", action="store_true", help="Continue with the next package after a failure.")
    parser.add_argument("--skip-size-check", action="store_true", help="Skip HTTP HEAD size checks.")
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_PREVIEW_ROOT / "renderpeople_free_objects.txt"),
        help="Manifest path written after extraction.",
    )
    parser.add_argument(
        "--metadata-out",
        default=str(DEFAULT_PREVIEW_ROOT / "renderpeople_free_download_meta.json"),
        help="Download metadata JSON path.",
    )
    parser.set_defaults(resume=True)
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


def safe_stem(value: str) -> str:
    stem = Path(urlparse(value).path).stem or Path(value).stem
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)


def classify_package(url: str) -> Package:
    filename = Path(urlparse(url).path).name
    stem = Path(filename).stem
    for code in KNOWN_FORMAT_CODES:
        suffix = f"_{code}"
        if stem.endswith(suffix):
            return Package(
                group=stem[: -len(suffix)],
                format_code=code,
                format_label=FORMAT_LABELS[code],
                url=url,
                filename=filename,
            )
    return Package(group=stem, format_code="BUNDLE", format_label="Bundle", url=url, filename=filename)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def request_with_retries(session: requests.Session, method: str, url: str, retries: int, timeout: float, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to {method.upper()} {url}: {last_error}")


def discover_packages(session: requests.Session, page_url: str, timeout: float, retries: int) -> list[Package]:
    response = request_with_retries(session, "GET", page_url, retries, timeout)
    parser = ZipLinkParser(page_url)
    parser.feed(response.text)
    urls = sorted(dict.fromkeys(parser.links), key=lambda item: (safe_stem(item), item))
    return [classify_package(url) for url in urls]


def normalize_format_token(token: str) -> str:
    key = token.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "all": "all",
        "preview": "preview",
        "blend": "BLD",
        "blender": "BLD",
        "bld": "BLD",
        "obj": "OBJ",
        "fbx": "FBX",
        "glb": "GLB",
        "usd": "USD",
        "alembic": "Alembic",
        "abc": "Alembic",
        "max": "MAX",
        "3dsmax": "MAX",
        "maya": "MAYA",
        "c4d": "C4D",
        "cinema4d": "C4D",
        "skp": "SKP",
        "sketchup": "SKP",
        "rhino": "3DM",
        "3dm": "3DM",
        "unity": "U3D",
        "u3d": "U3D",
        "unreal": "UE",
        "ue": "UE",
        "ue4": "UE4",
        "photoshop": "PSD",
        "psd": "PSD",
        "bundle": "BUNDLE",
    }
    if key not in aliases:
        raise SystemExit(f"Unknown RenderPeople format token: {token}")
    return aliases[key]


def selected_packages(packages: list[Package], args: argparse.Namespace) -> list[Package]:
    selected = packages
    if args.url:
        wanted = set(args.url)
        selected = [package for package in selected if package.url in wanted]
    if args.group:
        needles = [item.lower() for item in args.group]
        selected = [package for package in selected if any(needle in package.group.lower() for needle in needles)]

    format_tokens = [normalize_format_token(token) for token in args.formats.split(",") if token.strip()]
    if not format_tokens:
        format_tokens = ["preview"]
    if "all" not in format_tokens:
        if "preview" in format_tokens:
            selected = best_preview_packages(selected)
        else:
            wanted_formats = set(format_tokens)
            selected = [package for package in selected if package.format_code in wanted_formats]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def best_preview_packages(packages: list[Package]) -> list[Package]:
    groups: dict[str, list[Package]] = {}
    for package in packages:
        groups.setdefault(package.group, []).append(package)

    def score(package: Package) -> tuple[int, str]:
        try:
            rank = PREVIEW_FORMAT_ORDER.index(package.format_code)
        except ValueError:
            rank = len(PREVIEW_FORMAT_ORDER)
        return (rank, package.filename)

    return sorted((sorted(items, key=score)[0] for items in groups.values()), key=lambda item: (item.group, item.filename))


def size_package(session: requests.Session, package: Package, timeout: float, retries: int) -> Package:
    try:
        response = request_with_retries(session, "HEAD", package.url, retries, timeout, allow_redirects=True)
    except Exception:
        return package
    value = response.headers.get("content-length")
    if value and value.isdigit():
        return Package(**{**package.__dict__, "size_bytes": int(value)})
    return package


def with_sizes(session: requests.Session, packages: list[Package], timeout: float, retries: int, skip: bool) -> list[Package]:
    if skip:
        return packages
    sized = []
    with progress_bar(packages, total=len(packages), desc="RenderPeople size", unit="file") as pbar:
        for package in pbar:
            pbar.set_postfix(file=package.filename[:32])
            sized.append(size_package(session, package, timeout, retries))
    return sized


def download_package(
    session: requests.Session,
    package: Package,
    zip_dir: Path,
    *,
    resume: bool,
    overwrite: bool,
    chunk_size: int,
    timeout: float,
    retries: int,
) -> Path:
    target = zip_dir / package.filename
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() and not overwrite:
        if package.size_bytes is None or target.stat().st_size == package.size_bytes:
            progress_write(f"[RenderPeople] exists: {target}")
            return target
    if overwrite:
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    elif part.exists() and not resume:
        part.unlink()

    offset = part.stat().st_size if part.exists() and resume else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(package.url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
            response.raise_for_status()
            if offset and response.status_code != 206:
                offset = 0
                part.unlink(missing_ok=True)
            mode = "ab" if offset else "wb"
            total = package.size_bytes
            if total is None:
                content_length = response.headers.get("content-length")
                total = int(content_length) + offset if content_length and content_length.isdigit() else None
            with part.open(mode) as handle:
                with progress_bar(
                    total=total,
                    initial=offset,
                    desc=package.filename,
                    unit="B",
                    leave=False,
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                            pbar.update(len(chunk))
            part.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to download {package.url}: {last_error}")


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
    if not root.exists():
        return []
    grouped: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            grouped.setdefault(path.parent, []).append(path)
    return sorted((sorted(paths, key=candidate_score)[0] for paths in grouped.values()), key=str)


def write_outputs(metadata: list[dict], extract_root: Path, manifest: Path, metadata_out: Path) -> None:
    assets = find_asset_files(extract_root)
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


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    zip_dir = out_dir / "zips"
    extract_root = out_dir / "extracted"
    manifest = Path(args.manifest).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    chunk_size = args.chunk_size_mb * 1024 * 1024

    session = make_session()
    progress_write(f"[RenderPeople] Discover: {args.page_url}")
    discovered = discover_packages(session, args.page_url, args.timeout, args.retries)
    packages = selected_packages(discovered, args)
    packages = with_sizes(session, packages, args.timeout, args.retries, args.skip_size_check)
    if not packages:
        raise SystemExit("No RenderPeople packages matched the selected filters.")

    total = sum(package.size_bytes or 0 for package in packages)
    unknown = sum(1 for package in packages if package.size_bytes is None)
    progress_write(f"[RenderPeople] Selected: {len(packages)} package(s)")
    for package in packages:
        progress_write(f"  - {package.filename}  {format_bytes(package.size_bytes)}  {package.group}  {package.format_label}")
    total_text = format_bytes(total) if unknown == 0 else f"{format_bytes(total)} known + {unknown} unknown"
    progress_write(f"[RenderPeople] Total download size: {total_text}")
    progress_write(f"[RenderPeople] Output: {out_dir}")
    if args.extract or args.extract_only:
        progress_write(f"[RenderPeople] Extract dir: {extract_root}")
    if args.delete_zip_after_extract:
        progress_write("[RenderPeople] Zip cleanup: delete each zip after successful extraction")
    if args.dry_run:
        return 0

    metadata: list[dict] = []
    with progress_bar(packages, total=len(packages), desc="RenderPeople packages", unit="file") as pbar:
        for index, package in enumerate(pbar):
            pbar.set_postfix(file=package.filename[:32])
            zip_path = zip_dir / package.filename
            try:
                if not args.extract_only:
                    progress_write(f"[RenderPeople] Download {index + 1}/{len(packages)}: {package.filename}")
                    zip_path = download_package(
                        session,
                        package,
                        zip_dir,
                        resume=args.resume,
                        overwrite=args.overwrite,
                        chunk_size=chunk_size,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                if args.extract or args.extract_only:
                    if not zip_path.exists():
                        raise FileNotFoundError(f"Zip not found for extraction: {zip_path}")
                    extract_dir = extract_root / safe_stem(package.url)
                    progress_write(f"[RenderPeople] Extract: {zip_path.name} -> {extract_dir}")
                    extract_zip(zip_path, extract_dir, args.overwrite)
                    if args.delete_zip_after_extract:
                        zip_path.unlink(missing_ok=True)
                        progress_write(f"[RenderPeople] Deleted zip: {zip_path}")
                else:
                    extract_dir = None
                metadata.append(
                    {
                        "group": package.group,
                        "format": package.format_label,
                        "format_code": package.format_code,
                        "url": package.url,
                        "size_bytes": package.size_bytes,
                        "zip_path": str(zip_path),
                        "extract_dir": str(extract_dir) if extract_dir else None,
                    }
                )
            except Exception as exc:
                if not args.skip_failed:
                    raise
                progress_write(f"[RenderPeople] Failed: {package.filename}: {exc}")
                metadata.append({"url": package.url, "filename": package.filename, "status": "failed", "error": str(exc)})

    if args.extract or args.extract_only:
        write_outputs(metadata, extract_root, manifest, metadata_out)
    else:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps({"packages": metadata}, indent=2, ensure_ascii=False), encoding="utf-8")
        progress_write(f"[RenderPeople] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
