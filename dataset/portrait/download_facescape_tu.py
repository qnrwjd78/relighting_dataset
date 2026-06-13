from __future__ import annotations

import argparse
import http.cookiejar
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "facescape" / "tu_model" / "zips"
DEFAULT_EXTRACT_DIR = REPO_ROOT / "data" / "facescape" / "tu_model" / "extracted"
DOWNLOAD_PAGE = "https://drive.google.com/uc?export=download&id={file_id}"
USER_AGENT = "Mozilla/5.0 FaceScapeTUDownloader/1.0"


@dataclass(frozen=True)
class DriveFile:
    name: str
    file_id: str
    approx_bytes: int
    category: str


class GoogleDriveQuotaExceededError(RuntimeError):
    pass


OFFICIAL_TU_TRAINSET_FILES = [
    DriveFile("facescape_trainset_001_100.zip", "1FdBAh2LdQ7_JezC0dmCJlwxhHySfhOhW", 14_571_787_902, "trainset"),
    DriveFile("facescape_trainset_101_200.zip", "1N3864qDkRt-KReWaCIYDhdu06qKqfb9B", 14_679_110_665, "trainset"),
    DriveFile("facescape_trainset_201_300.zip", "1oa1f5XDevEr2bl45BVUS8XonBwK-pzf7", 14_731_404_929, "trainset"),
    DriveFile("facescape_trainset_301_400.zip", "1KIaknaxZ1JrbTGUWhEDVHOoEqIfpjcfp", 15_214_448_056, "trainset"),
    DriveFile("facescape_trainset_401_500.zip", "1UlpC51HR-LNu2IhqTqWxYAi_aPQSnkjn", 15_755_563_329, "trainset"),
    DriveFile("facescape_trainset_501_600.zip", "1OonKNXMRXQ_venX9Jl0C8N-Sdc3jBZAs", 15_684_580_527, "trainset"),
    DriveFile("facescape_trainset_601_700.zip", "1_ecg-nwUuWOTknRGOQx0x8eiRFdcXxNj", 15_816_193_778, "trainset"),
    DriveFile("facescape_trainset_701_800.zip", "18_uC_eiKGaRYYRE3FzelO8hQF_3QP7Xj", 15_820_433_187, "trainset"),
    DriveFile("facescape_trainset_801_847.zip", "1DUaoX4iuTkzykBnDhWPQE_xh0R4cKJc7", 7_471_565_205, "trainset"),
]

OFFICIAL_TU_EXTRA_FILES = [
    DriveFile("publishable_nomasaic_tex.zip", "1DEV9eJp3oRwa4jK0ASfX_S6cm5wQDhZp", 439_749_686, "extra"),
]

COPIED_TU_TRAINSET_FILES = [
    DriveFile("facescape_trainset_001_100.zip", "1OHLfRUbXHDjgK74Uw_nN1t_msQhcvj2l", 14_571_787_902, "trainset"),
    DriveFile("facescape_trainset_101_200.zip", "1r-kkVPLYXY0MIlyi2WYji-3mCThbqaM-", 14_679_110_665, "trainset"),
    DriveFile("facescape_trainset_201_300.zip", "1Zax3-fujFM4F32Gx_39g_WSIml4M2lP_", 14_731_404_929, "trainset"),
    DriveFile("facescape_trainset_301_400.zip", "1NQiDqT_L2N2haWSv5njS6HOF77zK9W3D", 15_214_448_056, "trainset"),
    DriveFile("facescape_trainset_401_500.zip", "1w-M7Fd3NLGsF217ZDkD90kImn0PBf7MX", 15_755_563_329, "trainset"),
    DriveFile("facescape_trainset_501_600.zip", "1KUwUMc7eRFhhAsqfjXg-0ldB_Dvv1F-N", 15_684_580_527, "trainset"),
    DriveFile("facescape_trainset_601_700.zip", "1_Z66efbw9HaRJPLPUOUQ3ldmVUnX9It6", 15_816_193_778, "trainset"),
    DriveFile("facescape_trainset_701_800.zip", "177_-fj4WbImt5bl1-kKWKxh3PXfeZ8jE", 15_820_433_187, "trainset"),
    DriveFile("facescape_trainset_801_847.zip", "1UM_1kRHGL1nP5_3yhXxvKjnoNOH2gKOn", 7_471_565_205, "trainset"),
]

COPIED_TU_EXTRA_FILES = [
    DriveFile("publishable_nomasaic_tex.zip", "13CEdtUBRdPixrhOLwRyTrI-oavY6czpP", 439_749_686, "extra"),
]


class GoogleDriveWarningParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_download_form = False
        self.action: str | None = None
        self.params: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value for key, value in attrs}
        if tag == "form" and attr_map.get("id") == "download-form":
            self.in_download_form = True
            self.action = attr_map.get("action")
            return
        if self.in_download_form and tag == "input":
            name = attr_map.get("name")
            value = attr_map.get("value")
            if name and value is not None:
                self.params[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.in_download_form = False


def parse_args() -> argparse.Namespace:
    all_file_names = [item.name for item in OFFICIAL_TU_TRAINSET_FILES + OFFICIAL_TU_EXTRA_FILES]
    parser = argparse.ArgumentParser(description="Download the FaceScape TU-Model zip files from Google Drive.")
    parser.add_argument(
        "--source",
        choices=("copied", "official"),
        default="copied",
        help="Google Drive file ID set to use. Default: copied.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Download directory. Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--trainset-only", action="store_true", help="Skip publishable_nomasaic_tex.zip.")
    parser.add_argument("--file", action="append", choices=all_file_names, help="Download only the named file. Can be repeated.")
    parser.add_argument("--start-at", choices=all_file_names, help="Start from this file in the selected file list.")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without downloading.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed zip files.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume .part files.")
    parser.add_argument("--extract", action="store_true", help="Extract each selected zip after download.")
    parser.add_argument("--extract-only", action="store_true", help="Extract selected zip files already present in --out-dir; skip download.")
    parser.add_argument("--extract-dir", default=str(DEFAULT_EXTRACT_DIR), help=f"Extraction directory. Default: {DEFAULT_EXTRACT_DIR}")
    parser.add_argument(
        "--delete-zip-after-extract",
        "--remove-zip-after-extract",
        dest="delete_zip_after_extract",
        action="store_true",
        help="Delete each zip only after it is extracted successfully.",
    )
    parser.add_argument("--skip-space-check", action="store_true", help="Skip the free disk space check.")
    parser.add_argument("--skip-failed", action="store_true", help="Continue with the next file when a download or extraction fails.")
    parser.add_argument("--chunk-size-mb", type=int, default=8, help="Download chunk size in MiB.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.delete_zip_after_extract and not (args.extract or args.extract_only):
        parser.error("--delete-zip-after-extract requires --extract or --extract-only")
    return args


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000.0 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1000.0
    return f"{size:.1f}TB"


def selected_files(args: argparse.Namespace) -> list[DriveFile]:
    if args.source == "official":
        trainset_files = OFFICIAL_TU_TRAINSET_FILES
        extra_files = OFFICIAL_TU_EXTRA_FILES
    else:
        trainset_files = COPIED_TU_TRAINSET_FILES
        extra_files = COPIED_TU_EXTRA_FILES

    files = list(trainset_files)
    if not args.trainset_only:
        files.extend(extra_files)
    if args.start_at:
        try:
            start_index = next(index for index, item in enumerate(files) if item.name == args.start_at)
        except StopIteration as exc:
            raise SystemExit(f"--start-at file is not in the selected file list: {args.start_at}") from exc
        files = files[start_index:]
    if args.file:
        wanted = set(args.file)
        files = [item for item in files if item.name in wanted]
    return files


def existing_parent(path: Path) -> Path:
    parent = path
    while not parent.exists():
        parent = parent.parent
    return parent


def remaining_estimate(files: list[DriveFile], out_dir: Path, overwrite: bool) -> int:
    total = 0
    for item in files:
        target = out_dir / item.name
        part = target.with_suffix(target.suffix + ".part")
        if target.exists() and not overwrite:
            continue
        already = part.stat().st_size if part.exists() and not overwrite else 0
        total += max(item.approx_bytes - already, 0)
    return total


def check_disk_space(files: list[DriveFile], out_dir: Path, overwrite: bool) -> None:
    parent = existing_parent(out_dir)
    free = shutil.disk_usage(parent).free
    needed = remaining_estimate(files, out_dir, overwrite)
    if free < needed:
        raise SystemExit(
            f"Not enough free space under {parent}: need about {format_bytes(needed)}, "
            f"available {format_bytes(free)}"
        )
    print(f"[FaceScapeTU] Free space OK: need about {format_bytes(needed)}, available {format_bytes(free)}")


def make_opener() -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def read_warning_page(opener: urllib.request.OpenerDirector, item: DriveFile, timeout: int) -> str:
    url = DOWNLOAD_PAGE.format(file_id=urllib.parse.quote(item.file_id))
    with opener.open(url, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", "replace")


def build_download_url(opener: urllib.request.OpenerDirector, item: DriveFile, timeout: int) -> str:
    page = read_warning_page(opener, item, timeout)
    parser = GoogleDriveWarningParser()
    parser.feed(page)
    if parser.action and parser.params:
        return parser.action + "?" + urllib.parse.urlencode(parser.params)

    confirm_match = re.search(r'name="confirm" value="([^"]+)"', page)
    params = {"id": item.file_id, "export": "download"}
    if confirm_match:
        params["confirm"] = confirm_match.group(1)
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(params)


def strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def html_error_message(body: str) -> str:
    text = strip_html(body)
    if "Quota exceeded" in text or "can't view or download this file at this time" in text:
        raise GoogleDriveQuotaExceededError(
            "Google Drive quota exceeded for this file. Try again later, use an authenticated "
            "Google Drive copy/download flow, or request a fresh official FaceScape share link."
        )
    return text[:500]


def download_one(
    opener: urllib.request.OpenerDirector,
    item: DriveFile,
    out_dir: Path,
    *,
    resume: bool,
    overwrite: bool,
    chunk_size: int,
    timeout: int,
) -> None:
    target = out_dir / item.name
    part = target.with_suffix(target.suffix + ".part")

    if target.exists() and not overwrite:
        print(f"[FaceScapeTU] Skip existing: {target}")
        return
    if overwrite:
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    elif part.exists() and not resume:
        part.unlink()

    offset = part.stat().st_size if part.exists() and resume else 0
    url = build_download_url(opener, item, timeout)
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    print(f"[FaceScapeTU] Download: {item.name} ({format_bytes(item.approx_bytes)} approx)")
    if offset:
        print(f"[FaceScapeTU] Resume from {format_bytes(offset)}")

    with opener.open(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            body = response.read(20000).decode("utf-8", "replace")
            raise RuntimeError(f"Google Drive returned an HTML page instead of data: {html_error_message(body)}")

        status = getattr(response, "status", 200)
        if offset and status != 206:
            print("[FaceScapeTU] Server ignored Range header; restarting this file.")
            offset = 0
            mode = "wb"
        else:
            mode = "ab" if offset else "wb"

        content_length = response.headers.get("Content-Length")
        remaining = int(content_length) if content_length and content_length.isdigit() else None
        expected_total = offset + remaining if remaining is not None else item.approx_bytes
        downloaded = offset
        last_report = time.monotonic()

        with part.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    percent = min(downloaded / expected_total * 100.0, 100.0) if expected_total else 0.0
                    print(f"[FaceScapeTU]   {format_bytes(downloaded)} / {format_bytes(expected_total)} ({percent:.1f}%)")
                    last_report = now

    part.rename(target)
    print(f"[FaceScapeTU] Done: {target}")


def zip_uncompressed_size(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        return sum(info.file_size for info in archive.infolist() if not info.is_dir())


def safe_zip_target(extract_dir: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    member = Path(normalized)
    if member.is_absolute() or ".." in member.parts:
        raise RuntimeError(f"Unsafe zip member path: {member_name}")
    target = (extract_dir / member).resolve()
    root = extract_dir.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Unsafe zip member path: {member_name}") from exc
    return target


def extract_one(
    zip_path: Path,
    extract_dir: Path,
    *,
    delete_zip_after_extract: bool,
    check_space: bool,
    chunk_size: int,
) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid zip file: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    total_size = zip_uncompressed_size(zip_path)
    if check_space:
        free = shutil.disk_usage(existing_parent(extract_dir)).free
        if free < total_size:
            raise RuntimeError(
                f"Not enough free space to extract {zip_path.name}: need about {format_bytes(total_size)}, "
                f"available {format_bytes(free)}"
            )

    print(f"[FaceScapeTU] Extract: {zip_path.name} -> {extract_dir} ({format_bytes(total_size)} uncompressed)")
    extracted = 0
    last_report = time.monotonic()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = safe_zip_target(extract_dir, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    extracted += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        percent = min(extracted / total_size * 100.0, 100.0) if total_size else 100.0
                        print(f"[FaceScapeTU]   {format_bytes(extracted)} / {format_bytes(total_size)} ({percent:.1f}%)")
                        last_report = now

    print(f"[FaceScapeTU] Extracted: {zip_path.name}")
    if delete_zip_after_extract:
        zip_path.unlink()
        print(f"[FaceScapeTU] Deleted zip: {zip_path}")


def print_plan(files: list[DriveFile], out_dir: Path, include_existing: bool = True) -> None:
    print(f"[FaceScapeTU] Output: {out_dir}")
    print("[FaceScapeTU] Files:")
    for item in files:
        suffix = ""
        if include_existing and (out_dir / item.name).exists():
            suffix = " [exists]"
        print(f"  - {item.name}  {format_bytes(item.approx_bytes)}  {item.category}{suffix}")
    print(f"[FaceScapeTU] Approx total: {format_bytes(sum(item.approx_bytes for item in files))}")


def main() -> int:
    args = parse_args()
    files = selected_files(args)
    if not files:
        raise SystemExit("No files selected.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    print(f"[FaceScapeTU] Source: {args.source}")
    print_plan(files, out_dir)
    if args.extract or args.extract_only:
        print(f"[FaceScapeTU] Extract dir: {extract_dir}")
    if args.delete_zip_after_extract:
        print("[FaceScapeTU] Zip cleanup: delete each zip after successful extraction")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.extract_only and not args.skip_space_check:
        space_check_files = files
        if args.extract and args.delete_zip_after_extract:
            space_check_files = [max(files, key=lambda item: item.approx_bytes)]
        check_disk_space(space_check_files, out_dir, args.overwrite)

    chunk_size = max(args.chunk_size_mb, 1) * 1024 * 1024
    failed: list[tuple[str, str]] = []
    if not args.extract_only:
        opener = make_opener()
        for item in files:
            try:
                download_one(
                    opener,
                    item,
                    out_dir,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    chunk_size=chunk_size,
                    timeout=args.timeout,
                )
                if args.extract:
                    extract_one(
                        out_dir / item.name,
                        extract_dir,
                        delete_zip_after_extract=args.delete_zip_after_extract,
                        check_space=not args.skip_space_check,
                        chunk_size=chunk_size,
                    )
            except KeyboardInterrupt:
                print("\n[FaceScapeTU] Interrupted. Partial .part file is kept for resume; completed zips are kept unless extraction already deleted them.", file=sys.stderr)
                return 130
            except Exception as exc:
                print(f"[FaceScapeTU] Failed: {item.name}: {exc}", file=sys.stderr)
                failed.append((item.name, str(exc)))
                if not args.skip_failed:
                    return 1

    if args.extract_only:
        for item in files:
            try:
                extract_one(
                    out_dir / item.name,
                    extract_dir,
                    delete_zip_after_extract=args.delete_zip_after_extract,
                    check_space=not args.skip_space_check,
                    chunk_size=chunk_size,
                )
            except KeyboardInterrupt:
                print("\n[FaceScapeTU] Interrupted during extraction. Zip file is kept.", file=sys.stderr)
                return 130
            except Exception as exc:
                print(f"[FaceScapeTU] Failed to extract {item.name}: {exc}", file=sys.stderr)
                failed.append((item.name, str(exc)))
                if not args.skip_failed:
                    return 1

    if failed:
        print("[FaceScapeTU] Failed files:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1

    if args.extract_only:
        print("[FaceScapeTU] All selected FaceScape TU files are extracted.")
    elif args.extract:
        print("[FaceScapeTU] All selected FaceScape TU files are downloaded and extracted.")
    else:
        print("[FaceScapeTU] All selected FaceScape TU files are downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
