from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_progress import progress_bar, progress_write

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Objaverse-XL objects and write outputs/previews/objaverse_xl/objaverse_xl_objects.txt."
    )
    parser.add_argument("--download-dir", default="data/objaverse_xl", help="Objaverse-XL metadata/object cache dir.")
    parser.add_argument("--report-out", default="outputs/objaverse_xl_reports", help="Where source/fileType reports are saved.")
    parser.add_argument("--report-only", action="store_true", help="Only download annotations and write count reports.")
    parser.add_argument(
        "--source",
        nargs="+",
        default=["sketchfab"],
        choices=["sketchfab", "github", "thingiverse", "smithsonian"],
        help="Objaverse-XL source(s) to download from.",
    )
    parser.add_argument("--file-types", nargs="+", default=["glb"], help="File types to download, e.g. glb obj fbx stl.")
    parser.add_argument("--limit", type=int, default=100, help="Number of filtered objects to download.")
    parser.add_argument("--start", type=int, default=0, help="Start offset within the filtered annotation table.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle filtered rows before applying start/limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--refresh", action="store_true", help="Refresh Objaverse-XL annotation parquet files.")
    parser.add_argument("--license-contains", default=None, help="Optional case-insensitive license substring filter.")
    parser.add_argument(
        "--numbered-out-dir",
        default="data/objaverse_xl/objects",
        help="Folder for 000001.glb style files. Use --numbered-mode none to skip.",
    )
    parser.add_argument(
        "--numbered-mode",
        choices=["copy", "symlink", "hardlink", "none"],
        default="copy",
        help="How numbered files are created from downloaded object files.",
    )
    parser.add_argument("--numbered-start", type=int, default=1, help="First number for numbered output files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing numbered output files.")
    parser.add_argument(
        "--no-fallback-copy",
        action="store_true",
        help="Do not fall back to copying when symlink/hardlink creation fails.",
    )
    parser.add_argument(
        "--write-manifest",
        default="outputs/previews/objaverse_xl/objaverse_xl_objects.txt",
        help="Renderer manifest to write. Relative paths are written when possible.",
    )
    parser.add_argument("--append-manifest", action="store_true", help="Append to existing manifest instead of replacing it.")
    parser.add_argument(
        "--github-save-repo-format",
        choices=["files", "zip", "tar", "tar.gz"],
        default="files",
        help="Objaverse-XL GitHub source save_repo_format.",
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


def norm_ext(value: object) -> str:
    ext = str(value).strip().lower()
    return ext[1:] if ext.startswith(".") else ext


def import_dependencies():
    try:
        import objaverse.xl as oxl
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Objaverse-XL dependencies. Install them with:\n"
            "  python -m pip install -U objaverse pandas pyarrow tqdm fsspec"
        ) from exc
    return oxl, pd


def write_reports(annotations, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    source_counts = annotations["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="count")
    source_counts.to_csv(report_dir / "source_counts.csv", index=False)

    source_filetype = (
        annotations.groupby(["source", "fileType"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )
    source_filetype.to_csv(report_dir / "source_filetype_counts.csv", index=False)

    license_counts = annotations["license"].value_counts(dropna=False).rename_axis("license").reset_index(name="count")
    license_counts.to_csv(report_dir / "license_counts.csv", index=False)

    print("\nSource counts:")
    print(source_counts.to_string(index=False))
    print(f"\nReports written to: {report_dir}")


def filter_annotations(annotations, args: argparse.Namespace):
    df = annotations.copy()
    df["source_norm"] = df["source"].astype(str).str.lower()
    df["fileType_norm"] = df["fileType"].map(norm_ext)

    source_set = {s.lower() for s in args.source}
    type_set = {norm_ext(t) for t in args.file_types}

    df = df[df["source_norm"].isin(source_set)]
    df = df[df["fileType_norm"].isin(type_set)]

    if args.license_contains:
        query = args.license_contains.lower()
        df = df[df["license"].astype(str).str.lower().str.contains(query, na=False)]

    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed)

    return df.reset_index(drop=True)


def materialize(src: Path, dst: Path, mode: str, overwrite: bool, fallback_copy: bool) -> bool:
    if not src.exists():
        progress_write(f"[WARN] Missing downloaded file, skip: {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            progress_write(f"[WARN] Existing output, skip: {dst}")
            return False
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return True

    try:
        if mode == "symlink":
            os.symlink(src.resolve(), dst)
        elif mode == "hardlink":
            os.link(src.resolve(), dst)
        else:
            raise ValueError(mode)
        return True
    except OSError as exc:
        if not fallback_copy:
            raise
        progress_write(f"[WARN] {mode} failed for {dst}: {exc}. Falling back to copy.")
        shutil.copy2(src, dst)
        return True


def write_object_manifest(lines: Iterable[str], path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_lines = [line.strip() for line in lines if line and line.strip()]

    if append and path.exists():
        existing = []
        with path.open("r", encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        merged = list(dict.fromkeys(existing + new_lines))
    else:
        merged = list(dict.fromkeys(new_lines))

    with path.open("w", encoding="utf-8") as f:
        for line in merged:
            f.write(line + "\n")
    progress_write(f"Wrote renderer object manifest: {path} ({len(merged)} entries)")


def main() -> int:
    args = parse_args()
    oxl, pd = import_dependencies()

    download_dir = resolve_repo_path(args.download_dir)
    report_dir = resolve_repo_path(args.report_out)
    download_dir.mkdir(parents=True, exist_ok=True)

    progress_write(f"[1] Loading Objaverse-XL annotations into: {download_dir}")
    annotations = oxl.get_annotations(download_dir=str(download_dir), refresh=args.refresh)
    progress_write(f"[2] Total annotations: {len(annotations):,}")
    progress_write(f"[2] Columns: {list(annotations.columns)}")

    write_reports(annotations, report_dir)
    if args.report_only:
        return 0

    if args.limit <= 0:
        raise SystemExit("--limit must be positive. Use a small limit first, e.g. --limit 100.")

    filtered = filter_annotations(annotations, args)
    progress_write(f"[3] Filtered objects: {len(filtered):,}")
    if len(filtered) == 0:
        raise SystemExit("No objects matched the filters. Try different --source or --file-types.")

    selected = filtered.iloc[args.start : args.start + args.limit].copy()
    if len(selected) == 0:
        raise SystemExit("--start is beyond the filtered table.")

    selection_csv = report_dir / "selected_objects.csv"
    selected.to_csv(selection_csv, index=False)
    progress_write(f"[4] Selected {len(selected):,} objects. Metadata: {selection_csv}")

    kwargs = {}
    if "github" in {s.lower() for s in args.source}:
        kwargs["save_repo_format"] = args.github_save_repo_format

    progress_write("[5] Downloading objects...")
    downloaded = oxl.download_objects(
        objects=selected,
        download_dir=str(download_dir),
        processes=args.processes,
        **kwargs,
    )
    progress_write(f"[6] Downloaded/found objects: {len(downloaded):,}")

    manifest_records = []
    for file_identifier, local_path in downloaded.items():
        row = selected[selected["fileIdentifier"] == file_identifier]
        item = {"fileIdentifier": file_identifier, "local_path": local_path}
        if len(row) > 0:
            r = row.iloc[0]
            item.update(
                {
                    "source": str(r["source"]),
                    "fileType": str(r["fileType"]),
                    "license": str(r["license"]),
                    "sha256": str(r["sha256"]),
                }
            )
        manifest_records.append(item)

    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = report_dir / "download_manifest.json"
    manifest_csv = report_dir / "download_manifest.csv"
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)
    pd.DataFrame(manifest_records).to_csv(manifest_csv, index=False)
    progress_write(f"[7] Download manifest: {manifest_csv}")

    object_manifest_lines = []
    numbered_records = []
    if args.numbered_mode == "none":
        object_manifest_lines = [repo_relative_or_abs(item["local_path"]) for item in manifest_records]
    else:
        numbered_dir = resolve_repo_path(args.numbered_out_dir)
        fallback_copy = not args.no_fallback_copy
        out_index = args.numbered_start
        with progress_bar(manifest_records, total=len(manifest_records), desc="Objaverse materialize", unit="object") as pbar:
            for item in pbar:
                src = Path(item["local_path"]).resolve()
                pbar.set_postfix(file=src.name[:32])
                ext = src.suffix.lower() or f".{norm_ext(item.get('fileType', 'glb'))}"
                dst = numbered_dir / f"{out_index:06d}{ext}"
                if materialize(src, dst, args.numbered_mode, args.overwrite, fallback_copy):
                    rel = repo_relative_or_abs(dst)
                    object_manifest_lines.append(rel)
                    numbered = dict(item)
                    numbered["numbered_path"] = rel
                    numbered_records.append(numbered)
                    out_index += 1

        numbered_csv = numbered_dir / "numbered_manifest.csv"
        pd.DataFrame(numbered_records).to_csv(numbered_csv, index=False)
        progress_write(f"[8] Numbered objects: {len(numbered_records):,} -> {numbered_dir}")

    if args.write_manifest:
        write_object_manifest(object_manifest_lines, resolve_repo_path(args.write_manifest), args.append_manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
