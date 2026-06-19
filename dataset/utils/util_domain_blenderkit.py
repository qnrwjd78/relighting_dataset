from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "configs").exists() and (path / "tokenlight_dataset").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def repo_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_queries(path: str | Path | None, defaults: list[str]) -> list[str]:
    if path is None:
        return list(dict.fromkeys(defaults))
    query_path = resolve_repo_path(path)
    return list(
        dict.fromkeys(
            line.strip()
            for line in query_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )


def run_command(cmd: list[str], show_output: bool, log_path: str | Path | None = None) -> None:
    text = " ".join(shlex.quote(part) for part in cmd)
    print(f"[DomainBlenderKit] cmd: {text}")
    if show_output:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    path = resolve_repo_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n$ {text}\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def build_download_parser(description: str, defaults: dict, queries: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--query", default=None, help="Single BlenderKit query. If omitted, all domain queries are searched.")
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--max-results", type=int, default=defaults["max_results"], help="Maximum downloads per query.")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--asset-type", choices=["scene", "model"], default="scene")
    parser.add_argument("--out-dir", default=defaults["out_dir"])
    parser.add_argument("--manifest-out", default=defaults["manifest_out"])
    parser.add_argument("--search-out", default=defaults["search_out"])
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default="blenderkit_key.txt")
    parser.add_argument("--free-only", dest="free_only", action="store_true", help="Download only free BlenderKit assets.")
    parser.add_argument("--include-paid", dest="free_only", action="store_false", help="Search paid assets too; use with --download-paid.")
    parser.add_argument("--download-paid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing-index", action="store_true")
    parser.add_argument("--index-json", default=defaults["index_json"])
    parser.add_argument("--show-subprocess-output", action="store_true")
    parser.add_argument("--batch-log", default=defaults["batch_log"])
    parser.set_defaults(domain_queries=queries, free_only=True)
    return parser


def download_main(args: argparse.Namespace) -> int:
    queries = [args.query] if args.query else read_queries(args.queries_file, args.domain_queries)
    if not queries:
        raise SystemExit("No BlenderKit queries were provided.")

    manifest = resolve_repo_path(args.manifest_out)
    search_out = resolve_repo_path(args.search_out)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    search_out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and manifest.exists():
        manifest.unlink()
    manifest.write_text("", encoding="utf-8")
    all_search_rows: list[dict] = []

    for query_index, query in enumerate(queries, 1):
        query_manifest = manifest.parent / f".{manifest.stem}.query_{query_index:03d}.jsonl"
        query_search = search_out.parent / f".{search_out.stem}.query_{query_index:03d}.json"
        cmd = [
            sys.executable,
            str(ROOT / "dataset" / "utils" / "util_search_download_blenderkit.py"),
            "--query",
            query,
            "--asset-type",
            args.asset_type,
            "--max-results",
            str(args.max_results),
            "--page-size",
            str(args.page_size),
            "--out-dir",
            args.out_dir,
            "--manifest-out",
            str(query_manifest),
            "--search-out",
            str(query_search),
            "--api-key-file",
            args.api_key_file,
            "--index-json",
            args.index_json,
        ]
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.free_only and not args.download_paid:
            cmd.append("--free-only")
        if args.download_paid:
            cmd.append("--download-paid")
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.skip_existing_index:
            cmd.append("--skip-existing-index")
        run_command(cmd, args.show_subprocess_output, args.batch_log)
        if query_manifest.exists():
            text = query_manifest.read_text(encoding="utf-8")
            if text:
                with manifest.open("a", encoding="utf-8") as out:
                    out.write(text)
                    if not text.endswith("\n"):
                        out.write("\n")
            query_manifest.unlink()
        if query_search.exists():
            try:
                rows = json.loads(query_search.read_text(encoding="utf-8"))
                if isinstance(rows, list):
                    all_search_rows.extend(rows)
            finally:
                query_search.unlink()
    search_out.write_text(json.dumps(all_search_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DomainBlenderKit] Wrote combined manifest: {manifest}")
    print(f"[DomainBlenderKit] Wrote combined search results: {search_out}")
    return 0


def build_preview_parser(description: str, defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", default=defaults["out_dir"], help="Folder to scan for .blend files when no manifest is given.")
    parser.add_argument("--download-manifest", default=defaults["manifest_out"])
    parser.add_argument("--index-json", default=defaults["index_json"])
    parser.add_argument("--preview-dir", default=defaults["preview_dir"])
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["current", "eevee", "cycles", "workbench"], default="cycles")
    parser.add_argument("--hdri-manifest", default=None)
    parser.add_argument("--hdri-strength", type=float, default=1.0)
    parser.add_argument("--hdri-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-scan-manifest", action="store_true")
    parser.add_argument("--delete-blend-after-preview", action="store_true")
    parser.add_argument("--show-subprocess-output", action="store_true")
    parser.add_argument("--batch-log", default=defaults["preview_log"])
    return parser


def scan_blends(root: Path, limit: int | None = None) -> list[Path]:
    blends = sorted(path for path in root.rglob("*.blend") if path.is_file())
    return blends[:limit] if limit is not None else blends


def write_scan_manifest(args: argparse.Namespace, defaults: dict) -> Path:
    root = resolve_repo_path(args.root)
    blends = scan_blends(root, args.limit)
    if not blends:
        raise SystemExit(f"No .blend files found under {root}")

    manifest = resolve_repo_path(defaults["scan_manifest"])
    if manifest.exists() and not args.overwrite_scan_manifest:
        return manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for blend in blends:
        rows.append(
            {
                "source": defaults["source_name"],
                "license": None,
                "asset_type": "scene",
                "asset_id": None,
                "asset_base_id": None,
                "name": blend.stem,
                "download_path": repo_relative(blend),
                "blend_paths": [repo_relative(blend)],
                "source_key": repo_relative(blend),
            }
        )
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"[DomainBlenderKit] Wrote scanned manifest: {manifest} ({len(rows)} blend files)")
    return manifest


def preview_main(args: argparse.Namespace, defaults: dict) -> int:
    manifest = resolve_repo_path(args.download_manifest)
    if not manifest.exists():
        manifest = write_scan_manifest(args, defaults)

    cmd = [
        sys.executable,
        str(ROOT / "dataset" / "utils" / "util_curate_downloaded_scenes.py"),
        "--download-manifest",
        str(manifest),
        "--index-json",
        args.index_json,
        "--preview-dir",
        args.preview_dir,
        "--blender-cmd",
        args.blender_cmd,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
        "--engine",
        args.engine,
    ]
    if args.hdri_manifest:
        cmd.extend(["--hdri-manifest", args.hdri_manifest])
        cmd.extend(["--hdri-strength", str(args.hdri_strength)])
        cmd.extend(["--hdri-seed", str(args.hdri_seed)])
    if not args.delete_blend_after_preview:
        cmd.append("--keep-blend")
    run_command(cmd, args.show_subprocess_output, args.batch_log)
    return 0
