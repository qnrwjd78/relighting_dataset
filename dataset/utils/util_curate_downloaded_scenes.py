from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "configs").exists() and (path / "tokenlight_dataset").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT = find_repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render review PNG/metadata for downloaded .blend files, append sequential JSON entries, then delete .blend files."
    )
    parser.add_argument("--download-manifest", required=True)
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument("--preview-dir", default="outputs/previews/blenderkit")
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["current", "eevee", "cycles", "workbench"], default="current")
    parser.add_argument("--hdri-manifest", default=None)
    parser.add_argument("--hdri-strength", type=float, default=1.0)
    parser.add_argument("--hdri-seed", type=int, default=0)
    parser.add_argument("--keep-blend", action="store_true")
    return parser.parse_args()


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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


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
        "schema": "background_scene_review_index_v1",
        "count": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def next_id(items: list[dict]) -> int:
    values = []
    for item in items:
        try:
            values.append(int(str(item.get("id", "0"))))
        except ValueError:
            pass
    return max(values, default=0) + 1


def download_link(record: dict) -> str | None:
    for key in ("download_api_url", "url", "resolved_url"):
        if record.get(key):
            return record[key]
    return None


def run_preview(
    blender_cmd: str,
    blend_path: Path,
    preview_path: Path,
    metadata_path: Path,
    width: int,
    height: int,
    samples: int,
    engine: str,
    hdri_manifest: str | None,
    hdri_strength: float,
    hdri_seed: int,
) -> None:
    script_path = ROOT / "dataset" / "utils" / "util_render_background_preview.py"
    cmd = shlex.split(blender_cmd) + [
        "-b",
        "--python",
        str(script_path),
        "--",
        "--blend",
        str(blend_path),
        "--preview",
        str(preview_path),
        "--metadata",
        str(metadata_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--samples",
        str(samples),
        "--engine",
        engine,
    ]
    if hdri_manifest:
        cmd.extend(["--hdri-manifest", hdri_manifest])
        cmd.extend(["--hdri-strength", str(hdri_strength)])
        cmd.extend(["--hdri-seed", str(hdri_seed)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def cleanup_blend(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    parent = path.parent
    while parent != ROOT and parent.exists():
        try:
            next(parent.iterdir())
            break
        except StopIteration:
            parent.rmdir()
            parent = parent.parent


def curate_records(args: argparse.Namespace) -> list[dict]:
    manifest = resolve_repo_path(args.download_manifest)
    index_json = resolve_repo_path(args.index_json)
    preview_dir = resolve_repo_path(args.preview_dir)
    records = load_jsonl(manifest)
    items = load_index(index_json)
    current_id = next_id(items)

    for record in records:
        for blend_value in record.get("blend_paths", []):
            blend_path = resolve_repo_path(blend_value)
            if not blend_path.exists():
                print(f"[Curate] Missing blend, skip: {blend_path}")
                continue
            item_id = f"{current_id:05d}"
            current_id += 1
            dataset_name = preview_dir.name
            preview_path = preview_dir / "img" / f"{dataset_name}_{item_id}.png"
            metadata_path = preview_dir / "metadata" / f"{dataset_name}_{item_id}.json"
            print(f"[Curate] {item_id} render preview for {blend_path}")
            run_preview(
                args.blender_cmd,
                blend_path,
                preview_path,
                metadata_path,
                args.width,
                args.height,
                args.samples,
                args.engine,
                args.hdri_manifest,
                args.hdri_strength,
                args.hdri_seed,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            item = {
                "id": item_id,
                "source": record.get("source"),
                "name": record.get("name"),
                "license": record.get("license"),
                "asset_type": record.get("asset_type") or record.get("format"),
                "asset_id": record.get("asset_id"),
                "asset_base_id": record.get("asset_base_id"),
                "download_link": download_link(record),
                "download_api_url": record.get("download_api_url"),
                "resolved_url": record.get("resolved_url"),
                "preview_png": repo_relative(preview_path),
                "metadata_json": repo_relative(metadata_path),
                "original_blend_path": repo_relative(blend_path),
                "blend_deleted": not args.keep_blend,
                "camera": metadata.get("camera"),
                "mesh_count": metadata.get("mesh_count"),
                "lights": metadata.get("lights", []),
                "bbox_min": metadata.get("bbox_min"),
                "bbox_max": metadata.get("bbox_max"),
                "hdri": metadata.get("hdri"),
                "hdri_strength": metadata.get("hdri_strength"),
                "subject_candidates": metadata.get("subject_candidates", []),
                "record": record,
            }
            items.append(item)
            write_index(index_json, items)
            if not args.keep_blend:
                cleanup_blend(blend_path)
                print(f"[Curate] Deleted blend: {blend_path}")

    return items


def main() -> int:
    args = parse_args()
    items = curate_records(args)
    print(f"[Curate] Index items: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
