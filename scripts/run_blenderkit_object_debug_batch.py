from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_AGENT = "relighting-dataset-blenderkit-object-debug/0.1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_classified_blenderkit_spatial import (  # noqa: E402
    download_file,
    resolve_download_url,
    safe_blend_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download BlenderKit object candidates one at a time and render three debug artifacts: "
            "candidate object JSON, source-camera colored preview, and Objaverse-like light-cube preview."
        )
    )
    parser.add_argument("--classification-csv", default="outputs/previews/blenderkit/blenderkit_preview_3way_candidates.csv")
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument("--category", default="object_candidate")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional ids like blenderkit_00059 or 00059.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-root", default="outputs/previews/blenderkit/object_debug")
    parser.add_argument("--download-dir", default="outputs/work/blenderkit_object_debug/blends")
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default="blenderkit_key.txt")
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--overwrite-blend", action="store_true")
    parser.add_argument("--overwrite-renders", action="store_true")
    parser.add_argument("--keep-blend", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--make-contact-sheets", action="store_true", default=True)
    parser.add_argument("--no-contact-sheets", action="store_false", dest="make_contact_sheets")
    parser.add_argument("--sheet-cols", type=int, default=3)
    parser.add_argument("--sheet-rows", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=270)
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def normalize_id(value: str) -> str:
    value = value.strip()
    if value.startswith("blenderkit_"):
        value = value[len("blenderkit_") :]
    return value.zfill(5)


def scene_id(item_id: str) -> str:
    return f"blenderkit_{normalize_id(item_id)}"


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        path = resolve_repo_path(args.api_key_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def load_classified_ids(path: Path, category: str) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("category") == category:
                ids.append(normalize_id(row["id"]))
    return ids


def load_items(args: argparse.Namespace) -> list[dict]:
    index = json.loads(resolve_repo_path(args.index_json).read_text(encoding="utf-8"))
    by_id = {normalize_id(str(item.get("id"))): item for item in index.get("items", [])}
    if args.ids:
        ids = [normalize_id(value) for value in args.ids]
    else:
        ids = load_classified_ids(resolve_repo_path(args.classification_csv), args.category)
    items = [by_id[item_id] for item_id in ids if item_id in by_id]
    end = args.start + args.limit if args.limit is not None else None
    return items[args.start : end]


def run_command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def blender_debug_command(args: argparse.Namespace, blend_path: Path, item: dict, out_dir: Path) -> list[str]:
    item_id = normalize_id(str(item.get("id")))
    metadata_json = item.get("metadata_json") or f"outputs/previews/blenderkit/metadata/blenderkit_{item_id}.json"
    return shlex.split(args.blender_cmd) + [
        "--background",
        str(blend_path),
        "--python",
        str(ROOT / "scripts" / "blenderkit_object_debug.py"),
        "--",
        "--scene-id",
        scene_id(item_id),
        "--metadata-json",
        str(resolve_repo_path(metadata_json)),
        "--out-dir",
        str(out_dir),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
    ]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_letterbox(path: Path, size: tuple[int, int], bg: str = "white") -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_comparison(scene: str, out_dir: Path, preview_png: str | None, panel_size: tuple[int, int]) -> Path | None:
    original = resolve_repo_path(preview_png) if preview_png else None
    colored = out_dir / "preview_render_scene_000000.png"
    objaverse = out_dir / "objaverse_like_render_000000.png"
    if not original or not original.exists() or not colored.exists() or not objaverse.exists():
        return None

    labels = ["original", "candidate", "objaverse+cube"]
    paths = [original, colored, objaverse]
    label_h = 34
    font = load_font(20)
    panels = []
    for path, label in zip(paths, labels):
        panel = Image.new("RGB", (panel_size[0], panel_size[1] + label_h), "white")
        panel.paste(resize_letterbox(path, panel_size), (0, 0))
        draw = ImageDraw.Draw(panel)
        draw.text((8, panel_size[1] + 6), label, fill="black", font=font)
        panels.append(panel)
    sheet = Image.new("RGB", (panel_size[0] * len(panels), panel_size[1] + label_h), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * panel_size[0], 0))
    out = out_dir / "comparison.png"
    sheet.save(out)
    return out


def make_contact_sheets(out_root: Path, rows: list[dict], args: argparse.Namespace) -> None:
    comparisons = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        path = Path(row.get("comparison", ""))
        if path.exists():
            comparisons.append((row["scene_id"], path))
    if not comparisons:
        return
    sheet_dir = out_root / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cols, sheet_rows = args.sheet_cols, args.sheet_rows
    per_sheet = cols * sheet_rows
    panel_w = args.panel_width
    panel_h = args.panel_height
    label_h = 26
    font = load_font(16)
    for sheet_index in range((len(comparisons) + per_sheet - 1) // per_sheet):
        subset = comparisons[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * panel_w, sheet_rows * (panel_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (sid, path) in enumerate(subset):
            x = (i % cols) * panel_w
            y = (i // cols) * (panel_h + label_h)
            sheet.paste(resize_letterbox(path, (panel_w, panel_h)), (x, y))
            draw.text((x + 6, y + panel_h + 4), sid, fill="black", font=font)
        sheet.save(sheet_dir / f"sheet_{sheet_index:03d}.jpg", quality=92)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene_id",
        "name",
        "status",
        "message",
        "out_dir",
        "candidate_json",
        "candidate_preview",
        "objaverse_like_render",
        "comparison",
        "blend_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def process_item(args: argparse.Namespace, item: dict, api_key: str, scene_uuid: str) -> dict:
    item_id = normalize_id(str(item.get("id")))
    sid = scene_id(item_id)
    out_root = resolve_repo_path(args.out_root)
    out_dir = out_root / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "scene_id": sid,
        "name": item.get("name", ""),
        "status": "pending",
        "message": "",
        "out_dir": out_dir.as_posix(),
        "candidate_json": (out_dir / "candidate_objects.json").as_posix(),
        "candidate_preview": (out_dir / "preview_render_scene_000000.png").as_posix(),
        "objaverse_like_render": (out_dir / "objaverse_like_render_000000.png").as_posix(),
        "comparison": (out_dir / "comparison.png").as_posix(),
        "blend_path": "",
    }
    if args.skip_existing and (out_dir / "candidate_objects.json").exists() and not args.overwrite_renders:
        comparison = make_comparison(sid, out_dir, item.get("preview_png"), (args.panel_width, args.panel_height))
        if comparison:
            result["comparison"] = comparison.as_posix()
        result["status"] = "ok"
        result["message"] = "skip_existing"
        return result

    download_api_url = item.get("download_api_url") or item.get("record", {}).get("download_api_url")
    if not download_api_url:
        result["status"] = "failed"
        result["message"] = "missing_download_api_url"
        return result

    blend_path = out_dir / "source.blend" if args.keep_blend else resolve_repo_path(args.download_dir) / safe_blend_name(item)
    result["blend_path"] = blend_path.as_posix()

    if args.dry_run:
        result["status"] = "dry_run"
        result["message"] = run_command_text(blender_debug_command(args, blend_path, item, out_dir))
        return result

    try:
        resolved_url = resolve_download_url(download_api_url, api_key, args.user_agent, scene_uuid)
        download_file(resolved_url, blend_path, api_key, args.user_agent, args.overwrite_blend)
        cmd = blender_debug_command(args, blend_path, item, out_dir)
        print(f"[ObjectDebug] Render {sid}: {run_command_text(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)
        comparison = make_comparison(sid, out_dir, item.get("preview_png"), (args.panel_width, args.panel_height))
        if comparison:
            result["comparison"] = comparison.as_posix()
        result["status"] = "ok"
        result["message"] = "rendered"
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, subprocess.CalledProcessError) as exc:
        result["status"] = "failed"
        result["message"] = str(exc)
        print(f"[ObjectDebug] FAILED {sid}: {exc}", file=sys.stderr)
    finally:
        if not args.keep_blend and not args.dry_run:
            blend_path.unlink(missing_ok=True)
            blend_path.with_suffix(blend_path.suffix + ".part").unlink(missing_ok=True)
    return result


def main() -> int:
    args = parse_args()
    api_key = load_api_key(args)
    if not api_key and not args.dry_run:
        raise SystemExit("Missing BlenderKit API key. Set BLENDERKIT_API_KEY or pass --api-key-file.")
    items = load_items(args)
    if not items:
        raise SystemExit("No items selected.")

    out_root = resolve_repo_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scene_uuid = str(uuid.uuid4())
    rows: list[dict] = []
    manifest = out_root / "batch_manifest.csv"

    print(f"[ObjectDebug] Selected items: {len(items)}")
    for index, item in enumerate(items, 1):
        sid = scene_id(str(item.get("id")))
        print(f"[ObjectDebug] {index}/{len(items)} {sid} {item.get('name', '')}")
        row = process_item(args, item, api_key, scene_uuid)
        rows.append(row)
        write_manifest(manifest, rows)
        time.sleep(args.sleep)

    if args.make_contact_sheets and not args.dry_run:
        make_contact_sheets(out_root, rows, args)
    write_manifest(manifest, rows)
    print(f"[ObjectDebug] Wrote manifest: {manifest}")
    if args.make_contact_sheets and not args.dry_run:
        print(f"[ObjectDebug] Contact sheets: {out_root / 'contact_sheets'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
