from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USER_AGENT = "relighting-dataset-blenderkit-classified-renderer/0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render selected BlenderKit preview classifications by re-downloading each .blend, "
            "rendering fixture components, and deleting the .blend after each scene."
        )
    )
    parser.add_argument("--classification", default="outputs/previews/blenderkit/blenderkit_scene_use_classification.txt")
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["single_scene_light_good", "background_good_for_portrait_or_object"],
        help="Classification section names to render.",
    )
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full.json")
    parser.add_argument("--output", default="outputs/blenderkit_classified_scenes")
    parser.add_argument("--work-dir", default="outputs/work/blenderkit_classified")
    parser.add_argument("--download-dir", default="data/blenderkit_render_cache")
    parser.add_argument("--preview-dir", default="outputs/previews/blenderkit_classified_light")
    parser.add_argument("--no-copy-previews", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("BLENDERKIT_API_KEY", ""))
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default="png")
    parser.add_argument("--hdri-mode", choices=["on", "off", "random"], default="on")
    parser.add_argument("--overwrite-blend", action="store_true")
    parser.add_argument("--keep-blend", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenes with an existing meta.json in output.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
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


def resolve_download_url(download_url: str, api_key: str, user_agent: str, scene_uuid: str) -> str:
    sep = "&" if "?" in download_url else "?"
    data = request_json(f"{download_url}{sep}scene_uuid={scene_uuid}", api_key, user_agent)
    for key in ("filePath", "file_path", "url", "downloadUrl", "download_url"):
        if data.get(key):
            return str(data[key])
    files = data.get("files")
    if isinstance(files, list):
        for item in files:
            for key in ("filePath", "url", "downloadUrl"):
                if item.get(key):
                    return str(item[key])
    raise RuntimeError(f"Download response did not contain a file URL: {data}")


def download_file(url: str, path: Path, api_key: str, user_agent: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"[BlenderKitClassified] Exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers=headers(api_key, user_agent))
    with urllib.request.urlopen(req, timeout=240) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)
    print(f"[BlenderKitClassified] Downloaded: {path}")


def section_slug(line: str) -> str | None:
    match = re.match(r"\[\d+\]\s+([^\s]+)", line.strip())
    return match.group(1) if match else None


def selected_ids(classification: Path, categories: set[str]) -> list[str]:
    ids: list[str] = []
    active = False
    for raw in classification.read_text(encoding="utf-8").splitlines():
        slug = section_slug(raw)
        if slug:
            active = slug in categories
            continue
        if not active:
            continue
        match = re.match(r"- blenderkit_(\d+)", raw)
        if match:
            ids.append(match.group(1).zfill(5))
    return ids


def safe_blend_name(item: dict) -> str:
    original = item.get("original_blend_path")
    if original:
        return Path(original).name
    base = item.get("asset_base_id") or item.get("asset_id") or item.get("id")
    return f"{base}.blend"


def load_selected_items(args: argparse.Namespace) -> list[dict]:
    ids = selected_ids(resolve_repo_path(args.classification), set(args.categories))
    index = json.loads(resolve_repo_path(args.index_json).read_text(encoding="utf-8"))
    by_id = {str(item.get("id")).zfill(5): item for item in index.get("items", [])}
    items = [by_id[item_id] for item_id in ids if item_id in by_id]
    return items[args.start : args.start + args.limit if args.limit is not None else None]


def write_runtime_config(base_config: Path, fixture_manifest: Path, output: str, work_dir: Path) -> Path:
    config = json.loads(base_config.read_text(encoding="utf-8"))
    config["output_root"] = output
    config["fixture_scene_manifest"] = repo_relative(fixture_manifest)
    config["scene_count"] = 0
    config.setdefault("fixtures", {})
    config["fixtures"]["enabled"] = True
    config["fixtures"]["render_if_manifest_exists"] = True
    runtime_config = work_dir / "runtime_config.json"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding="utf-8")
    return runtime_config


def write_one_scene_manifest(item: dict, blend_path: Path, work_dir: Path) -> Path:
    scene_id = f"blenderkit_{str(item.get('id')).zfill(5)}"
    row = {
        "scene_id": scene_id,
        "blend_path": repo_relative(blend_path),
        "camera": item.get("camera"),
        "fixtures": [
            {
                "id": "all",
                "prefixes": [""],
                "light_prefixes": [""],
            }
        ],
    }
    manifest = work_dir / "fixture_scene_one.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return manifest


def render_one(args: argparse.Namespace, item: dict, blend_path: Path, work_dir: Path) -> None:
    manifest = write_one_scene_manifest(item, blend_path, work_dir)
    runtime_config = write_runtime_config(resolve_repo_path(args.config), manifest, args.output, work_dir)
    cmd = shlex.split(args.blender_cmd) + [
        "-b",
        "--python",
        str(ROOT / "scripts" / "render_scene_relighting.py"),
        "--",
        "--config",
        repo_relative(runtime_config),
        "--output",
        args.output,
        "--only",
        "fixtures",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
        "--component-format",
        args.component_format,
        "--hdri-mode",
        args.hdri_mode,
    ]
    print("[BlenderKitClassified] Render:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def append_preview_index(preview_dir: Path, row: dict) -> None:
    index_path = preview_dir / "index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        items = data.get("items", [])
    else:
        items = []
    items = [item for item in items if item.get("scene_id") != row.get("scene_id")]
    items.append(row)
    payload = {"schema": "blenderkit_classified_light_preview_v1", "count": len(items), "items": items}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def copy_light_previews(args: argparse.Namespace, item: dict) -> None:
    if args.no_copy_previews or args.component_format == "exr":
        return
    item_id = str(item.get("id")).zfill(5)
    scene_id = f"blenderkit_{item_id}"
    scene_dir = resolve_repo_path(args.output) / "scenes" / scene_id
    preview_dir = resolve_repo_path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    sources = {
        "environment": scene_dir / "fixtures" / "environment.png",
        "contribution": scene_dir / "fixtures" / "fixture_all" / "contribution.png",
        "mask": scene_dir / "fixtures" / "fixture_all" / "mask.png",
    }
    for name, src in sources.items():
        if not src.exists():
            continue
        dst = preview_dir / f"{scene_id}_{name}.png"
        shutil.copyfile(src, dst)
        copied[name] = repo_relative(dst)

    if copied:
        append_preview_index(
            preview_dir,
            {
                "scene_id": scene_id,
                "name": item.get("name"),
                "preview_png": item.get("preview_png"),
                "metadata_json": item.get("metadata_json"),
                **copied,
            },
        )
        print(f"[BlenderKitClassified] Preview copied: {scene_id} -> {preview_dir}")


def main() -> int:
    args = parse_args()
    api_key = load_api_key(args)
    if not api_key:
        raise SystemExit("Missing BlenderKit API key. Set BLENDERKIT_API_KEY or pass --api-key-file.")

    items = load_selected_items(args)
    if not items:
        raise SystemExit("No selected BlenderKit items matched the requested categories.")

    download_dir = resolve_repo_path(args.download_dir)
    work_dir = resolve_repo_path(args.work_dir)
    output_root = resolve_repo_path(args.output)
    scene_uuid = str(uuid.uuid4())

    print(f"[BlenderKitClassified] Selected items: {len(items)}")
    for index, item in enumerate(items, 1):
        item_id = str(item.get("id")).zfill(5)
        scene_id = f"blenderkit_{item_id}"
        meta_path = output_root / "scenes" / scene_id / "meta.json"
        if args.skip_existing and meta_path.exists():
            print(f"[BlenderKitClassified] Skip existing {scene_id}: {meta_path}")
            continue

        download_api_url = item.get("download_api_url") or item.get("record", {}).get("download_api_url")
        if not download_api_url:
            print(f"[BlenderKitClassified] Skip missing download_api_url: {scene_id}")
            continue

        blend_path = download_dir / safe_blend_name(item)
        print(f"[BlenderKitClassified] {index}/{len(items)} {scene_id} {item.get('name')}")
        try:
            resolved_url = resolve_download_url(download_api_url, api_key, args.user_agent, scene_uuid)
            download_file(resolved_url, blend_path, api_key, args.user_agent, args.overwrite_blend)
            render_one(args, item, blend_path, work_dir)
            copy_light_previews(args, item)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[BlenderKitClassified] FAILED {scene_id}: {exc}", file=sys.stderr)
            continue
        finally:
            if not args.keep_blend:
                blend_path.unlink(missing_ok=True)
                part = blend_path.with_suffix(blend_path.suffix + ".part")
                part.unlink(missing_ok=True)
                print(f"[BlenderKitClassified] Deleted blend: {blend_path}")
        time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
