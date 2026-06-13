from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "previews" / "polyhaven_textures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create thumbnail previews for downloaded Poly Haven texture albedo maps.")
    parser.add_argument("--metadata", default=str(DEFAULT_PREVIEW_ROOT / "polyhaven_textures.json"))
    parser.add_argument("--out-dir", default=str(DEFAULT_PREVIEW_ROOT / "img"))
    parser.add_argument("--metadata-dir", default=str(DEFAULT_PREVIEW_ROOT / "metadata"))
    parser.add_argument("--index-out", default=str(DEFAULT_PREVIEW_ROOT / "polyhaven_textures_index.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def main() -> int:
    args = parse_args()
    metadata_path = resolve_repo_path(args.metadata)
    out_dir = resolve_repo_path(args.out_dir)
    item_metadata_dir = resolve_repo_path(args.metadata_dir)
    index_out = resolve_repo_path(args.index_out)
    if not metadata_path.exists():
        raise SystemExit(f"Texture metadata does not exist: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = payload.get("textures", payload if isinstance(payload, list) else [])
    if args.limit is not None:
        entries = entries[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    item_metadata_dir.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for index, entry in enumerate(entries, 1):
        item_id = f"polyhaven_texture_{index:06d}_{entry.get('id', 'texture')}"
        albedo = entry.get("maps", {}).get("albedo", {}).get("path")
        source = resolve_repo_path(albedo) if albedo else None
        preview = out_dir / f"{item_id}.png"
        meta = item_metadata_dir / f"{item_id}.json"
        status = "ok"
        error = None
        print(f"[PolyHavenTexturePreview] {index}/{len(entries)} {source}")
        if source is None or not source.exists():
            status = "failed"
            error = "missing albedo map"
        elif not preview.exists() or args.overwrite:
            try:
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    image.thumbnail((args.size, args.size), Image.Resampling.LANCZOS)
                    image.save(preview)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        payload = {"id": item_id, "dataset": "polyhaven_textures", "source_path": str(source), "preview": str(preview), "status": status, "error": error}
        meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        items.append({"id": item_id, "metadata": str(meta), "preview": str(preview), "source_path": str(source), "status": status})

    index_out.write_text(json.dumps({"dataset": "polyhaven_textures", "items": items}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PolyHavenTexturePreview] wrote index: {index_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
