from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_EXTS = {".glb", ".gltf", ".obj", ".ply", ".blend", ".scene_instance.json", ".stage_config.json"}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a manually downloaded HSSD/Habitat scene folder.")
    parser.add_argument("--root", default="data/indoor/hssd")
    parser.add_argument("--manifest", default="outputs/previews/hssd_indoor/hssd_assets.txt")
    parser.add_argument("--metadata-out", default="outputs/previews/hssd_indoor/hssd_index.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def supported(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in SCENE_EXTS or name.endswith(".scene_instance.json") or name.endswith(".stage_config.json")


def main() -> int:
    args = parse_args()
    root = resolve_repo_path(args.root)
    if not root.exists():
        raise SystemExit(f"HSSD root does not exist: {root}")
    assets = sorted(path for path in root.rglob("*") if path.is_file() and supported(path))
    if args.limit is not None:
        assets = assets[: args.limit]
    manifest = resolve_repo_path(args.manifest)
    metadata_out = resolve_repo_path(args.metadata_out)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in assets) + ("\n" if assets else ""), encoding="utf-8")
    metadata_out.write_text(
        json.dumps({"dataset": "hssd_indoor", "root": str(root), "count": len(assets), "manifest": str(manifest)}, indent=2),
        encoding="utf-8",
    )
    print(f"[HSSD] assets: {len(assets)}")
    print(f"[HSSD] wrote manifest: {manifest}")
    print(f"[HSSD] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
