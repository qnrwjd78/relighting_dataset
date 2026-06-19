from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a manually downloaded 3D-FRONT/3D-FUTURE folder and write a JSON-scene manifest."
    )
    parser.add_argument("--root", default="data/indoor/3dfront")
    parser.add_argument("--front-dir", default=None, help="Defaults to <root>/3D-FRONT.")
    parser.add_argument("--future-dir", default=None, help="Defaults to <root>/3D-FUTURE-model.")
    parser.add_argument("--texture-dir", default=None, help="Defaults to <root>/3D-FRONT-texture.")
    parser.add_argument("--manifest", default="outputs/previews/3dfront_indoor/3dfront_scene_jsons.txt")
    parser.add_argument("--metadata-out", default="outputs/previews/3dfront_indoor/3dfront_index.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_repo_path(args.root)
    front_dir = resolve_repo_path(args.front_dir) if args.front_dir else root / "3D-FRONT"
    future_dir = resolve_repo_path(args.future_dir) if args.future_dir else root / "3D-FUTURE-model"
    texture_dir = resolve_repo_path(args.texture_dir) if args.texture_dir else root / "3D-FRONT-texture"
    missing = [path for path in (front_dir, future_dir, texture_dir) if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing required 3D-FRONT folders:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nDownload 3D-FRONT, 3D-FUTURE-model, and 3D-FRONT-texture from the official 3D-FRONT release first."
        )

    jsons = sorted(front_dir.rglob("*.json"))
    if args.limit is not None:
        jsons = jsons[: args.limit]
    manifest = resolve_repo_path(args.manifest)
    metadata_out = resolve_repo_path(args.metadata_out)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in jsons) + ("\n" if jsons else ""), encoding="utf-8")
    metadata = {
        "dataset": "3dfront_indoor",
        "scene_count": len(jsons),
        "front_dir": str(front_dir),
        "future_dir": str(future_dir),
        "texture_dir": str(texture_dir),
        "manifest": str(manifest),
        "blenderproc_loader_hint": "Use blenderproc.loader.load_front3d(json_path, future_model_path, front_3D_texture_path, ...).",
    }
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[3D-FRONT] scenes: {len(jsons)}")
    print(f"[3D-FRONT] wrote manifest: {manifest}")
    print(f"[3D-FRONT] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
