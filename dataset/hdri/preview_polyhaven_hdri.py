from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "polyhaven_hdri"
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "previews" / "polyhaven_hdri"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tone-mapped PNG previews for Poly Haven HDRIs.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_PREVIEW_ROOT / "polyhaven_hdri_hdris.txt"))
    parser.add_argument("--out-dir", default=str(DEFAULT_PREVIEW_ROOT / "img"))
    parser.add_argument("--metadata-dir", default=str(DEFAULT_PREVIEW_ROOT / "metadata"))
    parser.add_argument("--index-out", default=str(DEFAULT_PREVIEW_ROOT / "polyhaven_hdri_index.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def collect_hdris(root: Path, manifest: Path) -> list[Path]:
    if manifest.exists():
        items = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(resolve_repo_path(line))
        return items
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in {".hdr", ".exr"})


def tonemap(image, width: int):
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing numpy/Pillow. Install requirements.txt to preview HDR/EXR files.") from exc
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr[..., :3]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    arr = arr / (1.0 + arr)
    arr = np.power(arr, 1.0 / 2.2)
    arr8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    result = Image.fromarray(arr8)
    if width > 0 and result.width != width:
        height = max(1, round(result.height * width / result.width))
        result = result.resize((width, height), Image.Resampling.LANCZOS)
    return result


def safe_id(index: int, path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in path.stem)
    return f"polyhaven_hdri_{index:06d}_{stem}"


def main() -> int:
    args = parse_args()
    root = resolve_repo_path(args.root)
    manifest = resolve_repo_path(args.manifest)
    out_dir = resolve_repo_path(args.out_dir)
    metadata_dir = resolve_repo_path(args.metadata_dir)
    index_out = resolve_repo_path(args.index_out)
    hdris = collect_hdris(root, manifest)
    if args.limit is not None:
        hdris = hdris[: args.limit]
    if not hdris:
        raise SystemExit(f"No HDRI files found under {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for index, hdri in enumerate(hdris, 1):
        item_id = safe_id(index, hdri)
        preview = out_dir / f"{item_id}.png"
        metadata = metadata_dir / f"{item_id}.json"
        status = "ok"
        error = None
        print(f"[PolyHavenHDRIPreview] {index}/{len(hdris)} {hdri}")
        if not preview.exists() or args.overwrite:
            try:
                try:
                    import imageio.v3 as iio
                except ModuleNotFoundError as exc:
                    raise RuntimeError("Missing imageio. Install requirements.txt to preview HDR/EXR files.") from exc
                image = iio.imread(hdri)
                tonemap(image, args.width).save(preview)
            except Exception as exc:
                status = "failed"
                error = str(exc)
                print(f"[PolyHavenHDRIPreview] Failed: {hdri}: {exc}")
        payload = {
            "id": item_id,
            "dataset": "polyhaven_hdri",
            "hdri": str(hdri),
            "preview": str(preview),
            "status": status,
            "error": error,
        }
        metadata.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        items.append({"id": item_id, "metadata": str(metadata), "preview": str(preview), "source_path": str(hdri), "status": status})

    index_out.write_text(json.dumps({"dataset": "polyhaven_hdri", "items": items}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PolyHavenHDRIPreview] wrote index: {index_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
