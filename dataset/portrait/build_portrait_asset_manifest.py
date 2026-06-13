from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge accepted portrait asset manifests into one renderer manifest.")
    parser.add_argument("--inputs", nargs="+", required=True, help="accepted.txt/object manifest files to merge.")
    parser.add_argument("--out", default="outputs/previews/portrait_assets/portrait_assets_objects.txt")
    parser.add_argument("--metadata-out", default="outputs/previews/portrait_assets/portrait_assets_manifest.json")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def repo_relative_or_abs(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(repo_relative_or_abs(resolve_repo_path(line)))
    return lines


def main() -> int:
    args = parse_args()
    inputs = [resolve_repo_path(path) for path in args.inputs]
    out = resolve_repo_path(args.out)
    metadata_out = resolve_repo_path(args.metadata_out)

    merged: list[str] = []
    sources = []
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"Input manifest does not exist: {path}")
        lines = read_lines(path)
        sources.append({"path": str(path), "count": len(lines)})
        merged.extend(lines)
    merged = list(dict.fromkeys(merged))

    out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    metadata_out.write_text(
        json.dumps({"schema": "portrait_asset_manifest_v1", "count": len(merged), "sources": sources, "items": merged}, indent=2),
        encoding="utf-8",
    )
    print(f"[PortraitManifest] wrote {len(merged)} asset(s): {out}")
    print(f"[PortraitManifest] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
