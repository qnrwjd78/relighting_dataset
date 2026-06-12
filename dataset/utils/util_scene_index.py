from __future__ import annotations

import json
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


def load_index(path: str | Path) -> list[dict]:
    path = resolve_repo_path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("items", []))
    return list(data)


def write_index(path: str | Path, items: list[dict]) -> None:
    path = resolve_repo_path(path)
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


def existing_source_keys(items: list[dict], source: str | None = None) -> set[str]:
    keys = set()
    for item in items:
        if source is not None and item.get("source") != source:
            continue
        key = item.get("source_key")
        if key:
            keys.add(str(key))
        record = item.get("record")
        if isinstance(record, dict) and record.get("source_key"):
            keys.add(str(record["source_key"]))
    return keys

