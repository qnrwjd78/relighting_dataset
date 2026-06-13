from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_progress import progress_bar, progress_write


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "configs").exists() and (path / "tokenlight_dataset").exists():
            return path
    return Path(__file__).resolve().parents[2]


ROOT = find_repo_root()
DEFAULT_QUERIES = [
    "apartment interior",
    "attic room",
    "auditorium",
    "back alley",
    "balcony",
    "bar interior",
    "basement",
    "interior room",
    "living room",
    "bedroom",
    "kitchen",
    "office room",
    "dining room",
    "bathroom",
    "studio room",
    "outdoor scene",
    "forest scene",
    "garden scene",
    "street scene",
    "urban scene",
    "warehouse",
    "workshop",
    "cafe interior",
    "restaurant interior",
    "classroom",
    "library room",
    "gallery interior",
    "bathhouse",
    "boutique interior",
    "break room",
    "cabin interior",
    "camp site",
    "castle interior",
    "cathedral interior",
    "cave scene",
    "city street",
    "cityscape",
    "conference room",
    "construction site",
    "control room",
    "courtyard",
    "cyberpunk street",
    "desert scene",
    "dungeon",
    "exhibition hall",
    "factory floor",
    "farm scene",
    "film set",
    "garage",
    "greenhouse",
    "gym interior",
    "hangar",
    "hospital room",
    "hotel lobby",
    "industrial bay",
    "industrial room",
    "laboratory",
    "laundry room",
    "loft interior",
    "market street",
    "medieval hall",
    "metro station",
    "modern house",
    "museum interior",
    "night street",
    "old house",
    "parking garage",
    "playground",
    "plaza",
    "post apocalyptic city",
    "railway station",
    "reception room",
    "residential street",
    "rooftop",
    "school hallway",
    "sci fi corridor",
    "sci fi interior",
    "server room",
    "shop interior",
    "shopping mall",
    "spa interior",
    "spaceship interior",
    "subway station",
    "temple interior",
    "theater interior",
    "train interior",
    "tunnel",
    "village street",
    "waiting room",
    "wood cabin",
    "zen garden",
    "abandoned building",
    "abandoned room",
    "ancient ruins",
    "arcade room",
    "art studio",
    "bank interior",
    "beach scene",
    "bookstore",
    "bridge scene",
    "bus stop",
    "casino interior",
    "church interior",
    "corridor",
    "dock scene",
    "elevator lobby",
    "fantasy environment",
    "fireplace room",
    "futuristic city",
    "game environment",
    "gas station",
    "harbor",
    "ice cave",
    "island scene",
    "japanese room",
    "jungle scene",
    "kids room",
    "lounge interior",
    "medical lab",
    "mountain scene",
    "office lobby",
    "palace interior",
    "park scene",
    "public square",
    "science fiction scene",
    "storage room",
    "swimming pool",
    "terrace",
    "vintage room",
    "western street",
    "winter scene",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly harvest BlenderKit scenes across many queries.")
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--asset-type", choices=["scene", "model"], default="scene")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument("--preview-dir", default="outputs/previews/blenderkit")
    parser.add_argument("--out-dir", default="data/blenderkit")
    parser.add_argument("--manifest-out", default="outputs/previews/blenderkit/downloads.jsonl")
    parser.add_argument("--search-out", default="outputs/previews/blenderkit/search_results.json")
    parser.add_argument("--download-paid", action="store_true")
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--preview-engine", choices=["current", "eevee", "cycles", "workbench"], default="cycles")
    parser.add_argument("--preview-samples", type=int, default=32)
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=704)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--show-subprocess-output", action="store_true")
    parser.add_argument("--batch-log", default="outputs/previews/blenderkit/blenderkit_batches.log")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_queries(args: argparse.Namespace) -> list[str]:
    queries = list(args.queries)
    if args.queries_file:
        path = resolve_repo_path(args.queries_file)
        queries = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return list(dict.fromkeys(queries))


def current_count(index_json: str) -> int:
    path = resolve_repo_path(index_json)
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return int(data.get("count", len(data.get("items", []))))
    return len(data)


def run_batch(query: str, args: argparse.Namespace) -> int:
    before = current_count(args.index_json)
    remaining = max(args.target_count - before, 0)
    if remaining <= 0:
        return 0
    cmd = [
        sys.executable,
        str(ROOT / "dataset" / "utils" / "util_search_download_blenderkit.py"),
        "--query",
        query,
        "--asset-type",
        args.asset_type,
        "--max-results",
        str(remaining),
        "--page-size",
        str(args.page_size),
        "--out-dir",
        args.out_dir,
        "--manifest-out",
        args.manifest_out,
        "--search-out",
        args.search_out,
        "--preview-and-delete",
        "--skip-existing-index",
        "--index-json",
        args.index_json,
        "--preview-dir",
        args.preview_dir,
        "--blender-cmd",
        args.blender_cmd,
        "--preview-engine",
        args.preview_engine,
        "--preview-samples",
        str(args.preview_samples),
        "--preview-width",
        str(args.preview_width),
        "--preview-height",
        str(args.preview_height),
    ]
    if args.download_paid:
        cmd.append("--download-paid")
    if args.free_only:
        cmd.append("--free-only")
    command_text = " ".join(shlex.quote(part) for part in cmd)
    progress_write(f"[RandomHarvest] query={query!r} count={before}/{args.target_count} page_size={args.page_size}")
    progress_write(f"[RandomHarvest] cmd: {command_text}")
    if args.show_subprocess_output:
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        batch_log = resolve_repo_path(args.batch_log)
        batch_log.parent.mkdir(parents=True, exist_ok=True)
        with batch_log.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n$ {command_text}\n")
            log.flush()
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    after = current_count(args.index_json)
    return after - before


def main() -> int:
    args = parse_args()
    queries = read_queries(args)
    if not queries:
        raise SystemExit("No queries provided.")

    rng = random.Random(args.seed)
    no_progress_rounds = 0
    initial_count = min(current_count(args.index_json), args.target_count)
    with progress_bar(total=args.target_count, initial=initial_count, desc="BlenderKit previews", unit="scene") as pbar:
        for round_index in range(1, args.max_rounds + 1):
            count = current_count(args.index_json)
            if count >= args.target_count:
                progress_write(f"[RandomHarvest] Done: {count}/{args.target_count}")
                return 0
            query = rng.choice(queries)
            pbar.set_postfix(query=query, no_progress=no_progress_rounds, count=count)
            try:
                added = run_batch(query, args)
            except subprocess.CalledProcessError as exc:
                progress_write(f"[RandomHarvest] Batch failed for query={query!r}: {exc}")
                added = 0
            if added > 0:
                pbar.update(added)
                no_progress_rounds = 0
            else:
                no_progress_rounds += 1
            pbar.set_postfix(query=query, added=added, no_progress=no_progress_rounds, count=current_count(args.index_json))
            if no_progress_rounds >= len(queries) * 3:
                progress_write("[RandomHarvest] Stopping after repeated no-progress rounds.")
                return 1
            if args.sleep > 0:
                import time

                time.sleep(args.sleep)

    print(f"[RandomHarvest] Reached max rounds with {current_count(args.index_json)}/{args.target_count}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
