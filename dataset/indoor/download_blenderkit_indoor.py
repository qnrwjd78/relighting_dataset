from __future__ import annotations

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_domain_blenderkit import build_download_parser, download_main


QUERIES = [
    "living room interior",
    "bedroom interior",
    "kitchen interior",
    "dining room interior",
    "office room interior",
    "hotel lobby interior",
    "cafe interior",
    "restaurant interior",
    "bar interior",
    "library room",
    "classroom interior",
    "gallery interior",
    "museum interior",
    "corridor interior",
    "hallway interior",
    "warehouse interior",
    "workshop interior",
    "laboratory interior",
    "studio room",
    "loft interior",
    "bathroom interior",
    "garage interior",
    "shop interior",
    "shopping mall interior",
    "theater interior",
    "church interior",
    "cathedral interior",
    "sci fi corridor",
    "spaceship interior",
    "room with lamps",
    "interior lighting scene",
]

DEFAULTS = {
    "max_results": 3,
    "out_dir": "data/indoor/blenderkit",
    "manifest_out": "outputs/previews/indoor_blenderkit/downloads.jsonl",
    "search_out": "outputs/previews/indoor_blenderkit/search_results.json",
    "index_json": "outputs/previews/indoor_blenderkit/indoor_blenderkit_index.json",
    "batch_log": "logs/downloads/indoor_blenderkit_download.log",
}


def main() -> int:
    parser = build_download_parser("Download BlenderKit indoor .blend scenes.", DEFAULTS, QUERIES)
    return download_main(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
