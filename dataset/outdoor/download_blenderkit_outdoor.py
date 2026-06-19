from __future__ import annotations

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_domain_blenderkit import build_download_parser, download_main


QUERIES = [
    "street scene",
    "city street",
    "urban scene",
    "night street",
    "residential street",
    "market street",
    "plaza scene",
    "courtyard scene",
    "park scene",
    "garden scene",
    "forest scene",
    "village street",
    "harbor scene",
    "dock scene",
    "gas station scene",
    "parking lot scene",
    "rooftop scene",
    "terrace scene",
    "playground scene",
    "camp site",
    "desert scene",
    "beach scene",
    "mountain scene",
    "jungle scene",
    "futuristic city",
    "cyberpunk street",
    "post apocalyptic city",
    "outdoor lighting scene",
    "street lights scene",
    "neon street scene",
]

DEFAULTS = {
    "max_results": 3,
    "out_dir": "data/outdoor/blenderkit",
    "manifest_out": "outputs/previews/outdoor_blenderkit/downloads.jsonl",
    "search_out": "outputs/previews/outdoor_blenderkit/search_results.json",
    "index_json": "outputs/previews/outdoor_blenderkit/outdoor_blenderkit_index.json",
    "batch_log": "logs/downloads/outdoor_blenderkit_download.log",
}


def main() -> int:
    parser = build_download_parser("Download BlenderKit outdoor .blend scenes.", DEFAULTS, QUERIES)
    return download_main(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
