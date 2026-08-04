from __future__ import annotations

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_domain_blenderkit import build_preview_parser, preview_main


DEFAULTS = {
    "source_name": "blenderkit_outdoor",
    "out_dir": "data/outdoor/blenderkit",
    "manifest_out": "outputs/previews/outdoor_blenderkit/downloads.jsonl",
    "scan_manifest": "outputs/previews/outdoor_blenderkit/scanned_blends.jsonl",
    "index_json": "outputs/previews/outdoor_blenderkit/outdoor_blenderkit_index.json",
    "preview_dir": "outputs/previews/outdoor_blenderkit",
    "preview_log": "logs/downloads/outdoor_blenderkit_preview.log",
}


def main() -> int:
    parser = build_preview_parser("Preview downloaded BlenderKit outdoor scenes.", DEFAULTS)
    return preview_main(parser.parse_args(), DEFAULTS)


if __name__ == "__main__":
    raise SystemExit(main())
