from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_run_script import ROOT, run


QUERIES = [
    "street scene",
    "city street",
    "urban scene",
    "outdoor scene",
    "park scene",
    "garden scene",
    "forest scene",
    "courtyard",
    "plaza",
    "night street",
    "village street",
    "harbor scene",
    "gas station",
    "neon street",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search/download Sketchfab outdoor scene candidates.")
    parser.add_argument("--queries", nargs="+", default=QUERIES)
    parser.add_argument("--licenses", default="cc0,by")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--extract", action="store_true", default=True)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed", action="store_true", default=True)
    parser.add_argument("--token-file", default="sketchfab_token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "dataset" / "portrait" / "download_sketchfab_human.py"),
        "--queries",
        *args.queries,
        "--licenses",
        args.licenses,
        "--max-results",
        str(args.max_results),
        "--page-size",
        str(args.page_size),
        "--out-dir",
        "data/outdoor/sketchfab",
        "--search-out",
        "outputs/previews/sketchfab_outdoor/search_results.json",
        "--manifest",
        "outputs/previews/sketchfab_outdoor/sketchfab_outdoor_objects.txt",
        "--metadata-out",
        "outputs/previews/sketchfab_outdoor/sketchfab_outdoor_download_meta.json",
        "--token-file",
        args.token_file,
    ]
    if args.extract:
        cmd.append("--extract")
    if args.extract_only:
        cmd.append("--extract-only")
    if args.delete_zip_after_extract:
        cmd.append("--delete-zip-after-extract")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_failed:
        cmd.append("--skip-failed")
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
