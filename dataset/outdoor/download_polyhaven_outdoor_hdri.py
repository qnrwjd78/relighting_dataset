from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_run_script import ROOT, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download outdoor/urban/nature Poly Haven HDRIs for outdoor relighting scenes.")
    parser.add_argument("--categories", nargs="+", default=["outdoor", "urban", "nature"])
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--resolution", default="2k")
    parser.add_argument("--format", choices=["hdr", "exr"], default="hdr")
    parser.add_argument("--sort", choices=["random", "downloads", "name"], default="random")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "dataset" / "hdri" / "download_polyhaven_hdri.py"),
        "--categories",
        *args.categories,
        "--per-category",
        str(args.per_category),
        "--resolution",
        args.resolution,
        "--format",
        args.format,
        "--sort",
        args.sort,
        "--seed",
        str(args.seed),
        "--out-dir",
        "data/outdoor/polyhaven_hdri",
        "--manifest",
        "outputs/previews/outdoor_polyhaven_hdri/outdoor_polyhaven_hdri_hdris.txt",
        "--metadata-out",
        "outputs/previews/outdoor_polyhaven_hdri/outdoor_polyhaven_hdri_index.json",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
