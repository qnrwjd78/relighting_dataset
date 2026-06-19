from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_run_script import ROOT, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview indoor Poly Haven HDRIs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "dataset" / "hdri" / "preview_polyhaven_hdri.py"),
        "--root",
        "data/indoor/polyhaven_hdri",
        "--manifest",
        "outputs/previews/indoor_polyhaven_hdri/indoor_polyhaven_hdri_hdris.txt",
        "--out-dir",
        "outputs/previews/indoor_polyhaven_hdri/img",
        "--metadata-dir",
        "outputs/previews/indoor_polyhaven_hdri/metadata",
        "--index-out",
        "outputs/previews/indoor_polyhaven_hdri/indoor_polyhaven_hdri_index.json",
        "--width",
        str(args.width),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.overwrite:
        cmd.append("--overwrite")
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
