from __future__ import annotations

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_preview_assets import main


if __name__ == "__main__":
    raise SystemExit(main("hsrd100"))
