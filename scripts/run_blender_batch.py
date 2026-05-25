from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TokenLight Blender component renderer.")
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full.json")
    parser.add_argument("--blender-exe", default=os.environ.get("BLENDER_EXE", "blender"))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--only", choices=["all", "spatial", "diffuse", "fixtures"], default="all")
    parser.add_argument("--background", action="store_true", help="Pass -b to Blender. Enabled by default.")
    parser.set_defaults(background=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    config = config.resolve()
    if not config.exists():
        raise FileNotFoundError(config)

    cmd = [args.blender_exe]
    if args.background:
        cmd.append("-b")
    cmd += [
        "--python",
        str(root / "scripts" / "render_components.py"),
        "--",
        "--config",
        str(config),
        "--start-index",
        str(args.start_index),
        "--only",
        args.only,
    ]
    if args.max_scenes is not None:
        cmd += ["--max-scenes", str(args.max_scenes)]
    if args.resolution is not None:
        cmd += ["--resolution", str(args.resolution)]
    if args.samples is not None:
        cmd += ["--samples", str(args.samples)]

    print("Running:", " ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd, cwd=root)
    except FileNotFoundError as exc:
        print(
            "Blender executable was not found. Install Blender, add it to PATH, "
            "set BLENDER_EXE, or pass --blender-exe.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
