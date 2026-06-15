from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Blender object relighting renderer.")
    parser.add_argument("--config", default="configs/tokenlight_synthetic_full.json")
    parser.add_argument("--blender-exe", default=os.environ.get("BLENDER_EXE", "blender"))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--component-format", choices=["exr", "png", "both"], default=None)
    parser.add_argument("--ambient-source", choices=["hdri", "scene"], default=None)
    parser.add_argument("--point-light-mode", choices=["component", "target"], default=None)
    parser.add_argument("--hdri-mode", choices=["on", "off", "random"], default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--only", choices=["all", "spatial", "diffuse", "fixtures"], default="all")
    parser.add_argument("--background", action="store_true", help="Pass -b to Blender. Enabled by default.")
    parser.set_defaults(background=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
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
        str(root / "scripts" / "render_object_relighting.py"),
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
    if args.width is not None:
        cmd += ["--width", str(args.width)]
    if args.height is not None:
        cmd += ["--height", str(args.height)]
    if args.samples is not None:
        cmd += ["--samples", str(args.samples)]
    if args.output is not None:
        cmd += ["--output", args.output]
    if args.component_format is not None:
        cmd += ["--component-format", args.component_format]
    if args.ambient_source is not None:
        cmd += ["--ambient-source", args.ambient_source]
    if args.point_light_mode is not None:
        cmd += ["--point-light-mode", args.point_light_mode]
    if args.hdri_mode is not None:
        cmd += ["--hdri-mode", args.hdri_mode]

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
