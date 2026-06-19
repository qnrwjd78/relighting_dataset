from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "outputs" / "previews" / "portrait_render_manifests"
CONFIG_DIR = ROOT / "configs" / "generated" / "portrait_render"
LOG_DIR = ROOT / "logs" / "portrait_render"



SKETCHFAB_SELECTED_IDS = [
    "sketchfab_human_000001",
    "sketchfab_human_000003",
    "sketchfab_human_000004",
    "sketchfab_human_000006",
    "sketchfab_human_000007",
    "sketchfab_human_000008",
    "sketchfab_human_000012",
    "sketchfab_human_000013",
    "sketchfab_human_000020",
]

SKETCHFAB_WEAK_IDS = [
    "sketchfab_human_000005",
    "sketchfab_human_000016",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    base_config: str
    manifest_name: str
    debug_output: str
    full_output: str


SOURCES = {
    "sketchfab": SourceSpec(
        name="sketchfab",
        base_config="configs/tokenlight_synthetic_full.json",
        manifest_name="sketchfab_selected_objects.txt",
        debug_output="outputs/portrait_debug/sketchfab_human",
        full_output="outputs/portrait_exr/sketchfab_human",
    ),
    "3dscan": SourceSpec(
        name="3dscan",
        base_config="configs/tokenlight_synthetic_full.json",
        manifest_name="3dscanstore_free_head_selected_objects.txt",
        debug_output="outputs/portrait_debug/3dscanstore_free_head",
        full_output="outputs/portrait_exr/3dscanstore_free_head",
    ),
    "facescape": SourceSpec(
        name="facescape",
        base_config="configs/tokenlight_synthetic_full.json",
        manifest_name="facescape_tu_neutral20_objects.txt",
        debug_output="outputs/portrait_debug/facescape_tu_neutral20",
        full_output="outputs/portrait_exr/facescape_tu_neutral20",
    ),
    "hsrd100": SourceSpec(
        name="hsrd100",
        base_config="configs/tokenlight_synthetic_full.json",
        manifest_name="hsrd100_lod1_objects.txt",
        debug_output="outputs/portrait_debug/hsrd100_lod1",
        full_output="outputs/portrait_exr/hsrd100_lod1",
    ),
}

DEFAULT_SOURCE_ORDER = ["sketchfab", "3dscan", "facescape", "hsrd100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run portrait-source TokenLight relighting renders."
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "debug", "full", "both"],
        default="prepare",
        help="prepare writes manifests/configs only; debug/full also run Blender.",
    )
    parser.add_argument("--sources", nargs="+", choices=DEFAULT_SOURCE_ORDER, default=DEFAULT_SOURCE_ORDER)
    parser.add_argument("--blender-exe", default=os.environ.get("BLENDER_EXE", "blender"))
    parser.add_argument("--dry-run", action="store_true", help="Print Blender commands without running them.")
    parser.add_argument("--include-weak-sketchfab", action="store_true")
    parser.add_argument("--facescape-count", type=int, default=20)
    parser.add_argument("--hsrd-limit", type=int, default=None, help="Optional cap for the HSRD100 LOD1 manifest.")
    parser.add_argument("--debug-max-scenes", type=int, default=10)
    parser.add_argument("--full-max-scenes", type=int, default=None)
    parser.add_argument("--debug-width", type=int, default=768)
    parser.add_argument("--debug-height", type=int, default=768)
    parser.add_argument("--debug-samples", type=int, default=32)
    parser.add_argument("--full-width", type=int, default=960)
    parser.add_argument("--full-height", type=int, default=960)
    parser.add_argument("--full-samples", type=int, default=32)
    parser.add_argument("--positions-per-scene", type=int, default=None)
    parser.add_argument("--only", choices=["all", "spatial", "diffuse", "fixtures"], default="spatial")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def repo_relative(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def normalize_manifest_line(line: str) -> str:
    line = line.strip()
    if line.startswith("/workspace/"):
        line = line[len("/workspace/") :]
    return repo_relative(line)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def selected_sketchfab_paths(include_weak: bool) -> list[str]:
    ids = list(SKETCHFAB_SELECTED_IDS)
    if include_weak:
        ids.extend(SKETCHFAB_WEAK_IDS)

    paths: list[str] = []
    missing: list[str] = []
    for item_id in ids:
        meta_path = ROOT / "outputs" / "previews" / "sketchfab_human" / "metadata" / f"{item_id}.json"
        if not meta_path.exists():
            missing.append(f"{item_id}: missing metadata")
            continue
        meta = load_json(meta_path)
        source = meta.get("source_path") or meta.get("asset")
        if not source:
            missing.append(f"{item_id}: no source_path")
            continue
        local = source[len("/workspace/") :] if str(source).startswith("/workspace/") else source
        if not repo_path(local).exists():
            missing.append(f"{item_id}: source missing {source}")
            continue
        paths.append(normalize_manifest_line(str(source)))

    if missing:
        print("[WARN] skipped sketchfab item(s):", file=sys.stderr)
        for row in missing:
            print(f"  - {row}", file=sys.stderr)
    return list(dict.fromkeys(paths))


def selected_3dscan_paths() -> list[str]:
    path = ROOT / "data" / "3dscanstore_free_head" / "extracted" / "Blender" / "Blender Scene.blend"
    if not path.exists():
        raise FileNotFoundError(path)
    return [repo_relative(path)]


def selected_facescape_paths(count: int) -> list[str]:
    paths: list[str] = []
    for subject in range(1, count + 1):
        path = (
            ROOT
            / "data"
            / "facescape"
            / "tu_model"
            / "extracted"
            / str(subject)
            / "models_reg"
            / "1_neutral.obj"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(repo_relative(path))
    return paths


def selected_hsrd100_paths(limit: int | None) -> list[str]:
    manifest = ROOT / "outputs" / "previews" / "hsrd100" / "hsrd100_lod1_objects.txt"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    lines = [
        normalize_manifest_line(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return lines[:limit] if limit is not None else lines


def build_manifest(source: str, args: argparse.Namespace) -> Path:
    spec = SOURCES[source]
    if source == "sketchfab":
        lines = selected_sketchfab_paths(args.include_weak_sketchfab)
    elif source == "3dscan":
        lines = selected_3dscan_paths()
    elif source == "facescape":
        lines = selected_facescape_paths(args.facescape_count)
    elif source == "hsrd100":
        lines = selected_hsrd100_paths(args.hsrd_limit)
    else:  # pragma: no cover
        raise ValueError(source)

    if not lines:
        raise RuntimeError(f"No assets selected for {source}")
    out = MANIFEST_DIR / spec.manifest_name
    write_lines(out, lines)
    print(f"[prepare] {source}: wrote {len(lines)} asset(s) -> {out.relative_to(ROOT)}")
    return out



def build_config(source: str, manifest: Path, args: argparse.Namespace) -> Path:
    spec = SOURCES[source]
    base_config = repo_path(spec.base_config)
    config = load_json(base_config)
    config["object_manifest"] = repo_relative(manifest)
    config["output_root"] = spec.full_output
    if args.positions_per_scene is not None:
        config.setdefault("spatial", {})["positions_per_scene"] = int(args.positions_per_scene)

    out = CONFIG_DIR / f"{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[prepare] {source}: wrote config -> {out.relative_to(ROOT)}")
    return out


def prepare(args: argparse.Namespace) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    for source in args.sources:
        manifest = build_manifest(source, args)
        configs[source] = build_config(source, manifest, args)
    return configs


def blender_command(source: str, config: Path, stage: str, args: argparse.Namespace) -> list[str]:
    spec = SOURCES[source]
    debug = stage == "debug"
    cmd = [
        args.blender_exe,
        "-b",
        "--python",
        str(ROOT / "scripts" / "render_object_relighting.py"),
        "--",
        "--config",
        str(config),
        "--output",
        spec.debug_output if debug else spec.full_output,
        "--width",
        str(args.debug_width if debug else args.full_width),
        "--height",
        str(args.debug_height if debug else args.full_height),
        "--samples",
        str(args.debug_samples if debug else args.full_samples),
        "--component-format",
        "png" if debug else "exr",
        "--hdri-mode",
        "on",
        "--ambient-source",
        "hdri",
        "--point-light-mode",
        "component",
        "--only",
        args.only,
    ]
    if debug:
        cmd += ["--debug", "--light-preview"]
        if args.debug_max_scenes is not None:
            cmd += ["--max-scenes", str(args.debug_max_scenes)]
    elif args.full_max_scenes is not None:
        cmd += ["--max-scenes", str(args.full_max_scenes)]
    if args.positions_per_scene is not None:
        cmd += ["--positions-per-scene", str(args.positions_per_scene)]
    return cmd


def run_stage(stage: str, configs: dict[str, Path], args: argparse.Namespace) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        cmd = blender_command(source, configs[source], stage, args)
        log_path = LOG_DIR / f"{source}_{stage}.log"
        print(f"[{stage}] {source}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            print(f"[ERROR] {source} {stage} failed with exit {result.returncode}. Log: {log_path}", file=sys.stderr)
            return result.returncode
        print(f"[{stage}] {source}: done. Log: {log_path.relative_to(ROOT)}")
    return 0


def main() -> int:
    args = parse_args()
    configs = prepare(args)
    if args.stage in {"debug", "both"}:
        code = run_stage("debug", configs, args)
        if code != 0:
            return code
    if args.stage in {"full", "both"}:
        code = run_stage("full", configs, args)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
