from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1]
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.util_progress import progress_bar, progress_write
from utils.util_scene_index import (
    ROOT,
    existing_source_keys,
    load_index,
    next_id,
    repo_relative,
    resolve_repo_path,
    write_index,
)


INFINIGEN_REPO_URL = "https://github.com/princeton-vl/infinigen.git"
INFINIGEN_PYTHON_MAJOR_MINOR = (3, 11)
DEFAULT_ROOM_TYPES = ["DiningRoom", "Bathroom", "Bedroom", "Kitchen", "LivingRoom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Infinigen scenes, render review PNG/metadata, append background_scene_index.json."
    )
    parser.add_argument("--mode", choices=["indoors", "nature", "both"], default="indoors")
    parser.add_argument("--add-count", type=int, default=5)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--random-seeds", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=20260612)
    parser.add_argument("--infinigen-dir", default="external/infinigen")
    parser.add_argument("--work-dir", default="data/infinigen/work")
    parser.add_argument("--index-json", default="outputs/previews/infinigen/infinigen_index.json")
    parser.add_argument("--preview-dir", default="outputs/previews/infinigen")
    parser.add_argument(
        "--python-cmd",
        default=os.environ.get("INFINIGEN_PYTHON_CMD"),
        help="Python used for Infinigen install/generation. Defaults to python3.11 if it is available.",
    )
    parser.add_argument("--blender-cmd", default=os.environ.get("BLENDER_CMD", "blender"))
    parser.add_argument("--use-launch-blender", action="store_true")
    parser.add_argument("--auto-setup", action="store_true", help="Clone/install Infinigen if it is not importable.")
    parser.add_argument(
        "--install-mode",
        choices=["none", "minimal", "terrain"],
        default="none",
        help="Only used with --auto-setup. minimal is enough for Infinigen Indoors; terrain is for Nature.",
    )
    parser.add_argument("--indoors-configs", nargs="+", default=["fast_solve.gin", "singleroom.gin"])
    parser.add_argument("--nature-configs", nargs="+", default=["desert.gin", "simple.gin"])
    parser.add_argument("--room-types", nargs="+", default=DEFAULT_ROOM_TYPES)
    parser.add_argument("--room-type", default=None, help="Force one Infinigen Indoors room type.")
    parser.add_argument("--indoor-extra-overrides", nargs="*", default=[])
    parser.add_argument("--nature-extra-overrides", nargs="*", default=[])
    parser.add_argument("--nature-populate", dest="nature_populate", action="store_true", default=True)
    parser.add_argument("--no-nature-populate", dest="nature_populate", action="store_false")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=["current", "eevee", "cycles", "workbench"], default="cycles")
    parser.add_argument("--keep-generated", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--show-subprocess-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.python_cmd = choose_infinigen_python(args.python_cmd)
    return args


def python_cmd_parts(python_cmd: str) -> list[str]:
    return shlex.split(python_cmd)


def python_version(python_cmd: str) -> tuple[int, int, int] | None:
    cmd = python_cmd_parts(python_cmd) + [
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor, micro = result.stdout.strip().split(".")[:3]
        return int(major), int(minor), int(micro)
    except ValueError:
        return None


def is_infinigen_python(version: tuple[int, int, int] | None) -> bool:
    return version is not None and version[:2] == INFINIGEN_PYTHON_MAJOR_MINOR


def choose_infinigen_python(requested_python_cmd: str | None) -> str:
    if requested_python_cmd:
        return requested_python_cmd
    candidates = ["python3.11", sys.executable, "python3", "python"]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if is_infinigen_python(python_version(candidate)):
            return candidate
    return sys.executable


def require_infinigen_python(args: argparse.Namespace) -> None:
    version = python_version(args.python_cmd)
    if is_infinigen_python(version):
        return
    if version is None:
        found = "could not run it"
    else:
        found = f"it is Python {version[0]}.{version[1]}.{version[2]}"
    required = ".".join(str(part) for part in INFINIGEN_PYTHON_MAJOR_MINOR)
    raise SystemExit(
        f"Infinigen requires Python {required}.*, but --python-cmd is {args.python_cmd!r} and {found}.\n"
        "Install Python 3.11 or create a Python 3.11 venv/conda env, then rerun with "
        "--python-cmd /path/to/python3.11."
    )


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    dry_run: bool = False,
    log_path: Path | None = None,
    show_output: bool = True,
) -> None:
    command_text = " ".join(shlex.quote(part) for part in cmd)
    progress_write(f"[InfinigenHarvest] cmd: {command_text}")
    if dry_run:
        return
    if show_output:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n$ {command_text}\n")
        log.flush()
        subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)


def check_importable(args: argparse.Namespace, infinigen_dir: Path, env: dict[str, str]) -> bool:
    cmd = python_cmd_parts(args.python_cmd) + ["-c", "import infinigen, infinigen_examples"]
    try:
        result = subprocess.run(
            cmd,
            cwd=infinigen_dir if infinigen_dir.exists() else ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def setup_infinigen(args: argparse.Namespace, infinigen_dir: Path, env: dict[str, str]) -> None:
    if check_importable(args, infinigen_dir, env):
        return
    if not args.dry_run:
        require_infinigen_python(args)
    if not args.auto_setup:
        raise SystemExit(
            "Infinigen is not importable. Pass --auto-setup, or install it first and set --infinigen-dir."
        )
    if not infinigen_dir.exists():
        infinigen_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", INFINIGEN_REPO_URL, str(infinigen_dir)], ROOT, env, args.dry_run)
    if args.dry_run:
        return
    run_cmd(["git", "submodule", "update", "--init"], infinigen_dir, env)
    if args.install_mode != "none":
        pip_cmd = python_cmd_parts(args.python_cmd) + ["-m", "pip", "install", "-e"]
        install_env = dict(env)
        if args.install_mode == "minimal":
            install_env["INFINIGEN_MINIMAL_INSTALL"] = "True"
            pip_cmd.append(".")
        else:
            pip_cmd.append(".[terrain,vis]")
        run_cmd(pip_cmd, infinigen_dir, install_env)
    if not check_importable(args, infinigen_dir, env):
        raise SystemExit("Infinigen setup finished, but import still fails. Check the install log above.")


def build_env(infinigen_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    parts = [str(infinigen_dir)]
    if pythonpath:
        parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def module_cmd(args: argparse.Namespace, module: str, module_args: list[str]) -> list[str]:
    py = python_cmd_parts(args.python_cmd)
    if args.use_launch_blender:
        return py + ["-m", "infinigen.launch_blender", "-m", module, "--"] + module_args
    return py + ["-m", module] + module_args


def choose_seed(args: argparse.Namespace, attempt: int, rng: random.Random) -> int:
    if args.random_seeds:
        return rng.randrange(0, 2_000_000_000)
    return args.seed_start + attempt


def choose_generator(args: argparse.Namespace, attempt: int) -> str:
    if args.mode == "both":
        return "indoors" if attempt % 2 == 0 else "nature"
    return args.mode


def choose_room_type(args: argparse.Namespace, seed: int) -> str | None:
    if args.room_type:
        return args.room_type
    if not args.room_types:
        return None
    return args.room_types[seed % len(args.room_types)]


def build_indoors_commands(args: argparse.Namespace, seed: int, scene_root: Path, room_type: str | None) -> tuple[list[list[str]], Path, dict]:
    coarse_dir = scene_root / "coarse"
    overrides = [
        "compose_indoors.terrain_enabled=False",
        "compose_indoors.restrict_single_supported_roomtype=True",
        "restrict_solving.solve_max_rooms=1",
    ]
    if room_type:
        overrides.append(f'restrict_solving.restrict_parent_rooms=["{room_type}"]')
    overrides.extend(args.indoor_extra_overrides)
    command = module_cmd(
        args,
        "infinigen_examples.generate_indoors",
        [
            "--seed",
            str(seed),
            "--task",
            "coarse",
            "--output_folder",
            str(coarse_dir),
            "-g",
            *args.indoors_configs,
            "-p",
            *overrides,
        ],
    )
    record = {
        "generator": "indoors",
        "seed": seed,
        "room_type": room_type,
        "configs": args.indoors_configs,
        "overrides": overrides,
    }
    return [command], coarse_dir / "scene.blend", record


def build_nature_commands(args: argparse.Namespace, seed: int, scene_root: Path) -> tuple[list[list[str]], Path, dict]:
    coarse_dir = scene_root / "coarse"
    fine_dir = scene_root / "fine"
    commands = [
        module_cmd(
            args,
            "infinigen_examples.generate_nature",
            [
                "--seed",
                str(seed),
                "--task",
                "coarse",
                "-g",
                *args.nature_configs,
                "--output_folder",
                str(coarse_dir),
                *extra_overrides_args(args.nature_extra_overrides),
            ],
        )
    ]
    blend_path = coarse_dir / "scene.blend"
    if args.nature_populate:
        commands.append(
            module_cmd(
                args,
                "infinigen_examples.generate_nature",
                [
                    "--seed",
                    str(seed),
                    "--task",
                    "populate",
                    "fine_terrain",
                    "-g",
                    *args.nature_configs,
                    "--input_folder",
                    str(coarse_dir),
                    "--output_folder",
                    str(fine_dir),
                    *extra_overrides_args(args.nature_extra_overrides),
                ],
            )
        )
        blend_path = fine_dir / "scene.blend"
    record = {
        "generator": "nature",
        "seed": seed,
        "configs": args.nature_configs,
        "overrides": args.nature_extra_overrides,
        "nature_populate": args.nature_populate,
    }
    return commands, blend_path, record


def extra_overrides_args(overrides: list[str]) -> list[str]:
    if not overrides:
        return []
    return ["-p", *overrides]


def render_blend_preview(
    args: argparse.Namespace,
    blend_path: Path,
    preview_path: Path,
    metadata_path: Path,
    log_path: Path,
) -> None:
    script_path = ROOT / "dataset" / "utils" / "util_render_background_preview.py"
    cmd = shlex.split(args.blender_cmd) + [
        "-b",
        "--python",
        str(script_path),
        "--",
        "--blend",
        str(blend_path),
        "--preview",
        str(preview_path),
        "--metadata",
        str(metadata_path),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
        "--engine",
        args.engine,
    ]
    run_cmd(cmd, ROOT, dict(os.environ), args.dry_run, log_path=log_path, show_output=args.show_subprocess_output)


def source_key(record: dict) -> str:
    if record["generator"] == "indoors":
        return (
            f"infinigen-indoors:{record['seed']}:{record.get('room_type')}:"
            f"{','.join(record['configs'])}:{','.join(record['overrides'])}"
        )
    return (
        f"infinigen-nature:{record['seed']}:"
        f"{','.join(record['configs'])}:{','.join(record['overrides'])}:{record.get('nature_populate')}"
    )


def stop_reached(items: list[dict], added: int, args: argparse.Namespace) -> bool:
    if args.target_count is not None and len(items) >= args.target_count:
        return True
    return added >= args.add_count


def main() -> int:
    args = parse_args()
    infinigen_dir = resolve_repo_path(args.infinigen_dir)
    work_dir = resolve_repo_path(args.work_dir)
    preview_dir = resolve_repo_path(args.preview_dir)
    index_json = resolve_repo_path(args.index_json)
    env = build_env(infinigen_dir)
    setup_infinigen(args, infinigen_dir, env)

    items = load_index(index_json)
    seen = set()
    if args.skip_existing:
        seen.update(existing_source_keys(items, "infinigen-indoors"))
        seen.update(existing_source_keys(items, "infinigen-nature"))

    added = 0
    attempts = 0
    max_attempts = args.max_attempts or max(args.add_count * 5, 20)
    current_id = next_id(items)
    rng = random.Random(args.rng_seed)
    total_goal = args.add_count
    if args.target_count is not None:
        total_goal = max(args.target_count - len(items), 0)

    with progress_bar(total=total_goal, desc="Infinigen previews", unit="scene") as pbar:
        while attempts < max_attempts:
            items = load_index(index_json)
            if stop_reached(items, added, args):
                break
            seed = choose_seed(args, attempts, rng)
            generator = choose_generator(args, attempts)
            room_type = choose_room_type(args, seed) if generator == "indoors" else None
            scene_root = work_dir / generator / f"seed_{seed:08d}"
            if generator == "indoors":
                commands, blend_path, record = build_indoors_commands(args, seed, scene_root, room_type)
                source = "infinigen-indoors"
            else:
                commands, blend_path, record = build_nature_commands(args, seed, scene_root)
                source = "infinigen-nature"
            key = source_key(record)
            attempts += 1
            if key in seen:
                continue

            item_id = f"{current_id:05d}"
            current_id += 1
            dataset_name = preview_dir.name
            preview_path = preview_dir / "img" / f"{dataset_name}_{item_id}.png"
            metadata_path = preview_dir / "metadata" / f"{dataset_name}_{item_id}.json"
            log_path = preview_dir / "logs" / f"{dataset_name}_{item_id}.log"
            pbar.set_postfix(id=item_id, ok=added, attempts=attempts, source=source)
            progress_write(f"[InfinigenHarvest] {item_id} source={source} seed={seed} room={room_type}")
            try:
                for command in commands:
                    run_cmd(
                        command,
                        infinigen_dir,
                        env,
                        args.dry_run,
                        log_path=log_path,
                        show_output=args.show_subprocess_output,
                    )
                if not args.dry_run and not blend_path.exists():
                    raise FileNotFoundError(f"Generated scene.blend not found: {blend_path}")
                render_blend_preview(args, blend_path, preview_path, metadata_path, log_path)
            except Exception as exc:
                progress_write(f"[InfinigenHarvest] Skip failed scene seed={seed}: {exc}; log={log_path}")
                if not args.keep_generated and scene_root.exists():
                    shutil.rmtree(scene_root, ignore_errors=True)
                continue

            if args.dry_run:
                added += 1
                pbar.update(1)
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            record.update(
                {
                    "source": source,
                    "source_key": key,
                    "repo_url": INFINIGEN_REPO_URL,
                    "commands": [" ".join(shlex.quote(part) for part in command) for command in commands],
                    "generated_blend": str(blend_path),
                    "generated_deleted": not args.keep_generated,
                }
            )
            metadata.update(
                {
                    "source": source,
                    "source_key": key,
                    "infinigen": record,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
            item = {
                "id": item_id,
                "source": source,
                "source_key": key,
                "name": f"{source}_{seed:08d}",
                "license": "Infinigen BSD-3-Clause; generated assets may depend on selected Infinigen config",
                "asset_type": "procedural_scene",
                "download_link": INFINIGEN_REPO_URL,
                "preview_png": repo_relative(preview_path),
                "metadata_json": repo_relative(metadata_path),
                "seed": seed,
                "room_type": room_type,
                "generated_blend": str(blend_path),
                "generated_deleted": not args.keep_generated,
                "camera": metadata.get("camera"),
                "mesh_count": metadata.get("mesh_count"),
                "lights": metadata.get("lights", []),
                "bbox_min": metadata.get("bbox_min"),
                "bbox_max": metadata.get("bbox_max"),
                "subject_candidates": metadata.get("subject_candidates", []),
                "record": record,
            }
            items.append(item)
            write_index(index_json, items)
            seen.add(key)
            added += 1
            pbar.update(1)
            pbar.set_postfix(id=item_id, ok=added, attempts=attempts, total=len(items))
            progress_write(f"[InfinigenHarvest] Added {item_id}; total={len(items)} added={added}")
            if not args.keep_generated and scene_root.exists():
                shutil.rmtree(scene_root, ignore_errors=True)

    final_items = load_index(index_json)
    print(f"[InfinigenHarvest] Done attempts={attempts} added={added} total={len(final_items)}")
    return 0 if added > 0 or stop_reached(final_items, added, args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
