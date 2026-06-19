from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HSSD scene/model repositories from Hugging Face.")
    parser.add_argument("--out-dir", default="data/indoor/hssd")
    parser.add_argument("--repos", nargs="+", default=["hssd/hssd-scenes", "hssd/hssd-models"])
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated/authenticated access.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir-use-symlinks", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--scene-limit", type=int, default=None, help="Download only the first N scene GLBs from hssd/hssd-scenes.")
    parser.add_argument("--scene-patterns", nargs="+", default=None, help="Explicit allow_patterns for hssd/hssd-scenes.")
    parser.add_argument("--prepare", action="store_true", help="Run prepare_hssd.py after download.")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing huggingface_hub. Install it with: pip install huggingface_hub") from exc


def scene_allow_patterns(repo_id: str, args: argparse.Namespace) -> list[str] | None:
    if repo_id != "hssd/hssd-scenes":
        return None
    if args.scene_patterns:
        return args.scene_patterns
    if args.scene_limit is None:
        return None

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=False, revision=args.revision)
    scenes = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("scenes/") and sibling.rfilename.endswith(".glb")
    )
    selected = scenes[: args.scene_limit]
    return selected + ["README.md", "*.json", "configs/*.json"]


def download_repo(repo_id: str, out_dir: Path, args: argparse.Namespace) -> Path:
    from huggingface_hub import snapshot_download

    target = out_dir / repo_id.split("/")[-1]
    allow_patterns = scene_allow_patterns(repo_id, args)
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(target),
        "local_dir_use_symlinks": args.local_dir_use_symlinks,
    }
    if args.token:
        kwargs["token"] = args.token
    if args.revision:
        kwargs["revision"] = args.revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    print(f"[HSSD] downloading {repo_id} -> {target}")
    if allow_patterns:
        scene_count = sum(1 for item in allow_patterns if item.startswith("scenes/") and item.endswith(".glb"))
        print(f"[HSSD] allow_patterns={len(allow_patterns)} scene_limit={scene_count or 'custom'}")
    snapshot_download(**kwargs)
    return target


def main() -> int:
    args = parse_args()
    ensure_huggingface_hub()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in args.repos:
        download_repo(repo_id, out_dir, args)
    if args.prepare:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "dataset" / "indoor" / "prepare_hssd.py"),
                "--root",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
