from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an unofficial Hugging Face mirror of 3D-FRONT. "
            "Official 3D-FRONT access requires accepting dataset terms separately."
        )
    )
    parser.add_argument("--repo-id", default="huanngzh/3D-Front")
    parser.add_argument("--out-dir", default="data/indoor/3dfront_hf")
    parser.add_argument("--token", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir-use-symlinks", choices=["auto", "true", "false"], default="auto")
    parser.add_argument(
        "--subset",
        choices=["scene", "test-scene", "scene-and-render", "full"],
        default="scene",
        help="Default downloads only 3D-FRONT-SCENE parts, not point clouds or rendered images.",
    )
    parser.add_argument("--prepare", action="store_true", help="Run prepare_3dfront.py after download if folders match.")
    parser.add_argument(
        "--accept-mirror-risk",
        action="store_true",
        help="Required because this is not the official 3D-FRONT distribution channel.",
    )
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


def main() -> int:
    args = parse_args()
    if not args.accept_mirror_risk:
        raise SystemExit(
            "Official 3D-FRONT download is terms-gated, so this script only downloads an unofficial HF mirror. "
            "Re-run with --accept-mirror-risk if you have checked the license/source and still want it."
        )
    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": str(out_dir),
        "local_dir_use_symlinks": args.local_dir_use_symlinks,
    }
    if args.token:
        kwargs["token"] = args.token
    if args.revision:
        kwargs["revision"] = args.revision
    if args.subset == "scene":
        kwargs["allow_patterns"] = ["3D-FRONT-SCENE.part*", "*.json", "README.md"]
    elif args.subset == "test-scene":
        kwargs["allow_patterns"] = ["3D-FRONT-TEST-SCENE.tar.gz", "*test*.json", "README.md"]
    elif args.subset == "scene-and-render":
        kwargs["allow_patterns"] = ["3D-FRONT-SCENE.part*", "3D-FRONT-RENDER.tar.gz", "*.json", "README.md"]
    print(f"[3D-FRONT HF mirror] downloading {args.repo_id} -> {out_dir}")
    snapshot_download(**kwargs)
    if args.prepare:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "dataset" / "indoor" / "prepare_3dfront.py"),
                "--root",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
