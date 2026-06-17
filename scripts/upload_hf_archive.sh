#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-JaehoChae/blender_relight}"
REPO_TYPE="${REPO_TYPE:-dataset}"
SOURCE="${SOURCE:-outputs/blenderkit_dataset}"
REMOTE_DIR="${REMOTE_DIR:-archives}"
STAGING_DIR="${STAGING_DIR:-outputs/hf_upload_archives}"
COMPRESSION="${COMPRESSION:-auto}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload compressed blender relight archive}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

truthy() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

cleanup_archive() {
  if truthy "$KEEP_ARCHIVE"; then
    return
  fi
  if [[ -n "${ARCHIVE_PATH:-}" && -f "$ARCHIVE_PATH" ]]; then
    echo "[CLEANUP] removing local archive: $ARCHIVE_PATH"
    rm -f "$ARCHIVE_PATH"
  fi
}
trap cleanup_archive EXIT

require_cmd tar
require_cmd "$PYTHON_CMD"

"$PYTHON_CMD" - <<'PY' || die "Missing Python package: huggingface_hub. Install with: python3 -m pip install -U huggingface_hub"
import huggingface_hub  # noqa: F401
PY

SOURCE_ABS="$(realpath "$SOURCE")"
[[ -e "$SOURCE_ABS" ]] || die "SOURCE does not exist: $SOURCE"

if [[ -d "$SOURCE_ABS" ]] && ! truthy "$ALLOW_INCOMPLETE"; then
  tmp_count="$(
    find "$SOURCE_ABS" -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name '.light_*.tmp.*' -o -name '.ambient.tmp.*' \) | head -100 | wc -l
  )"
  if ((tmp_count > 0)); then
    echo "[ERROR] Found temporary render files under SOURCE. The dataset may still be rendering." >&2
    echo "        Re-run after rendering finishes, or set ALLOW_INCOMPLETE=1 if you really want to upload now." >&2
    find "$SOURCE_ABS" -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name '.light_*.tmp.*' -o -name '.ambient.tmp.*' \) | head -20 >&2
    exit 1
  fi
fi

mkdir -p "$STAGING_DIR"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_NAME="$(basename "$SOURCE_ABS")"
ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-${SOURCE_NAME}_${TIMESTAMP}}"

case "$COMPRESSION" in
  auto)
    if command -v zstd >/dev/null 2>&1; then
      COMPRESSION="zst"
    else
      COMPRESSION="gz"
    fi
    ;;
  zst|gz) ;;
  *) die "COMPRESSION must be one of: auto, zst, gz" ;;
esac

if [[ "$COMPRESSION" == "zst" ]]; then
  require_cmd zstd
  ARCHIVE_PATH="${STAGING_DIR}/${ARCHIVE_BASENAME}.tar.zst"
  TAR_ARGS=(-cf "$ARCHIVE_PATH" --use-compress-program "zstd -T0 -19")
else
  ARCHIVE_PATH="${STAGING_DIR}/${ARCHIVE_BASENAME}.tar.gz"
  TAR_ARGS=(-czf "$ARCHIVE_PATH")
fi

REMOTE_PATH="${REMOTE_PATH:-${REMOTE_DIR}/$(basename "$ARCHIVE_PATH")}"

echo "[INFO] repo=${REPO_ID} repo_type=${REPO_TYPE}"
echo "[INFO] source=${SOURCE_ABS}"
echo "[INFO] archive=${ARCHIVE_PATH}"
echo "[INFO] remote_path=${REMOTE_PATH}"
echo "[INFO] compression=${COMPRESSION}"

SOURCE_PARENT="$(dirname "$SOURCE_ABS")"
SOURCE_BASE="$(basename "$SOURCE_ABS")"

echo "[COMPRESS] creating archive..."
tar "${TAR_ARGS[@]}" -C "$SOURCE_PARENT" "$SOURCE_BASE"
du -h "$ARCHIVE_PATH"

echo "[UPLOAD] uploading to Hugging Face..."
"$PYTHON_CMD" - "$REPO_ID" "$REPO_TYPE" "$ARCHIVE_PATH" "$REMOTE_PATH" "$HF_UPLOAD_COMMIT_MESSAGE" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, repo_type, archive_path, remote_path, commit_message = sys.argv[1:6]
archive = Path(archive_path)
if not archive.is_file():
    raise SystemExit(f"Archive does not exist: {archive}")

api = HfApi(token=os.environ.get("HF_TOKEN") or None)
api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
api.upload_file(
    path_or_fileobj=str(archive),
    path_in_repo=remote_path,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message=commit_message,
)
print(f"uploaded {archive} -> {repo_id}/{remote_path}")
PY

echo "[DONE] upload finished"
