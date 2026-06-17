#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-JaehoChae/blender_relight}"
REPO_TYPE="${REPO_TYPE:-dataset}"
SOURCE="${SOURCE:-outputs/blenderkit_dataset}"
REMOTE_DIR="${REMOTE_DIR:-completed_archives}"
STAGING_PARENT="${STAGING_PARENT:-outputs/hf_completed_upload_stage}"
ARCHIVE_DIR="${ARCHIVE_DIR:-outputs/hf_upload_archives}"
COMPRESSION="${COMPRESSION:-auto}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"
KEEP_STAGE="${KEEP_STAGE:-0}"
HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload completed blender relight scenes archive}"

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

cleanup() {
  if ! truthy "$KEEP_STAGE" && [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then
    echo "[CLEANUP] removing stage: $STAGE_ROOT"
    rm -rf "$STAGE_ROOT"
  fi
  if ! truthy "$KEEP_ARCHIVE" && [[ -n "${ARCHIVE_PATH:-}" && -f "$ARCHIVE_PATH" ]]; then
    echo "[CLEANUP] removing local archive: $ARCHIVE_PATH"
    rm -f "$ARCHIVE_PATH"
  fi
}
trap cleanup EXIT

require_cmd tar
require_cmd cp
require_cmd find
require_cmd "$PYTHON_CMD"

"$PYTHON_CMD" - <<'PY' || die "Missing Python package: huggingface_hub. Install with: python3 -m pip install -U huggingface_hub"
import huggingface_hub  # noqa: F401
PY

SOURCE_ABS="$(realpath "$SOURCE")"
[[ -d "$SOURCE_ABS" ]] || die "SOURCE must be a dataset directory or shard parent: $SOURCE"

SOURCE_ROOTS=()
if [[ -d "$SOURCE_ABS/scenes" ]]; then
  SOURCE_ROOTS+=("$SOURCE_ABS")
else
  mapfile -t SOURCE_ROOTS < <(
    find "$SOURCE_ABS" -mindepth 2 -maxdepth 2 -type d -name scenes -printf '%h\n' | sort
  )
fi

if ((${#SOURCE_ROOTS[@]} == 0)); then
  die "SOURCE does not contain scenes/ directly or one level below it: $SOURCE"
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_NAME="$(basename "$SOURCE_ABS")"
DATASET_NAME="${DATASET_NAME:-${SOURCE_NAME}_completed_${TIMESTAMP}}"
STAGE_ROOT="${STAGING_PARENT}/${DATASET_NAME}"
ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-${DATASET_NAME}}"

mkdir -p "$STAGING_PARENT" "$ARCHIVE_DIR"
rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT/scenes"

META_FILES=()
META_SOURCE_ROOTS=()
for source_root in "${SOURCE_ROOTS[@]}"; do
  while IFS= read -r meta; do
    META_FILES+=("$meta")
    META_SOURCE_ROOTS+=("$source_root")
  done < <(find "$source_root/scenes" -mindepth 2 -maxdepth 2 -type f -name meta.json | sort)
done

if ((${#META_FILES[@]} == 0)); then
  die "No completed scenes found. A completed scene must contain scenes/<scene_id>/meta.json."
fi

echo "[INFO] source roots=${#SOURCE_ROOTS[@]}"
for source_root in "${SOURCE_ROOTS[@]}"; do
  echo "[INFO]   root=${source_root}"
done
echo "[INFO] completed scenes snapshot=${#META_FILES[@]}"
echo "[INFO] source=${SOURCE_ABS}"
echo "[INFO] stage=${STAGE_ROOT}"

preview_stage="$STAGE_ROOT/scenes/preview"
mkdir -p "$preview_stage"

scene_ids=()
scene_source_roots=()
declare -A seen_scene_sources=()
for idx in "${!META_FILES[@]}"; do
  meta="${META_FILES[$idx]}"
  source_root="${META_SOURCE_ROOTS[$idx]}"
  scene_dir="$(dirname "$meta")"
  scene_id="$(basename "$scene_dir")"
  if [[ -n "${seen_scene_sources[$scene_id]:-}" ]]; then
    die "Duplicate scene_id=${scene_id} in ${seen_scene_sources[$scene_id]} and ${scene_dir}"
  fi
  seen_scene_sources[$scene_id]="$scene_dir"
  scene_ids+=("$scene_id")
  scene_source_roots+=("$source_root")
  cp -al "$scene_dir" "$STAGE_ROOT/scenes/$scene_id"
  preview_source="$source_root/scenes/preview"
  if [[ -d "$preview_source" ]]; then
    while IFS= read -r -d '' preview; do
      preview_dest="$preview_stage/$(basename "$preview")"
      if [[ ! -e "$preview_dest" ]]; then
        cp -al "$preview" "$preview_dest"
      fi
    done < <(find "$preview_source" -maxdepth 1 -type f -name "${scene_id}_*" -print0)
  fi
done

tmp_count_before="$(
  find "$STAGE_ROOT" -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name '.light_*.tmp.*' -o -name '.ambient.tmp.*' \) | wc -l
)"
if ((tmp_count_before > 0)); then
  echo "[INFO] removing ${tmp_count_before} temporary files from stage only"
  find "$STAGE_ROOT" -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name '.light_*.tmp.*' -o -name '.ambient.tmp.*' \) -delete
fi

"$PYTHON_CMD" - "$STAGE_ROOT" "$SOURCE_ABS" "${#SOURCE_ROOTS[@]}" "${SOURCE_ROOTS[@]}" "${scene_ids[@]}" -- "${scene_source_roots[@]}" <<'PY'
import json
import sys
from pathlib import Path

stage = Path(sys.argv[1])
source = Path(sys.argv[2])
source_root_count = int(sys.argv[3])
source_roots = sys.argv[4:4 + source_root_count]
rest = sys.argv[4 + source_root_count:]
sep = rest.index("--")
scene_ids = rest[:sep]
scene_source_roots = rest[sep + 1:]
if len(scene_ids) != len(scene_source_roots):
    raise SystemExit("Internal error: scene id/source-root mismatch")
manifest = {
    "schema": "hf_completed_scenes_upload_manifest_v1",
    "source": str(source),
    "source_roots": source_roots,
    "scene_count": len(scene_ids),
    "scenes": [
        {
            "scene_id": scene_id,
            "source_root": scene_source_root,
            "meta": f"scenes/{scene_id}/meta.json",
        }
        for scene_id, scene_source_root in zip(scene_ids, scene_source_roots)
    ],
}
(stage / "upload_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
PY

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
  ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_BASENAME}.tar.zst"
  TAR_ARGS=(-cf "$ARCHIVE_PATH" --use-compress-program "zstd -T0 -19")
else
  ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_BASENAME}.tar.gz"
  TAR_ARGS=(-czf "$ARCHIVE_PATH")
fi

REMOTE_PATH="${REMOTE_PATH:-${REMOTE_DIR}/$(basename "$ARCHIVE_PATH")}"

echo "[INFO] repo=${REPO_ID} repo_type=${REPO_TYPE}"
echo "[INFO] archive=${ARCHIVE_PATH}"
echo "[INFO] remote_path=${REMOTE_PATH}"
echo "[INFO] compression=${COMPRESSION}"
echo "[COMPRESS] creating archive from completed-scene stage..."
tar "${TAR_ARGS[@]}" -C "$STAGING_PARENT" "$DATASET_NAME"
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

echo "[DONE] uploaded ${#META_FILES[@]} completed scenes"
