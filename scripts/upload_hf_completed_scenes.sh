#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-JaehoChae/blender_relight}"
REPO_TYPE="${REPO_TYPE:-dataset}"
SOURCE="${SOURCE:-outputs/blenderkit_dataset}"
REMOTE_DIR="${REMOTE_DIR:-completed_archives}"
STAGING_PARENT="${STAGING_PARENT:-outputs/hf_completed_upload_stage}"
ARCHIVE_DIR="${ARCHIVE_DIR:-outputs/hf_upload_archives}"
COMPRESSION="${COMPRESSION:-auto}"
SPLIT_SIZE="${SPLIT_SIZE:-}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"
KEEP_STAGE="${KEEP_STAGE:-0}"
HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload completed blender relight scenes archive}"
ARCHIVE_PATH=""
ARCHIVE_PARTS=()
PARTS_MANIFEST_PATH=""

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
  if ! truthy "$KEEP_ARCHIVE"; then
    if [[ -n "${ARCHIVE_PATH:-}" && -f "$ARCHIVE_PATH" ]]; then
      echo "[CLEANUP] removing local archive: $ARCHIVE_PATH"
      rm -f "$ARCHIVE_PATH"
    fi
    for part in "${ARCHIVE_PARTS[@]:-}"; do
      if [[ -f "$part" ]]; then
        echo "[CLEANUP] removing local archive part: $part"
        rm -f "$part"
      fi
    done
    if [[ -n "${PARTS_MANIFEST_PATH:-}" && -f "$PARTS_MANIFEST_PATH" ]]; then
      echo "[CLEANUP] removing local parts manifest: $PARTS_MANIFEST_PATH"
      rm -f "$PARTS_MANIFEST_PATH"
    fi
  fi
}
trap cleanup EXIT

upload_one_file() {
  local local_path="$1"
  local remote_path="$2"
  local commit_message="$3"
  "$PYTHON_CMD" - "$REPO_ID" "$REPO_TYPE" "$local_path" "$remote_path" "$commit_message" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id, repo_type, local_path, remote_path, commit_message = sys.argv[1:6]
local = Path(local_path)
if not local.is_file():
    raise SystemExit(f"File does not exist: {local}")

api = HfApi(token=os.environ.get("HF_TOKEN") or None)
api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
api.upload_file(
    path_or_fileobj=str(local),
    path_in_repo=remote_path,
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message=commit_message,
)
print(f"uploaded {local} -> {repo_id}/{remote_path}")
PY
}

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
  ARCHIVE_EXT="tar.zst"
else
  ARCHIVE_EXT="tar.gz"
fi

echo "[INFO] repo=${REPO_ID} repo_type=${REPO_TYPE}"
echo "[INFO] compression=${COMPRESSION}"

if [[ -n "$SPLIT_SIZE" ]]; then
  require_cmd split
  PART_PREFIX="${ARCHIVE_DIR}/${ARCHIVE_BASENAME}.${ARCHIVE_EXT}.part-"
  PARTS_MANIFEST_PATH="${ARCHIVE_DIR}/${ARCHIVE_BASENAME}.${ARCHIVE_EXT}.parts.json"
  rm -f "${PART_PREFIX}"* "$PARTS_MANIFEST_PATH"

  echo "[INFO] split_size=${SPLIT_SIZE}"
  echo "[INFO] archive_parts_prefix=${PART_PREFIX}"
  echo "[COMPRESS] creating split archive parts from completed-scene stage..."
  if [[ "$COMPRESSION" == "zst" ]]; then
    tar -cf - -C "$STAGING_PARENT" "$DATASET_NAME" \
      | zstd -T0 -19 -c \
      | split -b "$SPLIT_SIZE" -d -a 4 - "$PART_PREFIX"
  else
    tar -czf - -C "$STAGING_PARENT" "$DATASET_NAME" \
      | split -b "$SPLIT_SIZE" -d -a 4 - "$PART_PREFIX"
  fi

  mapfile -t ARCHIVE_PARTS < <(find "$ARCHIVE_DIR" -maxdepth 1 -type f -name "$(basename "$PART_PREFIX")*" | sort)
  if ((${#ARCHIVE_PARTS[@]} == 0)); then
    die "Failed to create split archive parts."
  fi
  du -ch "${ARCHIVE_PARTS[@]}" | tail -n 1

  "$PYTHON_CMD" - "$PARTS_MANIFEST_PATH" "$ARCHIVE_BASENAME" "$ARCHIVE_EXT" "$SPLIT_SIZE" "${ARCHIVE_PARTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
archive_basename = sys.argv[2]
archive_ext = sys.argv[3]
split_size = sys.argv[4]
parts = [Path(p) for p in sys.argv[5:]]
part_names = [p.name for p in parts]
manifest = {
    "schema": "hf_split_archive_manifest_v1",
    "archive_name": f"{archive_basename}.{archive_ext}",
    "split_size": split_size,
    "part_count": len(part_names),
    "parts": part_names,
    "reconstruct": f"cat {archive_basename}.{archive_ext}.part-* > {archive_basename}.{archive_ext}",
}
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
PY

  echo "[UPLOAD] uploading ${#ARCHIVE_PARTS[@]} archive parts to Hugging Face..."
  for part in "${ARCHIVE_PARTS[@]}"; do
    remote_path="${REMOTE_DIR}/$(basename "$part")"
    upload_one_file "$part" "$remote_path" "$HF_UPLOAD_COMMIT_MESSAGE"
  done
  upload_one_file "$PARTS_MANIFEST_PATH" "${REMOTE_DIR}/$(basename "$PARTS_MANIFEST_PATH")" "$HF_UPLOAD_COMMIT_MESSAGE"
else
  ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_BASENAME}.${ARCHIVE_EXT}"
  REMOTE_PATH="${REMOTE_PATH:-${REMOTE_DIR}/$(basename "$ARCHIVE_PATH")}"

  echo "[INFO] archive=${ARCHIVE_PATH}"
  echo "[INFO] remote_path=${REMOTE_PATH}"
  echo "[COMPRESS] creating archive from completed-scene stage..."
  if [[ "$COMPRESSION" == "zst" ]]; then
    tar -cf "$ARCHIVE_PATH" --use-compress-program "zstd -T0 -19" -C "$STAGING_PARENT" "$DATASET_NAME"
  else
    tar -czf "$ARCHIVE_PATH" -C "$STAGING_PARENT" "$DATASET_NAME"
  fi
  du -h "$ARCHIVE_PATH"

  echo "[UPLOAD] uploading to Hugging Face..."
  upload_one_file "$ARCHIVE_PATH" "$REMOTE_PATH" "$HF_UPLOAD_COMMIT_MESSAGE"
fi

echo "[DONE] uploaded ${#META_FILES[@]} completed scenes"
