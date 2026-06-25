#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-outputs/objaverse_dataset_exr}"
PNG_STAGE_PARENT="${PNG_STAGE_PARENT:-${PNG_STAGE:-outputs/objaverse_dataset_png_batches}}"
REMOTE_DIR="${REMOTE_DIR:-completed_png_archives}"
COMPRESSION="${COMPRESSION:-auto}"
SPLIT_SIZE="${SPLIT_SIZE:-20G}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
CONVERT_WORKERS="${CONVERT_WORKERS:-}"
CONVERT_SEED="${CONVERT_SEED:-1234}"
SINGLE_LIGHTS="${SINGLE_LIGHTS:-all}"
MAX_SINGLE_LIGHTS="${MAX_SINGLE_LIGHTS:-0}"
TWO_LIGHT_SAMPLES="${TWO_LIGHT_SAMPLES:-32}"
GLOBAL_AMBIENT_SAMPLES="${GLOBAL_AMBIENT_SAMPLES:-7}"
GLOBAL_DIFFUSE_SAMPLES="${GLOBAL_DIFFUSE_SAMPLES:-0}"
INCLUDE_GLOBAL_DIFFUSE="${INCLUDE_GLOBAL_DIFFUSE:-0}"
BATCH_SCENES="${BATCH_SCENES:-500}"
KEEP_PNG_STAGE="${KEEP_PNG_STAGE:-0}"
OVERWRITE_PNG_STAGE="${OVERWRITE_PNG_STAGE:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

truthy() {
  [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]
}

cleanup_png_stage() {
  local stage="$1"
  if truthy "$KEEP_PNG_STAGE"; then
    return
  fi
  if [[ -n "$stage" && -d "$stage" ]]; then
    echo "[CLEANUP] removing PNG batch stage: $stage"
    rm -rf "$stage"
  fi
}

if ! [[ "$BATCH_SCENES" =~ ^[0-9]+$ ]] || ((BATCH_SCENES <= 0)); then
  echo "[ERROR] BATCH_SCENES must be a positive integer: $BATCH_SCENES" >&2
  exit 1
fi

mapfile -t SOURCE_ROOTS < <(
  if [[ -d "$SOURCE/scenes" ]]; then
    realpath "$SOURCE"
  else
    find "$SOURCE" -mindepth 2 -maxdepth 2 -type d -name scenes -printf '%h\n' | sort | xargs -r realpath
  fi
)
if ((${#SOURCE_ROOTS[@]} == 0)); then
  echo "[ERROR] SOURCE does not contain scenes/ directly or one level below it: $SOURCE" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$PNG_STAGE_PARENT"

for source_root in "${SOURCE_ROOTS[@]}"; do
  scene_count="$(find "$source_root/scenes" -mindepth 2 -maxdepth 2 -type f -name meta.json | wc -l)"
  if ((scene_count == 0)); then
    continue
  fi

  source_base="$(basename "$source_root")"
  offset=0
  while ((offset < scene_count)); do
    end=$((offset + BATCH_SCENES - 1))
    if ((end >= scene_count)); then
      end=$((scene_count - 1))
    fi

    batch_name="${source_base}_batch_${offset}_${end}_${RUN_ID}"
    png_stage="${PNG_STAGE_PARENT}/${batch_name}"

    echo "[BATCH] source=${source_root} scenes=${offset}-${end} stage=${png_stage}"
    rm -rf "$png_stage"

    convert_args=(
      --source "$source_root"
      --dest "$png_stage"
      --scene-offset "$offset"
      --scene-limit "$BATCH_SCENES"
      --seed "$CONVERT_SEED"
      --single-lights "$SINGLE_LIGHTS"
      --max-single-lights "$MAX_SINGLE_LIGHTS"
      --two-light-samples "$TWO_LIGHT_SAMPLES"
      --global-ambient-samples "$GLOBAL_AMBIENT_SAMPLES"
      --global-diffuse-samples "$GLOBAL_DIFFUSE_SAMPLES"
    )
    if [[ -n "$CONVERT_WORKERS" ]]; then
      convert_args+=(--workers "$CONVERT_WORKERS")
    fi
    if truthy "$OVERWRITE_PNG_STAGE"; then
      convert_args+=(--overwrite)
    fi
    if truthy "$INCLUDE_GLOBAL_DIFFUSE"; then
      convert_args+=(--include-global-diffuse)
    fi

    echo "[CONVERT] creating PNG batch..."
    "$PYTHON_CMD" scripts/convert_completed_exr_dataset_to_png.py "${convert_args[@]}"
    du -sh "$png_stage"

    echo "[UPLOAD] compressing and uploading PNG batch..."
    SOURCE="$png_stage" \
    REMOTE_DIR="$REMOTE_DIR" \
    DATASET_NAME="$batch_name" \
    ARCHIVE_BASENAME="$batch_name" \
    COMPRESSION="$COMPRESSION" \
    SPLIT_SIZE="$SPLIT_SIZE" \
    PYTHON_CMD="$PYTHON_CMD" \
    HF_UPLOAD_COMMIT_MESSAGE="${HF_UPLOAD_COMMIT_MESSAGE:-Upload completed TokenLight PNG scenes archive}" \
    scripts/upload_hf_completed_scenes.sh

    echo "[DONE] uploaded batch: $batch_name"
    cleanup_png_stage "$png_stage"
    offset=$((offset + BATCH_SCENES))
  done
done

echo "[DONE] all PNG batches uploaded"
