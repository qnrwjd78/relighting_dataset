#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 GPU_COUNT START_INDEX SCENE_NUM [full]"
  echo
  echo "Examples:"
  echo "  $0 4 500 250"
  echo "  $0 4 500 250 full"
  echo
  echo "Environment:"
  echo "  OUTPUTS_DIR=outputs"
  echo "  DATASET_NAME=tokenlight_synthetic_1280"
  echo "  ARCHIVE_DIR=."
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 1
fi

GPU_COUNT="$1"
START_INDEX="$2"
SCENE_NUM="$3"
MODE="${4:-partial}"

if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ && "${START_INDEX}" =~ ^[0-9]+$ && "${SCENE_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_COUNT, START_INDEX, and SCENE_NUM must be non-negative integers; GPU_COUNT and SCENE_NUM must be greater than 0."
  exit 1
fi

if [[ "${MODE}" != "partial" && "${MODE}" != "full" ]]; then
  echo "Optional mode must be 'full' when provided."
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
DATASET_NAME="${DATASET_NAME:-tokenlight_synthetic_1280}"
ARCHIVE_DIR="${ARCHIVE_DIR:-.}"
DATASET_DIR="${ROOT_DIR}/${OUTPUTS_DIR}/${DATASET_NAME}"
SCENES_DIR="${DATASET_DIR}/scenes"

if [[ ! -d "${SCENES_DIR}" ]]; then
  echo "Missing scenes directory: ${SCENES_DIR}"
  exit 1
fi

if [[ "${MODE}" == "full" ]]; then
  ARCHIVE_NAME="${DATASET_NAME}_full.tar.gz"
else
  ARCHIVE_NAME="${DATASET_NAME}_done.tar.gz"
fi
ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"

excludes=()

cd "${ROOT_DIR}"

if [[ "${MODE}" != "full" ]]; then
  shopt -s nullglob

  for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    from=$((START_INDEX + SCENE_NUM * gpu))
    to=$((from + SCENE_NUM - 1))
    max_num=-1
    max_scene=""

    for scene_dir in "${SCENES_DIR}"/scene_*; do
      [[ -d "${scene_dir}" ]] || continue
      scene_name="$(basename "${scene_dir}")"

      if [[ "${scene_name}" =~ ^scene_([0-9]+)$ ]]; then
        scene_num=$((10#${BASH_REMATCH[1]}))
        if ((scene_num >= from && scene_num <= to && scene_num > max_num)); then
          max_num="${scene_num}"
          max_scene="${scene_name}"
        fi
      fi
    done

    if ((max_num >= 0)); then
      excludes+=(--exclude="${DATASET_NAME}/scenes/${max_scene}")
      excludes+=(--exclude="${DATASET_NAME}/scenes/preview/${max_scene}_*.png")
      echo "[GPU ${gpu}] ${from}-${to}: exclude ${max_scene} and preview pngs"
    else
      echo "[GPU ${gpu}] ${from}-${to}: no scene found, nothing to exclude"
    fi
  done
else
  echo "full mode: no current scene exclusions"
fi

mkdir -p "${ARCHIVE_DIR}"
echo "Creating archive: ${ARCHIVE_PATH}"
tar -C "${OUTPUTS_DIR}" "${excludes[@]}" -czf "${ARCHIVE_PATH}" "${DATASET_NAME}"
if [[ -n "${ARCHIVE_PATH_FILE:-}" ]]; then
  printf '%s\n' "${ARCHIVE_PATH}" > "${ARCHIVE_PATH_FILE}"
fi
echo "Done: ${ARCHIVE_PATH}"
