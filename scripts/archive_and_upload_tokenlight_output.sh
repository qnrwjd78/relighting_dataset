#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 GPU_COUNT START_INDEX SCENE_NUM RCLONE_DEST [full]"
  echo
  echo "Examples:"
  echo "  $0 4 500 250 gdrive:datasets/"
  echo "  $0 4 500 250 gdrive:datasets/ full"
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 1
fi

GPU_COUNT="$1"
START_INDEX="$2"
SCENE_NUM="$3"
RCLONE_DEST="$4"
MODE="${5:-partial}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone was not found in PATH."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE_PATH_FILE="$(mktemp)"
trap 'rm -f "${ARCHIVE_PATH_FILE}"' EXIT

ARCHIVE_PATH_FILE="${ARCHIVE_PATH_FILE}" \
  "${SCRIPT_DIR}/archive_tokenlight_output.sh" \
  "${GPU_COUNT}" "${START_INDEX}" "${SCENE_NUM}" "${MODE}"

ARCHIVE_PATH="$(cat "${ARCHIVE_PATH_FILE}")"
if [[ "${ARCHIVE_PATH}" != /* ]]; then
  ARCHIVE_PATH="${ROOT_DIR}/${ARCHIVE_PATH}"
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Archive was not created: ${ARCHIVE_PATH}"
  exit 1
fi

echo "Uploading archive to: ${RCLONE_DEST}"
rclone copy "${ARCHIVE_PATH}" "${RCLONE_DEST}" --progress
echo "Upload done: ${RCLONE_DEST}"
