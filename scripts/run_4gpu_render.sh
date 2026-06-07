#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 START_INDEX SCENE_NUM [extra render_components.py args...]"
  echo "Example: $0 2000 250 --light-preview"
  exit 1
fi

START_INDEX="$1"
SCENE_NUM="$2"
shift 2

if ! [[ "$START_INDEX" =~ ^[0-9]+$ && "$SCENE_NUM" =~ ^[0-9]+$ ]]; then
  echo "START_INDEX and SCENE_NUM must be non-negative integers."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --config configs/tokenlight_synthetic_full.json
  --width 1280
  --height 704
  --samples 512
  --only all
)

cd "${ROOT_DIR}"

pids=()
for gpu in 0 1 2 3; do
  gpu_start=$((START_INDEX + gpu * SCENE_NUM))
  gpu_end=$((gpu_start + SCENE_NUM - 1))
  log_path="${LOG_DIR}/render_gpu${gpu}_${gpu_start}_${gpu_end}.log"

  echo "[GPU ${gpu}] scene_${gpu_start} ~ scene_${gpu_end} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" blender -b --python scripts/render_components.py -- \
    "${COMMON_ARGS[@]}" \
    --start-index "${gpu_start}" \
    --max-scenes "${SCENE_NUM}" \
    "$@" > "${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  echo "All GPU render jobs finished."
else
  echo "At least one GPU render job failed. Check logs in ${LOG_DIR}."
fi

exit "${status}"
