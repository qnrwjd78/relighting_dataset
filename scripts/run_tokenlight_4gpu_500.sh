#!/usr/bin/env bash
set -u

CONFIG="${1:-configs/tokenlight_synthetic_full.json}"
OUTPUT_PREFIX="${2:-outputs/objaverse_500/objaverse_xl}"
ONLY="${3:-spatial}"
COMPONENT_FORMAT="${4:-png}"

SCENES_PER_GPU="${SCENES_PER_GPU:-500}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-960}"
SAMPLES="${SAMPLES:-32}"
HDRI_MODE="${HDRI_MODE:-on}"
AMBIENT_SOURCE="${AMBIENT_SOURCE:-hdri}"
POINT_LIGHT_MODE="${POINT_LIGHT_MODE:-component}"
BLENDER_CMD="${BLENDER_CMD:-blender}"
LOG_DIR="${LOG_DIR:-logs/tokenlight_4gpu_500}"

mkdir -p "$OUTPUT_PREFIX"
if ! mkdir -p "$LOG_DIR" 2>/dev/null || [[ ! -w "$LOG_DIR" ]]; then
  fallback_log_dir="${LOG_DIR}_${USER:-user}"
  echo "[WARN] LOG_DIR=${LOG_DIR} is not writable; using ${fallback_log_dir}" >&2
  LOG_DIR="$fallback_log_dir"
  mkdir -p "$LOG_DIR"
fi

pids=()
names=()

cleanup() {
  echo "[STOP] killing running Blender jobs..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup INT TERM

for gpu in 0 1 2 3; do
  start=$((gpu * SCENES_PER_GPU))
  end=$((start + SCENES_PER_GPU - 1))
  out_dir="${OUTPUT_PREFIX}/cuda${gpu}_scenes_${start}_${end}"
  log_file="${LOG_DIR}/cuda${gpu}_scenes_${start}_${end}.log"

  mkdir -p "$out_dir"
  echo "[RUN] cuda=${gpu} scenes=${start}-${end} out=${out_dir}"

  CUDA_VISIBLE_DEVICES="$gpu" "$BLENDER_CMD" -b --python scripts/render_object_relighting.py -- \
    --config "$CONFIG" \
    --output "$out_dir" \
    --start-index "$start" \
    --max-scenes "$SCENES_PER_GPU" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --samples "$SAMPLES" \
    --component-format "$COMPONENT_FORMAT" \
    --only "$ONLY" \
    --ambient-source "$AMBIENT_SOURCE" \
    --point-light-mode "$POINT_LIGHT_MODE" \
    --hdri-mode "$HDRI_MODE" \
    > "$log_file" 2>&1 &

  pids+=("$!")
  names+=("cuda${gpu}:${start}-${end}")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  if wait "$pid"; then
    echo "[DONE] $name"
  else
    code=$?
    echo "[FAILED] $name exit=${code}"
    status=1
  fi
done

trap - INT TERM
echo "[ALL DONE] status=${status}"
echo "[LOGS] ${LOG_DIR}"
exit "$status"
