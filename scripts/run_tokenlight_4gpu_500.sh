#!/usr/bin/env bash
set -u

CONFIG="${1:-configs/tokenlight_synthetic_full.json}"
OUTPUT_PREFIX="${2:-outputs/objaverse_dataset_exr}"
ONLY="${3:-spatial}"
COMPONENT_FORMAT="${4:-exr}"

SCENES_PER_GPU="${SCENES_PER_GPU:-500}"
GPUS="${GPUS:-0 1 2 3}"
GLOBAL_START="${GLOBAL_START:-0}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-960}"
SAMPLES="${SAMPLES:-32}"
HDRI_MODE="${HDRI_MODE:-on}"
AMBIENT_SOURCE="${AMBIENT_SOURCE:-hdri}"
POINT_LIGHT_MODE="${POINT_LIGHT_MODE:-component}"
POSITIONS_PER_SCENE="${POSITIONS_PER_SCENE:-64}"
GLOBAL_DIFFUSE="${GLOBAL_DIFFUSE:-1}"
PER_LIGHT_DIFFUSE="${PER_LIGHT_DIFFUSE:-0}"
LIGHT_PREVIEW="${LIGHT_PREVIEW:-1}"
FLAT_OUTPUT="${FLAT_OUTPUT:-0}"
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
read -r -a GPU_LIST <<< "$GPUS"
if [[ "$FLAT_OUTPUT" == "1" || "$FLAT_OUTPUT" == "true" ]]; then
  if ((${#GPU_LIST[@]} != 1)); then
    echo "[ERROR] FLAT_OUTPUT=1 writes directly to OUTPUT_PREFIX and requires exactly one GPU. Set GPUS=0 or FLAT_OUTPUT=0." >&2
    exit 1
  fi
fi

EXTRA_RELIGHTING_ARGS=()
if [[ "$GLOBAL_DIFFUSE" == "1" || "$GLOBAL_DIFFUSE" == "true" ]]; then
  EXTRA_RELIGHTING_ARGS+=(--global-diffuse)
fi
if [[ "$PER_LIGHT_DIFFUSE" == "1" || "$PER_LIGHT_DIFFUSE" == "true" ]]; then
  EXTRA_RELIGHTING_ARGS+=(--per-light-diffuse)
fi
if [[ -n "$POSITIONS_PER_SCENE" ]]; then
  EXTRA_RELIGHTING_ARGS+=(--positions-per-scene "$POSITIONS_PER_SCENE")
fi
if [[ "$LIGHT_PREVIEW" == "1" || "$LIGHT_PREVIEW" == "true" ]]; then
  EXTRA_RELIGHTING_ARGS+=(--light-preview)
fi

cleanup() {
  echo "[STOP] killing running Blender jobs..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
} 
trap cleanup INT TERM

for idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$idx]}"
  start=$((GLOBAL_START + idx * SCENES_PER_GPU))
  end=$((start + SCENES_PER_GPU - 1))
  if [[ "$FLAT_OUTPUT" == "1" || "$FLAT_OUTPUT" == "true" ]]; then
    out_dir="$OUTPUT_PREFIX"
  else
    out_dir="${OUTPUT_PREFIX}/cuda${gpu}_scenes_${start}_${end}"
  fi
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
    "${EXTRA_RELIGHTING_ARGS[@]}" \
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
