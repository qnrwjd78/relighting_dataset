#!/usr/bin/env bash
set -u -o pipefail

CONFIG="${CONFIG:-configs/tokenlight_synthetic_full.json}"
CLASSIFICATION="${CLASSIFICATION:-outputs/previews/blenderkit/blenderkit_scene_use_classification.txt}"
INDEX_JSON="${INDEX_JSON:-outputs/previews/blenderkit/blenderkit_index.json}"
CATEGORIES="${CATEGORIES:-single_scene_light_good background_good_for_portrait_or_object}"

OUTPUT="${OUTPUT:-outputs/blenderkit_dataset}"
PREVIEW_DIR="${PREVIEW_DIR:-outputs/previews/blenderkit_dataset}"
WORK_DIR="${WORK_DIR:-outputs/work/blenderkit_dataset}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-data/blenderkit_spatial_cache}"
LOG_DIR="${LOG_DIR:-logs/blenderkit_spatial_4gpu}"

GPUS="${GPUS:-0 1 2 3}"
GLOBAL_START="${GLOBAL_START:-0}"
MAX_ITEMS="${MAX_ITEMS:-0}"

WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-960}"
SAMPLES="${SAMPLES:-32}"
COMPONENT_FORMAT="${COMPONENT_FORMAT:-exr}"
AMBIENT_SOURCE="${AMBIENT_SOURCE:-scene}"
POINT_LIGHT_MODE="${POINT_LIGHT_MODE:-component}"
HDRI_MODE="${HDRI_MODE:-on}"
GLOBAL_DIFFUSE="${GLOBAL_DIFFUSE:-0}"
PER_LIGHT_DIFFUSE="${PER_LIGHT_DIFFUSE:-0}"
LIGHT_VOLUME_PLACEMENT="${LIGHT_VOLUME_PLACEMENT:-camera-framed}"
LIGHT_VOLUME_DEPTH_OVER_SCALE="${LIGHT_VOLUME_DEPTH_OVER_SCALE:-}"
SPATIAL_BBOX_MODE="${SPATIAL_BBOX_MODE:-auto}"
SUBJECT_CANDIDATE_COUNT="${SUBJECT_CANDIDATE_COUNT:-3}"
CANDIDATE_PADDING="${CANDIDATE_PADDING:-1.15}"
CANDIDATE_OUTLIER_FACTOR="${CANDIDATE_OUTLIER_FACTOR:-20.0}"
POSITIONS_PER_SCENE="${POSITIONS_PER_SCENE:-}"
LIGHT_PREVIEW="${LIGHT_PREVIEW:-1}"
DEBUG_PREVIEW="${DEBUG_PREVIEW:-0}"
if [[ "${DEBUG:-}" == "1" || "${DEBUG:-}" == "true" ]]; then
  DEBUG_PREVIEW="1"
fi
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_BLEND="${KEEP_BLEND:-0}"
OVERWRITE_BLEND="${OVERWRITE_BLEND:-0}"
SLEEP="${SLEEP:-0.2}"

BLENDER_CMD="${BLENDER_CMD:-blender}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
API_KEY_FILE="${API_KEY_FILE:-blenderkit_key.txt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a GPU_LIST <<< "$GPUS"
read -r -a CATEGORY_ARGS <<< "$CATEGORIES"
read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"

if ((${#GPU_LIST[@]} == 0)); then
  echo "[ERROR] GPUS is empty." >&2
  exit 1
fi

if [[ -z "${BLENDERKIT_API_KEY:-}" && ! -f "$API_KEY_FILE" ]]; then
  echo "[ERROR] Missing BlenderKit API key. Set BLENDERKIT_API_KEY or API_KEY_FILE." >&2
  exit 1
fi

mkdir -p "$OUTPUT" "$PREVIEW_DIR" "$WORK_DIR" "$DOWNLOAD_DIR"
if ! mkdir -p "$LOG_DIR" 2>/dev/null || [[ ! -w "$LOG_DIR" ]]; then
  fallback_log_dir="${LOG_DIR}_${USER:-user}"
  echo "[WARN] LOG_DIR=${LOG_DIR} is not writable; using ${fallback_log_dir}" >&2
  LOG_DIR="$fallback_log_dir"
  mkdir -p "$LOG_DIR"
fi

TOTAL_ITEMS=$(
  CLASSIFICATION="$CLASSIFICATION" INDEX_JSON="$INDEX_JSON" CATEGORIES="$CATEGORIES" "$PYTHON_CMD" - <<'PY'
import json
import os
import re
from pathlib import Path

classification = Path(os.environ["CLASSIFICATION"])
index_json = Path(os.environ["INDEX_JSON"])
categories = set(os.environ["CATEGORIES"].split())

ids = []
active = False
for raw in classification.read_text(encoding="utf-8").splitlines():
    match = re.match(r"\[\d+\]\s+([^\s]+)", raw.strip())
    if match:
        active = match.group(1) in categories
        continue
    if not active:
        continue
    match = re.match(r"- blenderkit_(\d+)", raw)
    if match:
        ids.append(match.group(1).zfill(5))

index = json.loads(index_json.read_text(encoding="utf-8"))
available = {str(item.get("id")).zfill(5) for item in index.get("items", [])}
print(sum(1 for item_id in ids if item_id in available))
PY
)

if ! [[ "$TOTAL_ITEMS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Failed to count selected items: ${TOTAL_ITEMS}" >&2
  exit 1
fi

REMAINING=$((TOTAL_ITEMS - GLOBAL_START))
if ((REMAINING <= 0)); then
  echo "[ERROR] GLOBAL_START=${GLOBAL_START} is outside selected item count ${TOTAL_ITEMS}." >&2
  exit 1
fi

if ((MAX_ITEMS > 0 && MAX_ITEMS < REMAINING)); then
  RUN_ITEMS="$MAX_ITEMS"
else
  RUN_ITEMS="$REMAINING"
fi

NUM_SHARDS="${#GPU_LIST[@]}"
CHUNK_SIZE=$(((RUN_ITEMS + NUM_SHARDS - 1) / NUM_SHARDS))

api_args=()
if [[ -z "${BLENDERKIT_API_KEY:-}" && -f "$API_KEY_FILE" ]]; then
  api_args+=(--api-key-file "$API_KEY_FILE")
fi

pids=()
names=()

cleanup() {
  echo "[STOP] killing running BlenderKit spatial jobs..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup INT TERM

echo "[INFO] selected_items=${TOTAL_ITEMS} start=${GLOBAL_START} run_items=${RUN_ITEMS} shards=${NUM_SHARDS} chunk=${CHUNK_SIZE}"
echo "[INFO] output=${OUTPUT}"
echo "[INFO] ambient_source=${AMBIENT_SOURCE} point_light_mode=${POINT_LIGHT_MODE} debug_preview=${DEBUG_PREVIEW} dry_run=${DRY_RUN} skip_existing=${SKIP_EXISTING}"

for shard in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard]}"
  start=$((GLOBAL_START + shard * CHUNK_SIZE))
  shard_end=$((GLOBAL_START + RUN_ITEMS))
  if ((start >= shard_end)); then
    continue
  fi
  limit="$CHUNK_SIZE"
  if ((start + limit > shard_end)); then
    limit=$((shard_end - start))
  fi

  shard_name="shard${shard}_gpu${gpu}_start${start}_limit${limit}"
  shard_preview_dir="${PREVIEW_DIR}/shard_${shard}"
  shard_work_dir="${WORK_DIR}/shard_${shard}"
  shard_download_dir="${DOWNLOAD_DIR}/shard_${shard}"
  log_file="${LOG_DIR}/${shard_name}.log"
  mkdir -p "$shard_preview_dir" "$shard_work_dir" "$shard_download_dir"

  cmd=(
    "$PYTHON_CMD" scripts/render_classified_blenderkit_spatial.py
    --classification "$CLASSIFICATION"
    --index-json "$INDEX_JSON"
    --categories "${CATEGORY_ARGS[@]}"
    --config "$CONFIG"
    --output "$OUTPUT"
    --preview-dir "$shard_preview_dir"
    --work-dir "$shard_work_dir"
    --download-dir "$shard_download_dir"
    --blender-cmd "$BLENDER_CMD"
    --start "$start"
    --limit "$limit"
    --width "$WIDTH"
    --height "$HEIGHT"
    --samples "$SAMPLES"
    --component-format "$COMPONENT_FORMAT"
    --ambient-source "$AMBIENT_SOURCE"
    --point-light-mode "$POINT_LIGHT_MODE"
    --hdri-mode "$HDRI_MODE"
    --light-volume-placement "$LIGHT_VOLUME_PLACEMENT"
    --spatial-bbox-mode "$SPATIAL_BBOX_MODE"
    --subject-candidate-count "$SUBJECT_CANDIDATE_COUNT"
    --candidate-padding "$CANDIDATE_PADDING"
    --candidate-outlier-factor "$CANDIDATE_OUTLIER_FACTOR"
    --sleep "$SLEEP"
    "${api_args[@]}"
  )

  if [[ -n "$LIGHT_VOLUME_DEPTH_OVER_SCALE" ]]; then
    cmd+=(--light-volume-depth-over-scale "$LIGHT_VOLUME_DEPTH_OVER_SCALE")
  fi
  if [[ -n "$POSITIONS_PER_SCENE" ]]; then
    cmd+=(--positions-per-scene "$POSITIONS_PER_SCENE")
  fi
  if [[ "$LIGHT_PREVIEW" == "1" || "$LIGHT_PREVIEW" == "true" ]]; then
    cmd+=(--light-preview)
  else
    cmd+=(--no-light-preview)
  fi
  if [[ "$DEBUG_PREVIEW" == "1" || "$DEBUG_PREVIEW" == "true" ]]; then
    cmd+=(--debug)
  fi
  if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" ]]; then
    cmd+=(--skip-existing)
  fi
  if [[ "$KEEP_BLEND" == "1" || "$KEEP_BLEND" == "true" ]]; then
    cmd+=(--keep-blend)
  fi
  if [[ "$OVERWRITE_BLEND" == "1" || "$OVERWRITE_BLEND" == "true" ]]; then
    cmd+=(--overwrite-blend)
  fi
  if [[ "$GLOBAL_DIFFUSE" == "1" || "$GLOBAL_DIFFUSE" == "true" ]]; then
    cmd+=(--global-diffuse)
  fi
  if [[ "$PER_LIGHT_DIFFUSE" == "1" || "$PER_LIGHT_DIFFUSE" == "true" ]]; then
    cmd+=(--per-light-diffuse)
  fi
  if ((${#EXTRA_ARGS_ARRAY[@]} > 0)); then
    cmd+=("${EXTRA_ARGS_ARRAY[@]}")
  fi

  echo "[RUN] ${shard_name} log=${log_file}"
  {
    printf '[CMD]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } > "$log_file"

  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    cat "$log_file"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >> "$log_file" 2>&1 &
  pids+=("$!")
  names+=("$shard_name")
done

if ((${#pids[@]} == 0)); then
  trap - INT TERM
  echo "[DRY RUN DONE] commands written to ${LOG_DIR}"
  exit 0
fi

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
