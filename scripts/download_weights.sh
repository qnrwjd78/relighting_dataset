#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/weights"
MOGE_DIR="$WEIGHTS_DIR/moge-3-vitg"
WAN_DIR="$WEIGHTS_DIR/Wan2.2-TI2V-5B"

MOGE_FILE="$MOGE_DIR/model.pt"
WAN_FILE="$WAN_DIR/Wan2.2_VAE.pth"
MOGE_SHA256="ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c"
WAN_SHA256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install it with: python3 -m pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "$MOGE_DIR" "$WAN_DIR"
export HF_XET_HIGH_PERFORMANCE=1

if [[ ! -s "$MOGE_FILE" ]]; then
  hf download Ruicheng/moge-3-vitg model.pt --local-dir "$MOGE_DIR"
else
  echo "[weights] MoGe-3 already present: $MOGE_FILE"
fi

if [[ ! -s "$WAN_FILE" ]]; then
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$WAN_DIR"
else
  echo "[weights] Wan2.2 VAE already present: $WAN_FILE"
fi

printf '%s  %s\n' "$MOGE_SHA256" "$MOGE_FILE" | sha256sum --check --status
printf '%s  %s\n' "$WAN_SHA256" "$WAN_FILE" | sha256sum --check --status

echo "[weights] verified MoGe-3: $MOGE_FILE"
echo "[weights] verified Wan2.2 VAE: $WAN_FILE"
