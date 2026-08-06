# Relighting Dataset

Objaverse와 BlenderKit asset으로 relighting 데이터셋과 shadow mask를 생성하는 코드입니다.

## Directory

```text
scripts/
  download/                         asset 다운로드, 준비, preview 코드
  render_objaverse_random_dataset.py
  render_objaverse_fixed_dataset.py
  render_objaverse_fixed_multi_gpu.py  fixed renderer multi-GPU launcher
  render_blenderkit_dataset.py
  relighting_mask_pipeline.py       fixed renderer가 import하는 mask 코드
  convert_exr_dataset_to_png.py      완료 scene EXR을 PNG로 변환
  precompute_wan_vae_cache.py        PNG dataset Wan VAE latent cache
  wan_vae2_2.py                      cache용 Wan2.2 VAE encoder 구현
configs/                            renderer 설정
data/                               다운로드된 GLB, HDRI, texture, weight
metadata/                           asset manifest, index, 다운로드 report
outputs/                            preview 이미지와 최종 렌더 결과
```

기존 `dataset/` 디렉터리는 `scripts/download/`으로 이동했습니다.

## Docker

모든 다운로드와 렌더 명령은 Docker 안의 `/workspace`에서 실행합니다.

```bash
docker exec -it jaeho_relight_dataset bash
cd /workspace
```

Objaverse downloader 의존성이 없다면 한 번 설치합니다.

```bash
python3 -m pip install -U objaverse pandas pyarrow tqdm fsspec
```

## Objaverse Download

앞에서부터 Sketchfab GLB 2,000개를 다운로드하고 renderer용 manifest를 생성합니다.

```bash
python3 scripts/download/object/download_objaverse_xl.py \
  --source sketchfab \
  --file-types glb \
  --start 0 \
  --limit 2000 \
  --processes 8 \
  --numbered-mode none \
  --download-dir data/objaverse_xl \
  --report-out metadata/objaverse_xl/reports/front2000 \
  --write-manifest metadata/objaverse_xl/front2000_objects.txt
```

주요 결과는 다음과 같습니다.

```text
data/objaverse_xl/                                      실제 GLB와 annotation cache
metadata/objaverse_xl/reports/front2000/download_manifest.json 다운로드 결과
metadata/objaverse_xl/front2000_objects.txt                    renderer 입력 manifest
```

다운로드 수를 확인합니다.

```bash
wc -l metadata/objaverse_xl/front2000_objects.txt
sha256sum metadata/objaverse_xl/front2000_objects.txt
```

다른 서버에서 동일한 object와 index 순서를 보장하려면 첫 서버의
`metadata/objaverse_xl/front2000_objects.txt`를 복사한 뒤 exact selection으로 다운로드합니다.

```bash
python3 scripts/download/object/download_objaverse_xl.py \
  --source sketchfab \
  --file-types glb \
  --selection-manifest metadata/objaverse_xl/front2000_objects.txt \
  --processes 8 \
  --numbered-mode none \
  --download-dir data/objaverse_xl \
  --report-out metadata/objaverse_xl/reports/front2000 \
  --write-manifest metadata/objaverse_xl/front2000_objects.txt
```

단순히 `--start 0 --limit 2000`을 반복하는 방식은 Objaverse annotation snapshot이
바뀌면 선택 결과가 달라질 수 있으므로 서버 간 분산 작업에는 exact selection을 사용합니다.

## HDRI Download

다섯 category에서 30개씩 총 150개의 Poly Haven HDRI를 받습니다.

```bash
python3 scripts/download/hdri/download_polyhaven_hdri.py \
  --categories studio indoor outdoor urban nature \
  --per-category 30 \
  --resolution 2k \
  --format hdr \
  --out-dir data/polyhaven_hdri \
  --manifest metadata/polyhaven_hdri/polyhaven_hdri_hdris.txt \
  --metadata-out metadata/polyhaven_hdri/polyhaven_hdri_index.json
```

receiver에 Poly Haven texture를 사용하려면 추가로 받습니다. 이 단계는 선택 사항입니다.

```bash
python3 scripts/download/object/download_polyhaven_textures.py \
  --resolution 2k \
  --format jpg \
  --per-category 20
```

다운로드가 끝나면 random/fixed renderer 모두 config의
`metadata/polyhaven_textures/polyhaven_textures.json`을 자동으로 읽습니다.
바닥은 config의 `receiver_texture_probability` 확률로 image texture를 사용하고,
벽은 `wall_texture_probability` 설정을 따릅니다. Texture를 끄려면 renderer 명령에
`--no-receiver-textures`를 추가합니다. 다른 manifest는
`--receiver-texture-manifest PATH`로 지정할 수 있습니다.

## Random Objaverse Render

Objaverse asset마다 position, color, power를 config 설정에 따라 랜덤하게 생성합니다.

```bash
CUDA_VISIBLE_DEVICES=0 blender -b \
  --python scripts/render_objaverse_random_dataset.py -- \
  --config configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json \
  --output outputs/objaverse_random \
  --start-index 0 \
  --max-scenes 2000 \
  --object-min-size 0.6 \
  --object-target-size 0.9 \
  --rig-reference-object-size 1.2 \
  --canonical-world-scale 0.75 \
  --resolution 480 \
  --samples 16 \
  --component-format exr \
  --hdri-mode on \
  --pbr \
  --only all \
  --fail-fast
```

이 renderer는 config의 다음 manifest를 사용합니다.

```text
metadata/objaverse_xl/front2000_objects.txt
metadata/polyhaven_hdri/polyhaven_hdri_hdris.txt
metadata/polyhaven_textures/polyhaven_textures.json  선택 사항
```

## Fixed Objaverse Render

고정 renderer의 설정은 다음과 같습니다.

- 카메라 기준 canonical cube의 위쪽 절반에 `4 x 4 x 2` 조명 후보 32개
- power는 `0.30, 0.60, 0.90, 1.20` 중 하나로 고정 배정
- point light color는 흰색
- 유효한 조명 위치만 저장
- scene마다 HDRI 하나를 선택하여 source, light render, mask render에 동일하게 사용
- ambient-subtracted local shadow ratio mask 생성

먼저 scene 하나만 확인합니다.

```bash
CUDA_VISIBLE_DEVICES=0 blender -b \
  --python scripts/render_objaverse_fixed_dataset.py -- \
  --object-manifest metadata/objaverse_xl/front2000_objects.txt \
  --object-offset 0 \
  --object-limit 1 \
  --object-min-size 0.6 \
  --object-target-size 0.9 \
  --rig-reference-object-size 1.2 \
  --canonical-world-scale 0.75 \
  --output-root outputs/fixed32_test \
  --resolution 480 \
  --samples 16 \
  --component-format exr \
  --gpu-devices 0 \
  --fixed-upper-half-white-grid \
  --power-values 0.30 0.60 0.90 1.20 \
  --ambient-subtracted-shadow-ratio \
  --shadow-threshold 0.05 \
  --shadow-support-threshold 0.001 \
  --fail-fast
```

2,000 scene을 한 terminal에서 GPU 네 장으로 실행합니다. Launcher가 범위를 균등 분할하고 GPU별 Blender process를 생성하며, 각 process의 출력은 현재 terminal에 그대로 표시됩니다. 유효한 조명이 0개인 scene은 failed로 기록하고 다음 scene으로 진행합니다.

```bash
python3 scripts/render_objaverse_fixed_multi_gpu.py \
  --gpus 0 1 2 3 \
  --object-count 2000 \
  --output-root outputs/front2000_fixed32_size06_09_texture_exr_shards
```

현재 설정은 launcher 기본값이므로 위 명령만으로 object 크기 `0.6~0.9`, rig 기준 `1.2`, canonical world scale `0.75`, `480x480`, 16 samples, EXR, fixed 32개 흰색 point light와 네 power를 사용합니다. 재실행하면 정상 `meta.json`이 있는 scene은 자동으로 건너뜁니다. 처음부터 다시 렌더링하려면 `--no-resume`을 추가합니다.

Launcher는 내부적으로 `CUDA_VISIBLE_DEVICES`로 각 process에 GPU 한 장만 노출하고 Blender worker에는 `--gpu-devices 0`을 전달합니다. 완료 후 output root의 `dataset_manifest.json`에 전체 성공, 실패, 누락 수를 기록합니다.

조명 감쇠와 무관한 object-only geometry shadow/direct-lit mask를 사용하려면 다음 옵션을 추가합니다. `object-mask-erode-radius 1`은 direct-lit 계산에서 object 경계를 안쪽으로 1px 줄이고, `min-area 0`은 작은 component를 제거하지 않으며, `pad-radius 2`는 shadow 경계만 작게 확장합니다. Geometry mode는 mask용 white render를 만들지 않습니다.

```bash
--shadow-mask-mode geometry-ray \
--object-mask-erode-radius 1 \
--min-area 0 \
--pad-radius 2
```

여러 서버가 서로 다른 범위를 맡을 때는 global `--start-index`와 `--object-count`를 지정합니다. 두 서버의 manifest SHA256이 같은지 launcher 시작 출력에서 확인합니다.

```bash
# Server A: scene_000000 ~ scene_000999
python3 scripts/render_objaverse_fixed_multi_gpu.py \
  --gpus 0 1 2 3 \
  --start-index 0 \
  --object-count 1000 \
  --output-root outputs/front2000_fixed_part_0000_0999

# Server B: scene_001000 ~ scene_001999
python3 scripts/render_objaverse_fixed_multi_gpu.py \
  --gpus 0 1 2 3 \
  --start-index 1000 \
  --object-count 1000 \
  --output-root outputs/front2000_fixed_part_1000_1999
```

## BlenderKit Download And Render

API key를 환경 변수로 설정한 다음 scene index와 preview를 만듭니다.

```bash
export BLENDERKIT_API_KEY="$(tr -d '\r\n' < blenderkit_key.txt)"

python3 scripts/download/scene/preview_blenderkit.py \
  --target-count 2000 \
  --asset-type scene \
  --free-only \
  --show-subprocess-output
```

주요 결과는 다음과 같습니다.

```text
outputs/previews/blenderkit/blenderkit_index.json
outputs/previews/blenderkit/img/
outputs/previews/blenderkit/metadata/
```

classification 파일이 있으면 선택된 category만 사용합니다. 파일이 없으면 `blenderkit_index.json`의 모든 scene을 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/render_blenderkit_dataset.py \
  --index-json outputs/previews/blenderkit/blenderkit_index.json \
  --classification outputs/previews/blenderkit/blenderkit_scene_use_classification.txt \
  --api-key-file blenderkit_key.txt \
  --config configs/tokenlight_synthetic_full_ratio3p5_cube1p6.json \
  --output outputs/blenderkit_dataset \
  --start 0 \
  --limit 2000 \
  --width 480 \
  --height 480 \
  --samples 16 \
  --component-format exr \
  --ambient-source hdri \
  --hdri-mode on \
  --pbr \
  --skip-existing
```

## Output

Objaverse fixed output은 다음 형태입니다.

```text
outputs/fixed32_test/
  dataset_manifest.json
  scenes/
    scene_000000/
      meta.json
      source.exr
      pbr/depth.exr
      pbr/normal.exr
      masks/object_mask.png
      samples/position/position_000.exr
      samples/position/position_000_masks/
```

mask 생성 구현은 `scripts/relighting_mask_pipeline.py`에 있으며 fixed renderer가 import해서 사용합니다.

## EXR To PNG

`meta.json`이 있는 완료 scene만 변환합니다. `failed_scenes/`와 `meta.json`이 없는 미완료 scene은 자동으로 무시합니다.

4-GPU shard 결과를 하나의 `scenes/` 디렉터리로 통합하면서 변환합니다.

```bash
python3 scripts/convert_exr_dataset_to_png.py \
  --input outputs/front2000_fixed32_ratio_exr_shards \
  --output outputs/front2000_fixed32_ratio_png \
  --gpu-shard \
  --workers 16
```

입력이 shard 구조가 아니라 바로 `scenes/`를 포함하면 `--gpu-shard`를 빼면 됩니다.

```bash
python3 scripts/convert_exr_dataset_to_png.py \
  --input outputs/fixed32_test \
  --output outputs/fixed32_test_png \
  --workers 8
```

기본 설정은 mask 생성에 사용한 `position_*_white/`와 `mask_reference/ambient_white/`를 제외합니다. 이 중간 렌더도 PNG로 포함하려면 다음 옵션을 추가합니다.

```bash
--with-white
```

변환 규칙은 다음과 같습니다.

- source, position, white 조명 EXR: Reinhard tone mapping과 gamma 2.2
- depth EXR: scene별 1%~99% depth 정규화, 가까운 영역이 밝게 저장
- normal EXR: `[-1, 1]` 값을 `[0, 1]`로 인코딩
- albedo와 roughness EXR: `[0, 1]` 범위 PNG로 인코딩
- 기존 mask PNG와 NPY: 상대 경로를 유지하여 hardlink, 불가능하면 copy
- metadata: 변환된 EXR 경로와 key를 PNG 기준으로 갱신

먼저 대상 scene 수만 확인하려면 `--dry-run`을 사용합니다.

```bash
python3 scripts/convert_exr_dataset_to_png.py \
  --input outputs/front2000_fixed32_ratio_exr_shards \
  --output outputs/front2000_fixed32_ratio_png \
  --gpu-shard \
  --dry-run
```

## LGI Map Generation

Fixed-grid scene의 metric depth EXR, camera metadata, world-space light position으로
각 유효 조명의 3-channel LGI map을 생성합니다. `depth.png`는 시각화용으로
정규화되어 있으므로 입력으로 사용하지 않습니다.

```bash
python3 scripts/generate_lgi_maps.py \
  --scene-dir outputs/final_objaverse_front2000_480_exr_s16_fixed32x4_hdri_ratio/scenes/scene_000043 \
  --output-dir outputs/lgi_scene_000043_fixed32
```

각 `position_NNN.npz`에는 radian 단위 `lgi [3,H,W]`, `valid [H,W]`,
`min_abs [H,W]`, `hard [H,W]`, camera/light 좌표가 저장됩니다. 기본 ray sample
수는 16이고 hard candidate threshold는 5도입니다. `index.json`은 조명별 통계와
Blender GT shadow 비교를, `hard_overview.png`는 전체 fixed-grid 결과를 담습니다.

조명 하나만 확인하려면 `--position-id 0`을 추가합니다.

데이터셋의 모든 scene을 처리하려면 dataset root와 별도 output root를 지정합니다.
완료된 scene은 기본적으로 건너뛰므로 같은 명령으로 이어서 실행할 수 있습니다.

```bash
python3 scripts/generate_lgi_maps.py \
  --dataset-root outputs/final_objaverse_front2000_480_exr_s16_fixed32x4_hdri_ratio \
  --output-root outputs/final_objaverse_front2000_480_lgi_fixed32 \
  --workers 4
```

결과는 `OUTPUT_ROOT/scenes/scene_NNNNNN/`에 scene별로 저장되며 전체 처리 결과는
`OUTPUT_ROOT/dataset_index.json`에 기록됩니다. 기존 출력을 다시 만들려면
`--overwrite`를 추가합니다. worker 하나가 scene 하나를 처리하므로 `--workers`는
사용 가능한 CPU와 RAM에 맞춰 지정합니다.

학습 데이터 scene 내부에 LGI 채널을 각각 저장하려면 in-place 모드를 사용합니다.

```bash
python3 scripts/generate_lgi_maps.py \
  --dataset-root outputs/final_objaverse_front2000_480_exr_s16_fixed32x4_hdri_ratio \
  --in-place \
  --channel-files \
  --workers 4
```

이 모드는 각 scene에 `position_00/min.npy`, `max.npy`, `nearest.npy`를 저장합니다.
`valid.npy`, `min_abs.npy`, `hard.npy`, `camera_light.npz`, `preview.png`도 같은
position 디렉터리에 저장하며 scene-level metadata는 `lgi_index.json`입니다.

## Wan VAE Luminance Cache

`scripts/precompute_wan_vae_cache.py`는 RGB PNG를 luminance로 변환하고 3채널로
복제한 뒤 Wan2.2 VAE encoder에 넣습니다. Scene마다 `source_latent`와
`sample_latents`를 하나의 `.pt` 파일로 저장합니다. 현재 fixed renderer의
`meta.json` 구조와 예전 `samples_manifest.json` 구조를 모두 지원합니다.

Wan encoder 코드는 `scripts/wan_vae2_2.py`에 포함되어 있습니다. 전체 Wan 모델 대신 VAE encoder weight 하나만 다운로드합니다.

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir data/weights/Wan2.2-TI2V-5B
```

공식 weight SHA-256:

```text
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
```

Fixed PNG dataset을 GPU 네 장으로 luminance cache 처리합니다.

```bash
DATASET=outputs/front2000_fixed32_size09_texture_png
CACHE=outputs/front2000_fixed32_size09_texture_wanvae_luminance_480_cache

PIDS=()
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/precompute_wan_vae_cache.py \
    --dataset-root "$DATASET" \
    --ckpt-dir data/weights/Wan2.2-TI2V-5B \
    --out-dir "$CACHE" \
    --resolution 480 \
    --image-transform luminance \
    --batch-size 4 \
    --dtype bf16 \
    --device cuda \
    --num-shards 4 \
    --shard-id "$GPU" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
test "$STATUS" -eq 0
```

완료 scene 수를 확인합니다.

```bash
find "$CACHE/scenes" -name '*.pt' | wc -l
```
