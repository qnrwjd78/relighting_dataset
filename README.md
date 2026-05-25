# TokenLight Synthetic Dataset Builder

TokenLight 논문 기준의 synthetic dataset 생성 파이프라인입니다. 실제 real capture dataset은 제외하고, 공개 논문에 명시된 synthetic 구성만 코드화했습니다.

구현 범위:

- Spatial virtual point light: scene당 기본 64개 camera-relative 3D light position component render
- Ambient scaling: HDRI/environment render를 dataloader 단계에서 scale
- Global diffuse/spread: constant ambient + dominant area light 6개 spread level render
- Visible fixture synthetic: annotated indoor `.blend` scene의 fixture mask/contribution render
- Linear RGB EXR component 저장 후, 학습 pair는 Python에서 Reinhard tone mapping 전 linear 합성
- Canonical camera/light coordinate와 similarity-style scaling metadata 저장

논문과 동일하게 에셋 자체는 외부에서 채워 넣어야 합니다.

- Object pool: Objaverse/GLB/OBJ/FBX/STL/PLY 등
- HDRI pool: PolyHaven HDRI, 논문은 약 600개 사용
- Visible fixture scenes: artist-authored indoor `.blend`와 fixture annotation

## 설치

일반 Python 쪽:

```powershell
python -m pip install -r requirements.txt
```

Blender는 별도 설치가 필요합니다. `blender`가 PATH에 없으면 실행 시 `--blender-exe` 또는 `BLENDER_EXE`를 지정하세요.

## 에셋 manifest

예시 파일을 복사해서 실제 경로를 채우면 됩니다.

```powershell
Copy-Item manifests\objects.example.txt manifests\objects.txt
Copy-Item manifests\hdris.example.txt manifests\hdris.txt
Copy-Item manifests\fixture_scenes.example.jsonl manifests\fixture_scenes.jsonl
```

상대경로 규칙: config, manifest, `--dataset`, `--out`에 적은 상대경로는 모두 이 `tokenlight_dataset` 폴더 기준으로 해석됩니다. 예를 들어 `assets/objects/000001.glb`는 `tokenlight_dataset/assets/objects/000001.glb`를 뜻합니다.

`objects.txt`는 한 줄에 하나의 3D asset 경로입니다. 비워두면 Blender 기본 primitive로 렌더링 검증을 합니다.

`hdris.txt`는 한 줄에 하나의 `.hdr`/`.exr` HDRI 경로입니다. 비워두면 constant world color fallback을 씁니다.

`fixture_scenes.jsonl`은 visible fixture용입니다. 각 줄은 다음 형태입니다.

```json
{"scene_id":"room_000","blend_path":"assets/fixture_scenes/room_000.blend","camera":"Camera","fixtures":[{"id":"lamp_0","prefixes":["TL_FIXTURE_lamp_0"],"light_prefixes":["TL_LIGHT_lamp_0"]}]}
```

fixture geometry/material/light object 이름에 prefix를 붙여두면 renderer가 contribution과 mask를 분리합니다.

추천 asset 배치는 다음과 같습니다.

```text
assets/
  objects/
    000001.glb
  polyhaven/
    studio_small_09_4k.exr
  fixture_scenes/
    room_000.blend
```

## 렌더링

논문 재현 기본값은 `configs/tokenlight_synthetic_full.json`에 있습니다. 해상도 기본값은 960이고, spatial point position은 64개, diffuse spread는 6개입니다.

```powershell
python scripts\run_blender_batch.py --config configs\tokenlight_synthetic_full.json --max-scenes 1
```

Blender 경로 직접 지정:

```powershell
python scripts\run_blender_batch.py --blender-exe "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe" --config configs\tokenlight_synthetic_full.json --max-scenes 1
```

빠른 디버그:

```powershell
python scripts\run_blender_batch.py --config configs\tokenlight_synthetic_full.json --max-scenes 1 --resolution 256 --samples 32
```

## Pair/preview 합성

Blender 렌더링 후 EXR component에서 PNG preview와 `pairs.jsonl`을 만듭니다.

```powershell
python scripts\synthesize_pairs.py --dataset outputs\tokenlight_synthetic --out outputs\previews --mode all --count 32
```

모든 합성은 linear RGB에서 먼저 수행하고 마지막에 Reinhard tone mapping을 적용합니다.

학습 코드에서는 Dataset 클래스를 바로 쓸 수 있습니다.

```python
from tokenlight_dataset import TokenLightComponentDataset

dataset = TokenLightComponentDataset(
    "outputs/tokenlight_synthetic",
    modes=("spatial", "ambient", "diffuse", "fixture"),
    length=100_000,
    max_lights=3,
    return_torch=True,
)

sample = dataset[0]
input_tensor = sample["input"]    # CHW, [-1, 1]
target_tensor = sample["target"]  # CHW, [-1, 1]
condition = sample["condition"]
```

## 출력 구조

```text
outputs/tokenlight_synthetic/
  dataset_manifest.json
  scenes/
    scene_000000/
      meta.json
      masks/
        object_mask.png
      spatial/
        ambient.exr
        point_lights/
          light_000.exr
          ...
      diffuse/
        ambient_constant.exr
        spread_000.exr
        ...
      fixtures/
        environment.exr
        fixture_lamp_0/
          contribution.exr
          mask.png
```

## 논문 기준 메모

TokenLight는 Blender/Cycles로 synthetic scenes를 렌더링하고, Objaverse 기반 object-centric scenes, optional ground/wall geometry, PolyHaven HDRI pool, scene당 64개 point-light position, diffuse control용 6개 area-light spread render, visible fixture contribution/mask render를 사용합니다. 논문은 rendered images를 denoise 후 linear RGB로 저장하고, 학습 pair는 data loading 중 component를 linear sum한 뒤 Reinhard tone mapping합니다.

이 repo는 그 절차를 재현하는 generator입니다. 논문 저자 내부의 filtered Objaverse subset, 83개 artist-authored indoor scenes, exact hidden asset curation은 포함되어 있지 않으므로 manifest로 주입하도록 설계했습니다.
