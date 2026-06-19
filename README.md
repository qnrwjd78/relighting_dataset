# Relighting Dataset

Synthetic relighting dataset tools for object/portrait assets and Blender scene assets.

## Layout

```text
dataset/hdri/      HDRI download_*.py and preview_*.py entrypoints
dataset/portrait/  portrait/human download_*.py and preview_*.py entrypoints
dataset/object/    object/material download_*.py and preview_*.py entrypoints
dataset/scene/     scene download_*.py and preview_*.py entrypoints
dataset/indoor/    indoor scene/HDRI download_*.py, prepare_*.py, and preview_*.py entrypoints
dataset/outdoor/   outdoor scene/HDRI download_*.py and preview_*.py entrypoints
dataset/utils/   shared util_*.py helpers
scripts/   render_*_relighting.py entrypoints
scripts/utils/   shared util_*.py helpers
data/      downloaded source assets and caches
outputs/   previews and final rendered datasets
configs/   render configs
manifests/ example manifests only
```

## Dataset Prep

Download bulk datasets into `data/{dataset}`:

```bash
python3 dataset/hdri/download_polyhaven_hdri.py --resolution 2k --format hdr --per-category 30
python3 dataset/object/download_polyhaven_textures.py --resolution 2k --format jpg --per-category 20
python3 dataset/object/download_objaverse_xl.py --limit 5000
python3 dataset/portrait/download_hsrd100.py --lod LOD1 --extract
```

Prepare indoor/outdoor scene sources:

```bash
python3 dataset/indoor/download_blenderkit_indoor.py --query "living room interior" --max-results 5
python3 dataset/outdoor/download_blenderkit_outdoor.py --query "night street" --max-results 5

python3 dataset/indoor/download_sketchfab_indoor.py --max-results 50 --extract
python3 dataset/outdoor/download_sketchfab_outdoor.py --max-results 50 --extract

python3 dataset/indoor/download_polyhaven_indoor_hdri.py --per-category 30
python3 dataset/outdoor/download_polyhaven_outdoor_hdri.py --per-category 30

python3 dataset/indoor/prepare_3dfront.py --root data/indoor/3dfront
python3 dataset/indoor/prepare_hssd.py --root data/indoor/hssd
```

3D-FRONT and HSSD require accepting their dataset terms separately; the `prepare_*.py` scripts index already-downloaded folders.

Prepare portrait sources:

```bash
python3 dataset/portrait/download_facescape_tu.py --extract --delete-zip-after-extract
python3 dataset/portrait/download_renderpeople_free.py --extract --delete-zip-after-extract
python3 dataset/portrait/download_3dscanstore_free_head.py --extract --delete-zip-after-extract
python3 dataset/portrait/download_humano_free.py --dry-run
python3 dataset/portrait/download_sketchfab_human.py --dry-run
```

Preview portrait/object/HDRI assets into `outputs/previews/{dataset}`:

```bash
blender -b --python dataset/portrait/preview_renderpeople_free.py -- \
  --root data/renderpeople_free/extracted

blender -b --python dataset/portrait/preview_facescape_tu.py -- \
  --root data/facescape/tu_model/extracted \
  --add-eyes \
  --add-hair-cap \
  --overwrite

blender -b --python dataset/portrait/preview_hsrd100.py -- \
  --root data/hsrd100/LOD1

python3 dataset/portrait/preview_blenderkit_human.py \
  --api-key-file blenderkit_key.txt \
  --queries-file dataset/portrait/queries_blenderkit_human.txt \
  --target-count 50

python3 dataset/hdri/preview_polyhaven_hdri.py
```

Build a final curated portrait manifest from accepted source manifests:

```bash
python3 dataset/portrait/build_portrait_asset_manifest.py \
  --inputs \
    outputs/previews/renderpeople_free/accepted.txt \
    outputs/previews/humano_free/accepted.txt \
    outputs/previews/blenderkit_human/accepted.txt \
    outputs/previews/sketchfab_human/accepted.txt \
  --out outputs/previews/portrait_assets/portrait_assets_objects.txt
```

Preview outputs use:

```text
outputs/previews/{dataset}/img/{dataset}_000001.png
outputs/previews/{dataset}/metadata/{dataset}_000001.json
outputs/previews/{dataset}/{dataset}_index.json
```

## Render

Run commands directly in the terminal to see live progress bars. Avoid `nohup ... &` when you want progress feedback.

Object or portrait assets:

```bash
blender -b --python scripts/render_object_relighting.py -- \
  --config configs/tokenlight_synthetic_full.json \
  --output outputs/objaverse_xl \
  --width 1280 \
  --height 704 \
  --samples 512 \
  --max-scenes 500 \
  --component-format exr \
  --hdri-mode on \
  --only all
```

Blender scene assets use the scene entrypoint:

```bash
blender -b --python scripts/render_scene_relighting.py -- \
  --config configs/tokenlight_synthetic_full.json \
  --output outputs/blenderkit \
  --width 1280 \
  --height 704 \
  --samples 512 \
  --max-scenes 500 \
  --component-format exr \
  --hdri-mode on \
  --only fixtures
```

`--component-format` can be `exr`, `png`, or `both`.

- `exr`: linear HDR components
- `png`: tone-mapped PNG components
- `both`: saves both

PNG components are created from linear values using:

```python
tone_mapped = linear / (1 + linear)
png = tone_mapped ** (1 / 2.2)
```

`--hdri-mode` can be `on`, `off`, or `random`. HDRI is applied to ambient renders. Receiver floor/wall materials use downloaded Poly Haven PBR textures when `receiver_texture_manifest` exists, otherwise procedural materials are used.

## Output

```text
outputs/{dataset}/
  dataset_manifest.json
  scenes/
    scene_000000/
      meta.json
      spatial/
        ambient.exr
        point_lights/light_000.exr
      diffuse/
      masks/
      pbr/
```
