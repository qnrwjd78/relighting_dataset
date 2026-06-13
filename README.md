# Relighting Dataset

Synthetic relighting dataset tools for object/portrait assets and Blender scene assets.

## Layout

```text
dataset/   download_*.py and preview_*.py entrypoints
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
python3 dataset/download_polyhaven_hdri.py --resolution 2k --format hdr --per-category 30
python3 dataset/download_polyhaven_textures.py --resolution 2k --format jpg --per-category 20
python3 dataset/download_objaverse_xl.py --limit 5000
python3 dataset/download_hsrd100.py --lod LOD1 --extract
```

Prepare local/direct RenderPeople free packages, then preview the extracted assets:

```bash
python3 dataset/utils/util_prepare_renderpeople_free.py --zip-dir /path/to/renderpeople_zips
blender -b --python dataset/preview_renderpeople_free.py -- \
  --root data/renderpeople_free/extracted
```

Preview portrait/object assets into `outputs/previews/{dataset}`:

```bash
blender -b --python dataset/preview_hsrd100.py -- \
  --root data/hsrd100/LOD1
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
