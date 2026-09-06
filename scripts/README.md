# Script map

The root scripts are grouped by workflow below. Dataset outputs remain under
`outputs/` and model weights are centralized under `weights/`.

Run `./scripts/download_weights.sh` once to place all required model files under
the repository-level `weights/` directory. Model-consuming scripts use these
repository-relative paths by default.

The MoGe source is pinned as the `repos/MoGe` Git submodule. Initialize it with
`git submodule update --init --recursive` and install it with
`python3 -m pip install -e repos/MoGe` in a new environment.

## Rendering

- `render_objaverse_random_dataset.py`: general randomized Objaverse renderer.
- `render_objaverse_fixed_dataset.py`: base fixed-grid Blender renderer.
- `render_objaverse_fixed_multi_gpu.py`: base multi-GPU launcher.
- `render_objaverse_fixed_grid_dataset.py`: configurable `NxMxK` fixed-grid variant.
- `render_objaverse_fixed_grid_multi_gpu.py`: configurable fixed-grid multi-GPU launcher.
- `render_objaverse_fixed32_random2_dataset.py`: specialized two-power fixed32 experiment.
- `render_objaverse_inference_gt.py` and `render_objaverse_inference_gt_multi_gpu.py`: inference GT generation.
- `render_blenderkit_dataset.py`: BlenderKit renderer.

## Dataset processing

- `convert_exr_dataset_to_png.py`: EXR-to-PNG conversion and shard merge.
- `relighting_mask_pipeline.py`: shared mask generation used by Blender renderers.
- `generate_lgi_maps.py`: LGI map generation.
- `precompute_wan_vae_cache.py` and `wan_vae2_2.py`: Wan VAE cache worker and model.
- `objaverse_245_pipeline.py`: unified cache, source point-map, PBR/mask archive, verification, and upload CLI.
- `pipeline_utils.py`: shared archive, checksum, upload, and GPU-shard helpers.

## MoGe utilities

- `extract_moge3_source_pointmaps.py`: source-image point maps written per scene.
- `extract_moge3_pointmaps_to_tar.py`: streaming archive worker for arbitrary image batches.
- `run_moge3_pointmap_batches.py`: resumable batch coordinator for the streaming worker.

## Validation and inspection

- `validate_objaverse_shadow_materials.py`
- `evaluate_geometry_shadow_photometric_consistency.py`
- `build_relight_pair_dataset.py`
- `make_objaverse_height_plane_video.py`

Provider-specific download and preview tools are already organized below
`download/{hdri,object,portrait,scene,utils}/`. The `utils/` package contains
small reusable helpers rather than standalone dataset workflows.
