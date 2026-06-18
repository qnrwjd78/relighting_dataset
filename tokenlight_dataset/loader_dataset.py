from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class TokenLightPNGLoaderDataset:
    """On-the-fly pair synthesis from TokenLight-style PNG components.

    This mirrors TokenLightComponentDataset but reads PNG components and blends
    them in display space. It is useful for fast preview/prototype training.
    For physically correct light arithmetic, use linear EXR components instead.
    """

    def __init__(
        self,
        root: str | Path,
        length: int = 100_000,
        modes: tuple[str, ...] = ("spatial", "ambient", "diffuse", "fixture"),
        seed: int = 1234,
        max_lights: int = 1,
        return_torch: bool = False,
    ) -> None:
        self.root = resolve_repo_path(root)
        self.length = int(length)
        self.modes = tuple(modes)
        self.seed = int(seed)
        self.max_lights = int(max_lights)
        self.return_torch = bool(return_torch)
        scenes_root = self.root / "scenes"
        if not scenes_root.exists():
            raise FileNotFoundError(f"No TokenLight scenes directory found: {scenes_root}")
        self.scenes = sorted(p for p in scenes_root.iterdir() if p.is_dir() and (p / "meta.json").exists())
        if not self.scenes:
            raise FileNotFoundError(f"No TokenLight component scenes found under {scenes_root}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + int(index))
        for _ in range(30):
            scene_dir = rng.choice(self.scenes)
            meta = self._load_meta(scene_dir)
            mode = rng.choice(self.modes)
            try:
                if mode == "spatial":
                    input_img, target_img, condition = self._sample_spatial(scene_dir, meta, rng)
                elif mode == "ambient":
                    input_img, target_img, condition = self._sample_ambient(scene_dir, meta, rng)
                elif mode == "diffuse":
                    input_img, target_img, condition = self._sample_diffuse(scene_dir, meta, rng)
                elif mode == "global_diffuse":
                    input_img, target_img, condition = self._sample_global_diffuse(scene_dir, meta, rng)
                elif mode == "per_light_diffuse":
                    input_img, target_img, condition = self._sample_per_light_diffuse(scene_dir, meta, rng)
                elif mode == "fixture":
                    input_img, target_img, condition = self._sample_fixture(scene_dir, meta, rng)
                elif mode == "color_mix":
                    input_img, target_img, condition = self._sample_color_mix(scene_dir, meta, rng)
                elif mode == "light_intensity":
                    input_img, target_img, condition = self._sample_light_intensity(scene_dir, meta, rng)
                elif mode == "fixture_intensity":
                    input_img, target_img, condition = self._sample_fixture_intensity(scene_dir, meta, rng)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                condition["scene_id"] = meta["scene_id"]
                condition["component_format"] = "png"
                condition["blend_space"] = "display_png"
                return self._pack(input_img, target_img, condition)
            except Exception:
                continue
        raise RuntimeError("Failed to sample a valid TokenLight PNG pair after 30 attempts")

    def _load_meta(self, scene_dir: Path) -> dict:
        with (scene_dir / "meta.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _pack(self, input_img: np.ndarray, target_img: np.ndarray, condition: dict) -> dict[str, Any]:
        input_chw = np.transpose(clip01(input_img) * 2.0 - 1.0, (2, 0, 1)).astype(np.float32)
        target_chw = np.transpose(clip01(target_img) * 2.0 - 1.0, (2, 0, 1)).astype(np.float32)
        if not self.return_torch:
            return {"input": input_chw, "target": target_chw, "condition": condition}
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("return_torch=True requires PyTorch to be installed") from exc
        return {
            "input": torch.from_numpy(input_chw),
            "target": torch.from_numpy(target_chw),
            "condition": condition,
        }

    def _sample_spatial(self, scene_dir: Path, meta: dict, rng: random.Random):
        spatial = meta["spatial"]
        ambient = read_png_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        lights = valid_lights(spatial)
        selected = rng.sample(lights, k=min(rng.randint(1, max(1, self.max_lights)), len(lights)))
        ambient_scale = rng.uniform(0.25, 1.15)
        target = ambient_scale * ambient
        condition_lights = []
        intensity_lo, intensity_hi = intensity_range(spatial, [0.15, 1.25])
        for light in selected:
            component = read_spatial_png_component(scene_dir, spatial, light, ambient)
            color = sample_color(rng)
            intensity = rng.uniform(intensity_lo, intensity_hi)
            target += intensity * component * color.reshape(1, 1, 3)
            condition_lights.append(
                {
                    "position": light["canonical_position"],
                    "color": color.tolist(),
                    "intensity": intensity,
                    "radius": light.get("canonical_radius"),
                    "base_energy": light.get("canonical_energy"),
                }
            )
        return clip01(ambient), clip01(target), {"task": "spatial", "ambient_scale": ambient_scale, "lights": condition_lights}

    def _sample_color_mix(self, scene_dir: Path, meta: dict, rng: random.Random):
        spatial = meta["spatial"]
        ambient = read_png_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        lights = valid_lights(spatial)
        max_mix_lights = min(max(2, self.max_lights), len(lights))
        light_count = rng.randint(1 if len(lights) == 1 else 2, max_mix_lights)
        selected = rng.sample(lights, k=light_count)
        ambient_scale = rng.uniform(0.35, 1.05)
        input_img = ambient_scale * ambient
        target_img = ambient_scale * ambient
        condition_lights = []
        intensity_lo, intensity_hi = intensity_range(spatial, [0.15, 1.25])
        for light in selected:
            component = read_spatial_png_component(scene_dir, spatial, light, ambient)
            color_in = sample_color(rng)
            color_out = sample_color(rng)
            intensity_in = rng.uniform(intensity_lo, intensity_hi)
            intensity_out = rng.uniform(intensity_lo, intensity_hi)
            input_img += intensity_in * component * color_in.reshape(1, 1, 3)
            target_img += intensity_out * component * color_out.reshape(1, 1, 3)
            condition_lights.append(
                {
                    "position": light["canonical_position"],
                    "color_in": color_in.tolist(),
                    "color_out": color_out.tolist(),
                    "intensity_in": intensity_in,
                    "intensity_out": intensity_out,
                    "radius": light.get("canonical_radius"),
                    "base_energy": light.get("canonical_energy"),
                }
            )
        return clip01(input_img), clip01(target_img), {"task": "color_mix", "ambient_scale": ambient_scale, "lights": condition_lights}

    def _sample_light_intensity(self, scene_dir: Path, meta: dict, rng: random.Random):
        spatial = meta["spatial"]
        ambient = read_png_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        lights = valid_lights(spatial)
        light_count = min(rng.randint(1, max(1, self.max_lights)), len(lights))
        selected = rng.sample(lights, k=light_count)
        ambient_scale = rng.uniform(0.35, 1.05)
        input_img = ambient_scale * ambient
        target_img = ambient_scale * ambient
        condition_lights = []
        for light in selected:
            component = read_spatial_png_component(scene_dir, spatial, light, ambient)
            color = sample_color(rng)
            intensity_in, intensity_out = sample_intensity_pair(rng)
            input_img += intensity_in * component * color.reshape(1, 1, 3)
            target_img += intensity_out * component * color.reshape(1, 1, 3)
            condition_lights.append(
                {
                    "position": light["canonical_position"],
                    "color": color.tolist(),
                    "intensity_in": intensity_in,
                    "intensity_out": intensity_out,
                    "intensity_delta": intensity_out - intensity_in,
                    "turning_off": 1 if intensity_out < intensity_in else 0,
                    "radius": light.get("canonical_radius"),
                    "base_energy": light.get("canonical_energy"),
                }
            )
        return clip01(input_img), clip01(target_img), {"task": "light_intensity", "ambient_scale": ambient_scale, "lights": condition_lights}

    def _sample_ambient(self, scene_dir: Path, meta: dict, rng: random.Random):
        spatial = meta["spatial"]
        ambient = read_png_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        a_in = rng.uniform(0.25, 1.15)
        a_out = rng.uniform(0.05, 1.35)
        return (
            clip01(a_in * ambient),
            clip01(a_out * ambient),
            {"task": "ambient", "ambient_scale_in": a_in, "ambient_scale_out": a_out, "ambient_scale_delta": a_out / max(a_in, 1e-6)},
        )

    def _sample_diffuse(self, scene_dir: Path, meta: dict, rng: random.Random):
        if global_diffuse_meta(meta):
            return self._sample_global_diffuse(scene_dir, meta, rng)

        diffuse = meta["diffuse"]
        ambient = read_png_component(scene_dir, diffuse.get("ambient_output", diffuse["ambient_render"]))
        src, dst = rng.sample(diffuse["spreads"], 2)
        color = sample_color(rng)
        intensity = rng.uniform(0.25, 1.2)
        ambient_scale = rng.uniform(0.3, 1.1)
        source = read_png_component(scene_dir, src.get("output", src))
        target = read_png_component(scene_dir, dst.get("output", dst))
        input_img = ambient_scale * ambient + intensity * source * color.reshape(1, 1, 3)
        target_img = ambient_scale * ambient + intensity * target * color.reshape(1, 1, 3)
        return (
            clip01(input_img),
            clip01(target_img),
            {
                "task": "diffuse",
                "spread_in": src["normalized_spread"],
                "spread_out": dst["normalized_spread"],
                "spread_delta": dst["normalized_spread"] - src["normalized_spread"],
                "color": color.tolist(),
                "intensity": intensity,
                "ambient_scale": ambient_scale,
            },
        )

    def _sample_global_diffuse(self, scene_dir: Path, meta: dict, rng: random.Random):
        diffuse = global_diffuse_meta(meta)
        if not diffuse:
            raise RuntimeError(f"No global_diffuse metadata in {scene_dir}")
        variants = [row for row in diffuse.get("variants", []) if row.get("render")]
        if len(variants) < 2:
            raise RuntimeError(f"Need at least two global_diffuse variants in {scene_dir}")

        src, dst = rng.sample(variants, 2)
        complete_targets = bool(diffuse.get("complete_target_variants", True))
        if complete_targets:
            input_img = read_png_component(scene_dir, src)
            target_img = read_png_component(scene_dir, dst)
            ambient_scale = None
            intensity = None
            color = None
        else:
            ambient_entry = diffuse.get("ambient_output", diffuse.get("ambient_render"))
            if ambient_entry is None:
                raise RuntimeError(f"Component global_diffuse metadata needs ambient_output in {scene_dir}")
            ambient = read_png_component(scene_dir, ambient_entry)
            source = read_png_component(scene_dir, src)
            target = read_png_component(scene_dir, dst)
            ambient_range = diffuse.get("ambient_scale_range", [0.85, 1.15])
            intensity_range = diffuse.get("intensity_range", [0.85, 1.15])
            ambient_scale = rng.uniform(float(ambient_range[0]), float(ambient_range[1]))
            intensity = rng.uniform(float(intensity_range[0]), float(intensity_range[1]))
            color = np.asarray(diffuse.get("light", {}).get("color", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
            input_img = ambient_scale * ambient + intensity * source * color
            target_img = ambient_scale * ambient + intensity * target * color
        return (
            clip01(input_img),
            clip01(target_img),
            {
                "task": "global_diffuse",
                "dg_in": float(src.get("dg", src.get("normalized_diffuse", 0.0))),
                "dg_out": float(dst.get("dg", dst.get("normalized_diffuse", 0.0))),
                "dg_delta": float(dst.get("dg", dst.get("normalized_diffuse", 0.0)))
                - float(src.get("dg", src.get("normalized_diffuse", 0.0))),
                "complete_target_variants": complete_targets,
                "spread_in": src.get("spread_degrees"),
                "spread_out": dst.get("spread_degrees"),
                "ambient_scale": ambient_scale,
                "intensity": intensity,
                "color": None if color is None else color.reshape(3).tolist(),
            },
        )

    def _sample_per_light_diffuse(self, scene_dir: Path, meta: dict, rng: random.Random):
        spatial = meta["spatial"]
        ambient = read_png_component(scene_dir, spatial.get("ambient_output", spatial["ambient_render"]))
        lights = [light for light in spatial["point_lights"] if light.get("valid", True) and light.get("diffuse_variants")]
        if not lights:
            raise RuntimeError(f"No per-light diffuse variants in {scene_dir}")

        light = rng.choice(lights)
        variants = [row for row in light.get("diffuse_variants", []) if row.get("render")]
        if len(variants) < 2:
            raise RuntimeError(f"Need at least two diffuse variants for light {light.get('id')}")

        src, dst = rng.sample(variants, 2)
        color = sample_color(rng)
        intensity = rng.uniform(0.25, 1.2)
        ambient_scale = rng.uniform(0.3, 1.1)
        source = read_spatial_png_component(scene_dir, spatial, light, ambient, src)
        target = read_spatial_png_component(scene_dir, spatial, light, ambient, dst)
        input_img = ambient_scale * ambient + intensity * source * color.reshape(1, 1, 3)
        target_img = ambient_scale * ambient + intensity * target * color.reshape(1, 1, 3)
        return (
            clip01(input_img),
            clip01(target_img),
            {
                "task": "per_light_diffuse",
                "position": light["canonical_position"],
                "d_in": float(src.get("d", src.get("normalized_diffuse", 0.0))),
                "d_out": float(dst.get("d", dst.get("normalized_diffuse", 0.0))),
                "d_delta": float(dst.get("d", dst.get("normalized_diffuse", 0.0)))
                - float(src.get("d", src.get("normalized_diffuse", 0.0))),
                "color": color.tolist(),
                "intensity": intensity,
                "ambient_scale": ambient_scale,
                "base_energy": light.get("canonical_energy"),
            },
        )

    def _sample_fixture(self, scene_dir: Path, meta: dict, rng: random.Random):
        fixture_meta = meta["fixtures"]
        env = read_png_component(scene_dir, fixture_meta.get("environment_output", fixture_meta["environment_render"]))
        fixtures = [fixture for fixture in fixture_meta["fixtures"] if fixture.get("contribution_render")]
        selected = rng.choice(fixtures)
        base = env.copy()
        others = [fixture for fixture in fixtures if fixture["id"] != selected["id"]]
        rng.shuffle(others)
        for other in others[: fixture_meta.get("max_non_selected_fixtures_in_ambient", 5)]:
            if rng.random() < 0.6:
                base += rng.uniform(0.15, 1.0) * read_png_component(scene_dir, other.get("contribution_output", other["contribution_render"]))
        color = sample_color(rng)
        intensity = rng.uniform(0.1, 1.2)
        ambient_scale = rng.uniform(0.35, 1.1)
        contribution = read_png_component(scene_dir, selected.get("contribution_output", selected["contribution_render"]))
        off = ambient_scale * base
        on = off + intensity * contribution * color.reshape(1, 1, 3)
        turn_on = rng.random() < 0.5
        return (
            clip01(off if turn_on else on),
            clip01(on if turn_on else off),
            {
                "task": "fixture",
                "fixture_id": selected["id"],
                "mask": selected["mask_render"],
                "transition_on": 1 if turn_on else 0,
                "color": color.tolist(),
                "intensity": intensity,
                "ambient_scale": ambient_scale,
            },
        )

    def _sample_fixture_intensity(self, scene_dir: Path, meta: dict, rng: random.Random):
        fixture_meta = meta["fixtures"]
        env = read_png_component(scene_dir, fixture_meta.get("environment_output", fixture_meta["environment_render"]))
        fixtures = [fixture for fixture in fixture_meta["fixtures"] if fixture.get("contribution_render")]
        selected = rng.choice(fixtures)
        ambient_scale = rng.uniform(0.35, 1.1)
        base = ambient_scale * env
        others = [fixture for fixture in fixtures if fixture["id"] != selected["id"]]
        rng.shuffle(others)
        other_terms = []
        for other in others[: fixture_meta.get("max_non_selected_fixtures_in_ambient", 5)]:
            if rng.random() < 0.55:
                other_intensity = rng.uniform(0.05, 0.9)
                base += other_intensity * read_png_component(scene_dir, other.get("contribution_output", other["contribution_render"]))
                other_terms.append({"fixture_id": other["id"], "intensity": other_intensity})
        contribution = read_png_component(scene_dir, selected.get("contribution_output", selected["contribution_render"]))
        intensity_in, intensity_out = sample_intensity_pair(rng)
        input_img = base + intensity_in * contribution
        target_img = base + intensity_out * contribution
        return (
            clip01(input_img),
            clip01(target_img),
            {
                "task": "fixture_intensity",
                "fixture_id": selected["id"],
                "mask": selected["mask_render"],
                "intensity_in": intensity_in,
                "intensity_out": intensity_out,
                "intensity_delta": intensity_out - intensity_in,
                "turning_off": 1 if intensity_out < intensity_in else 0,
                "ambient_scale": ambient_scale,
                "other_fixtures": other_terms,
            },
        )


TokenLightLoaderDataset = TokenLightPNGLoaderDataset


def valid_lights(spatial: dict) -> list[dict]:
    lights = [light for light in spatial["point_lights"] if light.get("valid", True) and light.get("render")]
    if not lights:
        raise RuntimeError("No valid spatial point lights")
    return lights


def intensity_range(spatial: dict, fallback: list[float]) -> tuple[float, float]:
    value = spatial.get("intensity_range", fallback)
    return float(value[0]), float(value[1])


def read_spatial_png_component(
    scene_dir: Path,
    spatial: dict,
    light: dict,
    ambient: np.ndarray,
    component_entry: dict | None = None,
) -> np.ndarray:
    component = read_png_component(scene_dir, (component_entry or light).get("output", component_entry or light))
    if spatial.get("point_light_output_semantics") == "ambient_plus_point_light_target":
        component = component - ambient
    source_color = np.asarray(light.get("component_color", [1.0, 1.0, 1.0]), dtype=np.float32)
    source_color = np.maximum(source_color.reshape(1, 1, 3), 1e-4)
    component = component / source_color
    return clip01(component)


def read_png_component(scene_dir: Path, entry: dict | str) -> np.ndarray:
    path = resolve_component_png_path(scene_dir, entry)
    try:
        import imageio.v3 as iio

        img = iio.imread(path)
        return normalize_png_array(img)
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(path) as img:
            return normalize_png_array(np.asarray(img.convert("RGB")))
    except Exception:
        pass

    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imread returned None")
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3][:, :, ::-1]
        return normalize_png_array(img)
    except Exception as exc:
        raise RuntimeError(f"Could not read PNG component {path}: {exc}") from exc


def resolve_component_png_path(scene_dir: Path, entry: dict | str) -> Path:
    candidates: list[str] = []
    if isinstance(entry, dict):
        for key in ("render_png", "png", "render"):
            value = entry.get(key)
            if value:
                candidates.append(str(value))
    else:
        candidates.append(str(entry))

    for candidate in candidates:
        path = scene_dir / candidate
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        if path.exists():
            return path

    raw = candidates[0] if candidates else str(entry)
    raise FileNotFoundError(f"No PNG component found for {raw} under {scene_dir}")


def normalize_png_array(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    scale = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else None
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC PNG image, got shape {img.shape}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {img.shape}")
    img = img[:, :, :3].astype(np.float32, copy=False)
    if scale is not None and scale > 0.0:
        img = img / scale
    elif img.max(initial=0.0) > 1.5:
        img = img / 255.0
    return clip01(img)


def clip01(img: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(img, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def global_diffuse_meta(meta: dict) -> dict | None:
    return meta.get("global_diffuse") or meta.get("spatial", {}).get("global_diffuse")


def sample_color(rng: random.Random) -> np.ndarray:
    palette = [
        (1.0, 1.0, 1.0),
        (1.0, 0.86, 0.68),
        (0.68, 0.82, 1.0),
        (1.0, 0.34, 0.24),
        (0.25, 0.50, 1.0),
        (0.35, 1.0, 0.55),
    ]
    if rng.random() < 0.65:
        return np.array((1.0, 1.0, 1.0), dtype=np.float32)
    if rng.random() < 0.75:
        return np.array(rng.choice(palette), dtype=np.float32)
    return np.array([rng.uniform(0.45, 1.0) for _ in range(3)], dtype=np.float32)


def sample_intensity_pair(rng: random.Random) -> tuple[float, float]:
    mode = rng.random()
    if mode < 0.55:
        return rng.uniform(0.65, 1.15), rng.choice([0.0, 0.15, 0.3, 0.5, 0.75])
    if mode < 0.8:
        return rng.choice([0.0, 0.15, 0.3, 0.5]), rng.uniform(0.65, 1.15)
    return rng.uniform(0.05, 1.15), rng.uniform(0.05, 1.15)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()
