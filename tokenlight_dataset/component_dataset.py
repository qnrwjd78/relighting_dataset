from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .exr_io import read_exr
from .tonemap import reinhard


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class TokenLightComponentDataset:
    """On-the-fly pair synthesis from TokenLight-style linear EXR components.

    Returns images normalized to [-1, 1], matching common diffusion training
    conventions. Conditions are intentionally plain dictionaries so model code
    can choose its own tokenization.
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
        self.scenes = sorted(p for p in (self.root / "scenes").glob("scene_*") if (p / "meta.json").exists())
        if not self.scenes:
            raise FileNotFoundError(f"No TokenLight component scenes found under {self.root / 'scenes'}")

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
                elif mode == "fixture":
                    input_img, target_img, condition = self._sample_fixture(scene_dir, meta, rng)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                condition["scene_id"] = meta["scene_id"]
                return self._pack(input_img, target_img, condition)
            except Exception:
                continue
        raise RuntimeError("Failed to sample a valid TokenLight pair after 30 attempts")

    def _load_meta(self, scene_dir: Path) -> dict:
        with (scene_dir / "meta.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _pack(self, input_img: np.ndarray, target_img: np.ndarray, condition: dict) -> dict[str, Any]:
        input_chw = np.transpose(input_img * 2.0 - 1.0, (2, 0, 1)).astype(np.float32)
        target_chw = np.transpose(target_img * 2.0 - 1.0, (2, 0, 1)).astype(np.float32)
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
        ambient = read_exr(scene_dir / spatial["ambient_render"])
        lights = spatial["point_lights"]
        selected = rng.sample(lights, k=min(rng.randint(1, max(1, self.max_lights)), len(lights)))
        ambient_scale = rng.uniform(0.25, 1.15)
        target_linear = ambient_scale * ambient
        condition_lights = []
        for light in selected:
            contrib = read_exr(scene_dir / light["render"])
            color = sample_color(rng)
            intensity = rng.uniform(0.15, 1.25)
            target_linear += intensity * contrib * color.reshape(1, 1, 3)
            condition_lights.append(
                {
                    "position": light["canonical_position"],
                    "color": color.tolist(),
                    "intensity": intensity,
                    "radius": light.get("canonical_radius"),
                }
            )
        return reinhard(ambient), reinhard(target_linear), {"task": "spatial", "ambient_scale": ambient_scale, "lights": condition_lights}

    def _sample_ambient(self, scene_dir: Path, meta: dict, rng: random.Random):
        ambient = read_exr(scene_dir / meta["spatial"]["ambient_render"])
        a_in = rng.uniform(0.25, 1.15)
        a_out = rng.uniform(0.05, 1.35)
        return (
            reinhard(a_in * ambient),
            reinhard(a_out * ambient),
            {"task": "ambient", "ambient_scale_in": a_in, "ambient_scale_out": a_out, "ambient_scale_delta": a_out / max(a_in, 1e-6)},
        )

    def _sample_diffuse(self, scene_dir: Path, meta: dict, rng: random.Random):
        diffuse = meta["diffuse"]
        ambient = read_exr(scene_dir / diffuse["ambient_render"])
        src, dst = rng.sample(diffuse["spreads"], 2)
        color = sample_color(rng)
        intensity = rng.uniform(0.25, 1.2)
        ambient_scale = rng.uniform(0.3, 1.1)
        source = read_exr(scene_dir / src["render"])
        target = read_exr(scene_dir / dst["render"])
        input_linear = ambient_scale * ambient + intensity * source * color.reshape(1, 1, 3)
        target_linear = ambient_scale * ambient + intensity * target * color.reshape(1, 1, 3)
        return (
            reinhard(input_linear),
            reinhard(target_linear),
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

    def _sample_fixture(self, scene_dir: Path, meta: dict, rng: random.Random):
        fixture_meta = meta["fixtures"]
        env = read_exr(scene_dir / fixture_meta["environment_render"])
        fixtures = fixture_meta["fixtures"]
        selected = rng.choice(fixtures)
        base = env.copy()
        others = [f for f in fixtures if f["id"] != selected["id"]]
        rng.shuffle(others)
        for other in others[: fixture_meta.get("max_non_selected_fixtures_in_ambient", 5)]:
            if rng.random() < 0.6:
                base += rng.uniform(0.15, 1.0) * read_exr(scene_dir / other["contribution_render"]) * sample_color(rng).reshape(1, 1, 3)
        color = sample_color(rng)
        intensity = rng.uniform(0.1, 1.2)
        ambient_scale = rng.uniform(0.35, 1.1)
        contribution = read_exr(scene_dir / selected["contribution_render"])
        off = ambient_scale * base
        on = off + intensity * contribution * color.reshape(1, 1, 3)
        turn_on = rng.random() < 0.5
        return (
            reinhard(off if turn_on else on),
            reinhard(on if turn_on else off),
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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()
