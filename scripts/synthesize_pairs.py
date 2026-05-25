from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenlight_dataset.exr_io import read_exr
from tokenlight_dataset.tonemap import reinhard, to_uint8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize TokenLight training pairs from linear EXR components.")
    parser.add_argument("--dataset", default="outputs/tokenlight_synthetic")
    parser.add_argument("--out", default="outputs/previews")
    parser.add_argument("--mode", choices=["all", "spatial", "ambient", "diffuse", "fixture"], default="all")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-lights", type=int, default=1, help="Spatial mode can sum 1..N active point lights.")
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_scenes(dataset: Path) -> list[Path]:
    scenes_dir = dataset / "scenes"
    scenes = sorted(p for p in scenes_dir.glob("scene_*") if (p / "meta.json").exists())
    if not scenes:
        raise FileNotFoundError(f"No scenes with meta.json under {scenes_dir}")
    return scenes


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


def save_pair(out_dir: Path, index: int, input_img: np.ndarray, target_img: np.ndarray, condition: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{index:06d}"
    input_path = out_dir / f"{stem}_input.png"
    target_path = out_dir / f"{stem}_target.png"
    Image.fromarray(to_uint8(input_img)).save(input_path)
    Image.fromarray(to_uint8(target_img)).save(target_path)
    return {
        "input": input_path.name,
        "target": target_path.name,
        "condition": condition,
    }


def sample_spatial(scene_dir: Path, meta: dict, rng: random.Random, max_lights: int) -> tuple[np.ndarray, np.ndarray, dict]:
    ambient = read_exr(scene_dir / meta["spatial"]["ambient_render"])
    lights = meta["spatial"]["point_lights"]
    n_active = rng.randint(1, max(1, max_lights))
    selected = rng.sample(lights, k=min(n_active, len(lights)))
    ambient_scale = rng.uniform(0.25, 1.15)

    input_linear = ambient.copy()
    target_linear = ambient_scale * ambient
    cond_lights = []

    for light_meta in selected:
        contrib = read_exr(scene_dir / light_meta["render"])
        color = sample_color(rng)
        intensity = rng.uniform(0.15, 1.25)
        target_linear = target_linear + intensity * contrib * color.reshape(1, 1, 3)
        cond_lights.append(
            {
                "position": light_meta["canonical_position"],
                "color": color.tolist(),
                "intensity": intensity,
                "radius": light_meta.get("canonical_radius"),
            }
        )

    return (
        reinhard(input_linear),
        reinhard(target_linear),
        {
            "task": "spatial",
            "ambient_scale": ambient_scale,
            "lights": cond_lights,
        },
    )


def sample_ambient(scene_dir: Path, meta: dict, rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    ambient = read_exr(scene_dir / meta["spatial"]["ambient_render"])
    a_in = rng.uniform(0.25, 1.15)
    a_out = rng.uniform(0.05, 1.35)
    return (
        reinhard(a_in * ambient),
        reinhard(a_out * ambient),
        {
            "task": "ambient",
            "ambient_scale_in": a_in,
            "ambient_scale_out": a_out,
            "ambient_scale_delta": a_out / max(a_in, 1e-6),
        },
    )


def sample_diffuse(scene_dir: Path, meta: dict, rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    diffuse = meta.get("diffuse")
    if not diffuse or not diffuse.get("spreads"):
        raise RuntimeError(f"Scene {scene_dir} has no diffuse renders")

    ambient = read_exr(scene_dir / diffuse["ambient_render"])
    spreads = diffuse["spreads"]
    src, dst = rng.sample(spreads, 2)
    source = read_exr(scene_dir / src["render"])
    target = read_exr(scene_dir / dst["render"])
    color = sample_color(rng)
    intensity = rng.uniform(0.25, 1.2)
    ambient_scale = rng.uniform(0.3, 1.1)

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


def sample_fixture(scene_dir: Path, meta: dict, rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    fixture_meta = meta.get("fixtures")
    if not fixture_meta or not fixture_meta.get("fixtures"):
        raise RuntimeError(f"Scene {scene_dir} has no fixture renders")

    env = read_exr(scene_dir / fixture_meta["environment_render"])
    fixtures = fixture_meta["fixtures"]
    selected = rng.choice(fixtures)
    selected_contrib = read_exr(scene_dir / selected["contribution_render"])

    ambient_linear = env.copy()
    others = [f for f in fixtures if f["id"] != selected["id"]]
    rng.shuffle(others)
    for other in others[: fixture_meta.get("max_non_selected_fixtures_in_ambient", 5)]:
        if rng.random() < 0.6:
            c = sample_color(rng)
            lam = rng.uniform(0.15, 1.0)
            ambient_linear += lam * read_exr(scene_dir / other["contribution_render"]) * c.reshape(1, 1, 3)

    color = sample_color(rng)
    intensity = rng.uniform(0.1, 1.2)
    ambient_scale = rng.uniform(0.35, 1.1)
    turn_on = rng.random() < 0.5
    lit = ambient_scale * ambient_linear + intensity * selected_contrib * color.reshape(1, 1, 3)
    base = ambient_scale * ambient_linear

    if turn_on:
        input_linear, target_linear = base, lit
        transition = 1
    else:
        input_linear, target_linear = lit, base
        transition = 0

    return (
        reinhard(input_linear),
        reinhard(target_linear),
        {
            "task": "fixture",
            "fixture_id": selected["id"],
            "mask": selected["mask_render"],
            "transition_on": transition,
            "color": color.tolist(),
            "intensity": intensity,
            "ambient_scale": ambient_scale,
        },
    )


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset = resolve_repo_path(args.dataset)
    out_dir = resolve_repo_path(args.out)
    scenes = list_scenes(dataset)

    records = []
    modes = ["spatial", "ambient", "diffuse", "fixture"] if args.mode == "all" else [args.mode]
    attempts = 0
    while len(records) < args.count:
        attempts += 1
        if attempts > args.count * 20:
            raise RuntimeError("Could not synthesize enough pairs. Check rendered component availability.")
        scene_dir = rng.choice(scenes)
        meta = load_json(scene_dir / "meta.json")
        mode = rng.choice(modes)
        try:
            if mode == "spatial":
                input_img, target_img, condition = sample_spatial(scene_dir, meta, rng, args.max_lights)
            elif mode == "ambient":
                input_img, target_img, condition = sample_ambient(scene_dir, meta, rng)
            elif mode == "diffuse":
                input_img, target_img, condition = sample_diffuse(scene_dir, meta, rng)
            else:
                input_img, target_img, condition = sample_fixture(scene_dir, meta, rng)
        except Exception as exc:
            print(f"Skipping {mode} for {scene_dir.name}: {exc}")
            continue

        condition["scene_id"] = meta["scene_id"]
        records.append(save_pair(out_dir, len(records), input_img, target_img, condition))

    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} pairs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
