from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


OBJECT_SECTION = "single_scene_light_good"
BACKGROUND_SECTION = "background_good_for_portrait_or_object"

ENV_NAME_TOKENS = (
    "plane",
    "floor",
    "wall",
    "ceiling",
    "backdrop",
    "background",
    "room",
    "ground",
    "terrain",
    "base",
    "podium",
    "stage",
    "platform",
)
SUPPORT_NAME_TOKENS = ("table", "desk", "shelf", "counter", "stand", "base", "podium", "platform")
OBJECT_NAME_TOKENS = (
    "bottle",
    "serum",
    "cream",
    "product",
    "cosmetic",
    "perfume",
    "lamp",
    "light",
    "candle",
    "chair",
    "car",
    "drone",
    "robot",
    "watch",
    "phone",
    "cup",
    "mug",
    "glass",
    "flower",
    "sphere",
    "cube",
    "cylinder",
    "object",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive object mesh selections and background insertion anchors from BlenderKit preview classifications plus scene_full_dump.json files."
    )
    parser.add_argument("--classification", default="outputs/previews/blenderkit/blenderkit_scene_use_classification.txt")
    parser.add_argument("--index-json", default="outputs/previews/blenderkit/blenderkit_index.json")
    parser.add_argument("--dump-root", default="outputs/previews/blenderkit/scene_full_dump")
    parser.add_argument("--out-dir", default="outputs/previews/blenderkit/scene_decisions")
    parser.add_argument("--object-section", default=OBJECT_SECTION)
    parser.add_argument("--background-section", default=BACKGROUND_SECTION)
    parser.add_argument("--max-object-meshes", type=int, default=80)
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_classification(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        section_match = re.match(r"\[\d+\]\s+([^\s]+)", line)
        if section_match:
            current = section_match.group(1)
            sections.setdefault(current, [])
            continue
        item_match = re.match(r"- blenderkit_(\d+)", line)
        if current and item_match:
            sections[current].append(f"blenderkit_{item_match.group(1).zfill(5)}")
    return sections


def load_index(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {f"blenderkit_{str(item.get('id')).zfill(5)}": item for item in data.get("items", [])}


def has_token(name: str, tokens: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(token in lower for token in tokens)


def extent(mesh: dict) -> list[float]:
    return [float(v) for v in mesh.get("bbox_extent_world", [0.0, 0.0, 0.0])]


def center(mesh: dict) -> list[float]:
    return [float(v) for v in mesh.get("bbox_center_world", [0.0, 0.0, 0.0])]


def bbox_min(mesh: dict) -> list[float]:
    return [float(v) for v in mesh.get("bbox_min_world", [0.0, 0.0, 0.0])]


def bbox_max(mesh: dict) -> list[float]:
    return [float(v) for v in mesh.get("bbox_max_world", [0.0, 0.0, 0.0])]


def volume(mesh: dict) -> float:
    return float(mesh.get("bbox_volume_world", 0.0))


def projection(mesh: dict) -> dict:
    return mesh.get("active_camera_projection") or {}


def visible_mesh(mesh: dict) -> bool:
    return bool(mesh.get("visible_viewport", True)) and not bool(mesh.get("hide_render", False))


def is_flat_support_like(mesh: dict) -> bool:
    ex = extent(mesh)
    horizontal = max(ex[0], ex[1], 1e-8)
    return ex[2] < horizontal * 0.28 and horizontal > 0.08


def material_hints(mesh: dict) -> dict:
    image_textures = 0
    has_emission = False
    transparent_or_glass = False
    for mat in mesh.get("materials", []):
        image_textures += len(mat.get("image_textures") or [])
        has_emission = has_emission or bool(mat.get("has_emission"))
        alpha = mat.get("alpha")
        if isinstance(alpha, (int, float)) and alpha < 0.98:
            transparent_or_glass = True
        name = str(mat.get("name") or "").lower()
        if "glass" in name or "transparent" in name:
            transparent_or_glass = True
    return {"image_textures": image_textures, "has_emission": has_emission, "transparent_or_glass": transparent_or_glass}


def scene_extent(dump: dict) -> list[float]:
    return [float(v) for v in dump.get("scene_bbox", {}).get("extent", [1.0, 1.0, 1.0])]


def scene_volume(dump: dict) -> float:
    ex = scene_extent(dump)
    return max(ex[0] * ex[1] * ex[2], 1e-8)


def score_object_mesh(mesh: dict, dump: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    name = str(mesh.get("name", ""))
    proj = projection(mesh)
    area = float(proj.get("normalized_area") or 0.0)
    proj_center = proj.get("normalized_center") or [0.5, 0.5]
    center_dist = math.sqrt((float(proj_center[0]) - 0.5) ** 2 + (float(proj_center[1]) - 0.5) ** 2)
    vol_ratio = volume(mesh) / scene_volume(dump)

    if visible_mesh(mesh):
        score += 1.0
        reasons.append("visible/renderable")
    else:
        score -= 4.0
        reasons.append("hidden_or_not_renderable")
    if proj.get("any_corner_in_frame"):
        score += 1.5
        reasons.append("projects_into_active_camera")
    if area > 0:
        score += min(area * 8.0, 2.0)
        reasons.append(f"screen_area={area:.4f}")
    score += max(0.0, 0.8 - center_dist)
    if has_token(name, OBJECT_NAME_TOKENS):
        score += 0.8
        reasons.append("object_name_hint")
    if has_token(name, ENV_NAME_TOKENS):
        score -= 2.5
        reasons.append("environment/support_name_hint")
    if is_flat_support_like(mesh):
        score -= 1.8
        reasons.append("flat_support_like")
    if vol_ratio > 0.35:
        score -= 4.0
        reasons.append(f"too_large_scene_volume_ratio={vol_ratio:.3f}")
    elif vol_ratio < 1e-7:
        score -= 0.5
        reasons.append("very_tiny_volume")
    hints = material_hints(mesh)
    if hints["image_textures"]:
        score += 0.2
        reasons.append("has_image_texture")
    if hints["has_emission"]:
        score += 0.4
        reasons.append("has_emission")
    if hints["transparent_or_glass"]:
        score += 0.25
        reasons.append("transparent_or_glass")
    return score, reasons


def union_bbox(meshes: list[dict]) -> dict:
    if not meshes:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "center": [0.0, 0.0, 0.0], "extent": [0.0, 0.0, 0.0]}
    mins = [bbox_min(mesh) for mesh in meshes]
    maxs = [bbox_max(mesh) for mesh in meshes]
    out_min = [min(v[i] for v in mins) for i in range(3)]
    out_max = [max(v[i] for v in maxs) for i in range(3)]
    return {
        "min": out_min,
        "max": out_max,
        "center": [(out_min[i] + out_max[i]) * 0.5 for i in range(3)],
        "extent": [out_max[i] - out_min[i] for i in range(3)],
    }


def decide_object(scene_id: str, dump: dict, item: dict | None, max_meshes: int) -> dict:
    meshes = dump.get("meshes", [])
    scored = []
    for mesh in meshes:
        score, reasons = score_object_mesh(mesh, dump)
        scored.append({"mesh": mesh, "score": score, "reasons": reasons})
    scored.sort(key=lambda row: row["score"], reverse=True)
    selected = [row for row in scored if row["score"] >= 1.4]
    if not selected and scored:
        selected = scored[: min(5, len(scored))]
    if len(selected) > max_meshes:
        selected = selected[:max_meshes]
    selected_meshes = [row["mesh"] for row in selected]
    selected_area = sum(float(projection(mesh).get("normalized_area") or 0.0) for mesh in selected_meshes)
    confidence = "high" if selected and selected_area > 0.04 else "medium" if selected else "low"
    if len(selected) == max_meshes and len(scored) > max_meshes:
        confidence = "low"
    return {
        "scene_id": scene_id,
        "name": (item or {}).get("name"),
        "decision_type": "object_mesh_selection",
        "mode": "single_group_auto",
        "confidence": confidence,
        "candidate_objects": [mesh["name"] for mesh in selected_meshes],
        "candidate_mesh_ids": [mesh["mesh_id"] for mesh in selected_meshes],
        "excluded_objects": [row["mesh"]["name"] for row in scored if row["mesh"] not in selected_meshes],
        "candidate_bbox_world": union_bbox(selected_meshes),
        "active_camera": dump.get("active_camera"),
        "preview_png": (item or {}).get("preview_png"),
        "source_dump": f"outputs/previews/blenderkit/scene_full_dump/{scene_id}/scene_full_dump.json",
        "ranked_meshes": [
            {
                "mesh_id": row["mesh"].get("mesh_id"),
                "name": row["mesh"].get("name"),
                "score": round(row["score"], 4),
                "bbox_center_world": row["mesh"].get("bbox_center_world"),
                "bbox_extent_world": row["mesh"].get("bbox_extent_world"),
                "screen": row["mesh"].get("active_camera_projection"),
                "reasons": row["reasons"],
            }
            for row in scored[:50]
        ],
    }


def score_background_anchor(mesh: dict, dump: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    name = str(mesh.get("name", ""))
    ex = extent(mesh)
    proj = projection(mesh)
    area = float(proj.get("normalized_area") or 0.0)
    proj_center = proj.get("normalized_center") or [0.5, 0.5]

    if visible_mesh(mesh):
        score += 1.0
        reasons.append("visible/renderable")
    if is_flat_support_like(mesh):
        score += 3.0
        reasons.append("flat_horizontal_support_like")
    if has_token(name, ENV_NAME_TOKENS + SUPPORT_NAME_TOKENS):
        score += 1.0
        reasons.append("environment/support_name_hint")
    if proj.get("any_corner_in_frame"):
        score += 0.8
        reasons.append("projects_into_active_camera")
    if area > 0:
        score += min(area * 4.0, 1.5)
        reasons.append(f"screen_area={area:.4f}")
    if float(proj_center[1]) < 0.6:
        score += 0.4
        reasons.append("lower_or_mid_frame_anchor")
    horizontal = max(ex[0], ex[1])
    vertical = ex[2]
    if horizontal > vertical:
        score += 0.4
        reasons.append("wider_than_tall")
    return score, reasons


def decide_background(scene_id: str, dump: dict, item: dict | None) -> dict:
    meshes = dump.get("meshes", [])
    scored = []
    for mesh in meshes:
        score, reasons = score_background_anchor(mesh, dump)
        scored.append({"mesh": mesh, "score": score, "reasons": reasons})
    scored.sort(key=lambda row: row["score"], reverse=True)
    anchor = scored[0]["mesh"] if scored else None
    scene_bbox = dump.get("scene_bbox", {})
    if anchor:
        mn = bbox_min(anchor)
        mx = bbox_max(anchor)
        ctr = center(anchor)
        position = [ctr[0], ctr[1], mx[2]]
        anchor_extent = extent(anchor)
        scale_ref = max(min(max(anchor_extent[0], anchor_extent[1]) * 0.25, max(scene_extent(dump)) * 0.2), 0.1)
        confidence = "high" if scored[0]["score"] >= 4.0 else "medium"
        placement_mode = "anchor_mesh_top_center"
        reason = scored[0]["reasons"]
    else:
        mn = scene_bbox.get("min", [0.0, 0.0, 0.0])
        mx = scene_bbox.get("max", [0.0, 0.0, 0.0])
        ctr = scene_bbox.get("center", [0.0, 0.0, 0.0])
        position = [ctr[0], ctr[1], mn[2]]
        scale_ref = max(max(scene_extent(dump)) * 0.15, 0.1)
        confidence = "low"
        placement_mode = "scene_bbox_bottom_center_fallback"
        reason = ["no_mesh_anchor_found"]
    return {
        "scene_id": scene_id,
        "name": (item or {}).get("name"),
        "decision_type": "background_insertion_anchor",
        "confidence": confidence,
        "placement_mode": placement_mode,
        "placement_world": position,
        "up_axis_world": [0.0, 0.0, 1.0],
        "recommended_object_max_extent": scale_ref,
        "anchor_mesh": anchor.get("name") if anchor else None,
        "anchor_mesh_id": anchor.get("mesh_id") if anchor else None,
        "anchor_bbox_world": {"min": mn, "max": mx, "center": ctr, "extent": extent(anchor) if anchor else scene_extent(dump)},
        "active_camera": dump.get("active_camera"),
        "preview_png": (item or {}).get("preview_png"),
        "source_dump": f"outputs/previews/blenderkit/scene_full_dump/{scene_id}/scene_full_dump.json",
        "reason": reason,
        "ranked_anchors": [
            {
                "mesh_id": row["mesh"].get("mesh_id"),
                "name": row["mesh"].get("name"),
                "score": round(row["score"], 4),
                "bbox_center_world": row["mesh"].get("bbox_center_world"),
                "bbox_extent_world": row["mesh"].get("bbox_extent_world"),
                "screen": row["mesh"].get("active_camera_projection"),
                "reasons": row["reasons"],
            }
            for row in scored[:30]
        ],
    }


def load_dump(dump_root: Path, scene_id: str) -> dict | None:
    path = dump_root / scene_id / "scene_full_dump.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_object_csv(path: Path, decisions: list[dict]) -> None:
    fields = ["scene_id", "name", "confidence", "candidate_count", "candidate_objects", "bbox_center", "bbox_extent", "preview_png"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            bbox = d["candidate_bbox_world"]
            writer.writerow(
                {
                    "scene_id": d["scene_id"],
                    "name": d.get("name") or "",
                    "confidence": d["confidence"],
                    "candidate_count": len(d["candidate_objects"]),
                    "candidate_objects": ";".join(d["candidate_objects"]),
                    "bbox_center": json.dumps(bbox["center"]),
                    "bbox_extent": json.dumps(bbox["extent"]),
                    "preview_png": d.get("preview_png") or "",
                }
            )


def write_background_csv(path: Path, decisions: list[dict]) -> None:
    fields = ["scene_id", "name", "confidence", "placement_world", "recommended_object_max_extent", "anchor_mesh", "preview_png"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            writer.writerow(
                {
                    "scene_id": d["scene_id"],
                    "name": d.get("name") or "",
                    "confidence": d["confidence"],
                    "placement_world": json.dumps(d["placement_world"]),
                    "recommended_object_max_extent": d["recommended_object_max_extent"],
                    "anchor_mesh": d.get("anchor_mesh") or "",
                    "preview_png": d.get("preview_png") or "",
                }
            )


def main() -> int:
    args = parse_args()
    classification = parse_classification(resolve_repo_path(args.classification))
    index = load_index(resolve_repo_path(args.index_json))
    dump_root = resolve_repo_path(args.dump_root)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    object_decisions: list[dict] = []
    background_decisions: list[dict] = []
    missing = {args.object_section: [], args.background_section: []}

    for sid in classification.get(args.object_section, []):
        dump = load_dump(dump_root, sid)
        if dump is None:
            missing[args.object_section].append(sid)
            continue
        object_decisions.append(decide_object(sid, dump, index.get(sid), args.max_object_meshes))

    for sid in classification.get(args.background_section, []):
        dump = load_dump(dump_root, sid)
        if dump is None:
            missing[args.background_section].append(sid)
            continue
        background_decisions.append(decide_background(sid, dump, index.get(sid)))

    write_json(out_dir / "single_scene_light_good_object_decisions.json", object_decisions)
    write_object_csv(out_dir / "single_scene_light_good_object_decisions.csv", object_decisions)
    write_json(out_dir / "background_good_insertion_decisions.json", background_decisions)
    write_background_csv(out_dir / "background_good_insertion_decisions.csv", background_decisions)
    write_json(out_dir / "missing_scene_full_dump.json", missing)

    print(f"Wrote {out_dir / 'single_scene_light_good_object_decisions.json'} ({len(object_decisions)} scenes)")
    print(f"Wrote {out_dir / 'background_good_insertion_decisions.json'} ({len(background_decisions)} scenes)")
    print(f"Wrote {out_dir / 'missing_scene_full_dump.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
