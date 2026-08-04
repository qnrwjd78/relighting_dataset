"""Reusable object, direct-light, and object-shadow mask generation pipeline."""

from __future__ import annotations

import shutil
from array import array
from collections import deque
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ModuleNotFoundError as exc:  # pragma: no cover - Blender-only module
    raise SystemExit("scripts/relighting_mask_pipeline.py must be imported by Blender Python.") from exc


def load_exr_luminance(path: Path) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        channels = max(int(image.channels), 1)
        pixels = array("f", [0.0]) * (width * height * channels)
        image.pixels.foreach_get(pixels)
        arr = np.asarray(pixels, dtype=np.float32).reshape(height, width, channels)
        return 0.2126 * arr[..., 0] + 0.7152 * arr[..., min(1, channels - 1)] + 0.0722 * arr[..., min(2, channels - 1)]
    finally:
        bpy.data.images.remove(image)


def load_png_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        channels = max(int(image.channels), 1)
        pixels = array("f", [0.0]) * (width * height * channels)
        image.pixels.foreach_get(pixels)
        arr = np.asarray(pixels, dtype=np.float32).reshape(height, width, channels)
        mask = arr[..., 0] > 0.5
        if mask.shape != shape:
            raise RuntimeError(f"Mask shape mismatch for {path}: {mask.shape} != {shape}")
        return mask
    finally:
        bpy.data.images.remove(image)


def component_exr_path(scene_dir: Path, output: dict) -> Path:
    rel = output.get("exr") or output.get("primary")
    if not rel or not str(rel).lower().endswith(".exr"):
        raise RuntimeError(f"Expected EXR output, got {output}")
    return scene_dir / str(rel)


def positive_percentile(values: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return fallback
    scale = float(np.percentile(positive, percentile))
    return scale if scale > 1.0e-8 else fallback


def threshold_token(value: float) -> str:
    return f"{max(0, int(round(float(value) * 100.0))):03d}"


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    clean = np.zeros(mask.shape, dtype=bool)
    areas: list[int] = []
    kept_count = removed_count = kept_pixels = removed_pixels = 0
    for y in range(height):
        for x0 in np.flatnonzero(mask[y] & ~visited[y]):
            if visited[y, x0] or not mask[y, x0]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, int(x0))])
            visited[y, x0] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= height:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= width or (ny == cy and nx == cx):
                            continue
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            area = len(pixels)
            areas.append(area)
            if area >= min_area:
                kept_count += 1
                kept_pixels += area
                yy, xx = zip(*pixels)
                clean[np.asarray(yy), np.asarray(xx)] = True
            else:
                removed_count += 1
                removed_pixels += area
    return clean, {
        "component_count": len(areas),
        "kept_component_count": kept_count,
        "removed_component_count": removed_count,
        "kept_pixels": kept_pixels,
        "removed_pixels": removed_pixels,
        "area_min": int(min(areas)) if areas else 0,
        "area_max": int(max(areas)) if areas else 0,
    }


def mesh_shadow_snapshot(objects) -> dict[str, bool]:
    return {
        obj.name: bool(obj.visible_shadow)
        for obj in objects
        if obj.type == "MESH" and hasattr(obj, "visible_shadow")
    }


def set_mesh_shadow_visibility(objects, enabled: bool) -> None:
    for obj in objects:
        if obj.type == "MESH" and hasattr(obj, "visible_shadow"):
            obj.visible_shadow = bool(enabled)


def restore_mesh_shadow_visibility(snapshot: dict[str, bool]) -> None:
    for obj in bpy.data.objects:
        if obj.name in snapshot and hasattr(obj, "visible_shadow"):
            obj.visible_shadow = bool(snapshot[obj.name])


def build_subject_world_bvh(subject_objects) -> tuple[BVHTree, int]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    triangles = []
    for obj in subject_objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            base = len(vertices)
            vertices.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
            triangles.extend(tuple(base + index for index in triangle.vertices) for triangle in mesh.loop_triangles)
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise RuntimeError("Cannot build geometry shadow BVH: subject has no triangles")
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True), len(triangles)


def camera_ray(camera, inverse_projection, x: int, y: int, width: int, height: int) -> tuple[Vector, Vector]:
    ndc_x = 2.0 * (float(x) + 0.5) / float(width) - 1.0
    ndc_y = 2.0 * (float(y) + 0.5) / float(height) - 1.0
    camera_point = inverse_projection @ Vector((ndc_x, ndc_y, -1.0, 1.0))
    camera_point /= camera_point.w
    origin = camera.matrix_world.translation.copy()
    world_point = camera.matrix_world @ Vector(camera_point.xyz)
    return origin, (world_point - origin).normalized()


def receiver_surface_points(config: dict, camera, receiver_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    runtime = config.setdefault("_runtime", {})
    cache = runtime.get("geometry_shadow_receiver_points")
    height, width = receiver_mask.shape
    if cache is not None and cache.get("shape") == [height, width] and cache.get("camera") == camera.name:
        return cache["rows"], cache["cols"], cache["points"], cache["stats"]

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=float(scene.render.pixel_aspect_x),
        scale_y=float(scene.render.pixel_aspect_y),
    )
    inverse_projection = projection.inverted()
    rows = []
    cols = []
    points = []
    requested = int(receiver_mask.sum())
    for y, x in np.argwhere(receiver_mask):
        origin, direction = camera_ray(camera, inverse_projection, int(x), int(y), width, height)
        hit, location, _normal, _face, hit_object, _matrix = scene.ray_cast(depsgraph, origin, direction)
        if not hit or bool(hit_object.get("TL_SUBJECT", False)):
            continue
        rows.append(int(y))
        cols.append(int(x))
        points.append((float(location.x), float(location.y), float(location.z)))
    stats = {
        "receiver_mask_pixels": requested,
        "receiver_surface_points": len(points),
        "receiver_surface_coverage": float(len(points)) / float(max(requested, 1)),
    }
    cache = {
        "shape": [height, width],
        "camera": camera.name,
        "rows": np.asarray(rows, dtype=np.int32),
        "cols": np.asarray(cols, dtype=np.int32),
        "points": np.asarray(points, dtype=np.float32),
        "stats": stats,
    }
    runtime["geometry_shadow_receiver_points"] = cache
    return cache["rows"], cache["cols"], cache["points"], stats


def object_surface_points(
    config: dict,
    camera,
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    runtime = config.setdefault("_runtime", {})
    cache = runtime.get("geometry_direct_object_points")
    height, width = object_mask.shape
    if cache is not None and cache.get("shape") == [height, width] and cache.get("camera") == camera.name:
        return cache["rows"], cache["cols"], cache["points"], cache["normals"], cache["stats"]

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=float(scene.render.pixel_aspect_x),
        scale_y=float(scene.render.pixel_aspect_y),
    )
    inverse_projection = projection.inverted()
    rows = []
    cols = []
    points = []
    normals = []
    requested = int(object_mask.sum())
    for y, x in np.argwhere(object_mask):
        origin, direction = camera_ray(camera, inverse_projection, int(x), int(y), width, height)
        hit, location, normal, _face, hit_object, _matrix = scene.ray_cast(depsgraph, origin, direction)
        if not hit or not bool(hit_object.get("TL_SUBJECT", False)):
            continue
        rows.append(int(y))
        cols.append(int(x))
        points.append((float(location.x), float(location.y), float(location.z)))
        normals.append((float(normal.x), float(normal.y), float(normal.z)))
    stats = {
        "object_mask_pixels": requested,
        "object_surface_points": len(points),
        "object_surface_coverage": float(len(points)) / float(max(requested, 1)),
    }
    cache = {
        "shape": [height, width],
        "camera": camera.name,
        "rows": np.asarray(rows, dtype=np.int32),
        "cols": np.asarray(cols, dtype=np.int32),
        "points": np.asarray(points, dtype=np.float32),
        "normals": np.asarray(normals, dtype=np.float32),
        "stats": stats,
    }
    runtime["geometry_direct_object_points"] = cache
    return cache["rows"], cache["cols"], cache["points"], cache["normals"], stats


def geometry_object_shadow_mask(
    config: dict,
    camera,
    subject_objects,
    receiver_mask: np.ndarray,
    light_position: Vector,
) -> tuple[np.ndarray, dict]:
    runtime = config.setdefault("_runtime", {})
    bvh_cache = runtime.get("geometry_shadow_subject_bvh")
    if bvh_cache is None:
        subject_bvh, triangle_count = build_subject_world_bvh(subject_objects)
        bvh_cache = {"bvh": subject_bvh, "triangle_count": triangle_count}
        runtime["geometry_shadow_subject_bvh"] = bvh_cache
    rows, cols, points, receiver_stats = receiver_surface_points(config, camera, receiver_mask)
    shadow = np.zeros(receiver_mask.shape, dtype=bool)
    epsilon = max(float(config.get("_runtime", {}).get("canonical_scale", 1.0)) * 1.0e-5, 1.0e-6)
    occluded = 0
    for row, col, point_values in zip(rows, cols, points):
        point = Vector(point_values)
        to_light = light_position - point
        distance = float(to_light.length)
        if distance <= 2.0 * epsilon:
            continue
        direction = to_light / distance
        hit_location, _normal, _index, hit_distance = bvh_cache["bvh"].ray_cast(
            point + direction * epsilon,
            direction,
            distance - 2.0 * epsilon,
        )
        if hit_location is not None and hit_distance is not None:
            shadow[int(row), int(col)] = True
            occluded += 1
    return shadow, {
        **receiver_stats,
        "subject_triangle_count": int(bvh_cache["triangle_count"]),
        "ray_epsilon": epsilon,
        "occluded_receiver_pixels": occluded,
        "raw_shadow_ratio": float(shadow.mean()),
    }


def geometry_object_direct_lit_mask(
    config: dict,
    camera,
    subject_objects,
    object_mask: np.ndarray,
    light_position: Vector,
) -> tuple[np.ndarray, dict]:
    runtime = config.setdefault("_runtime", {})
    bvh_cache = runtime.get("geometry_shadow_subject_bvh")
    if bvh_cache is None:
        subject_bvh, triangle_count = build_subject_world_bvh(subject_objects)
        bvh_cache = {"bvh": subject_bvh, "triangle_count": triangle_count}
        runtime["geometry_shadow_subject_bvh"] = bvh_cache
    rows, cols, points, normals, object_stats = object_surface_points(config, camera, object_mask)
    direct_lit = np.zeros(object_mask.shape, dtype=bool)
    epsilon = max(float(runtime.get("canonical_scale", 1.0)) * 1.0e-5, 1.0e-6)
    front_facing = 0
    unoccluded = 0
    for row, col, point_values, normal_values in zip(rows, cols, points, normals):
        point = Vector(point_values)
        normal = Vector(normal_values).normalized()
        to_light = light_position - point
        distance = float(to_light.length)
        if distance <= 2.0 * epsilon:
            continue
        direction = to_light / distance
        if float(normal.dot(direction)) <= 0.0:
            continue
        front_facing += 1
        hit_location, _hit_normal, _index, hit_distance = bvh_cache["bvh"].ray_cast(
            point + direction * epsilon,
            direction,
            distance - 2.0 * epsilon,
        )
        if hit_location is None or hit_distance is None:
            direct_lit[int(row), int(col)] = True
            unoccluded += 1
    return direct_lit, {
        **object_stats,
        "subject_triangle_count": int(bvh_cache["triangle_count"]),
        "ray_epsilon": epsilon,
        "front_facing_object_pixels": front_facing,
        "unoccluded_direct_lit_pixels": unoccluded,
        "raw_direct_lit_ratio": float(direct_lit.mean()),
    }


def render_geometry_ray_masks(
    relight,
    scene_dir: Path,
    rel_base: str,
    config: dict,
    camera,
    subject_objects,
    light: dict,
    object_mask: np.ndarray,
    receiver_mask: np.ndarray,
    ambient_source: dict,
    args,
) -> dict:
    power_scale = float(light.get("power_scale", args.default_power_scale))
    light_position = Vector(light["world_position"])

    shadow_raw, geometry_stats = geometry_object_shadow_mask(
        config, camera, subject_objects, receiver_mask, light_position
    )
    shadow_raw &= ~object_mask
    shadow_clean, shadow_stats = remove_small_components(shadow_raw, int(args.min_area))
    shadow_pad = relight.dilate_binary(shadow_clean, int(args.pad_radius)) & ~object_mask

    direct_raw, direct_geometry_stats = geometry_object_direct_lit_mask(
        config, camera, subject_objects, object_mask, light_position
    )
    direct_clean, direct_stats = remove_small_components(direct_raw, int(args.min_area))

    mask_dir = scene_dir / f"{rel_base}_masks"
    shadow_stem = f"object_shadow_geometry_ray_clean_minarea{int(args.min_area):02d}"
    shadow_pad_stem = f"{shadow_stem}_pad{int(args.pad_radius):02d}"
    direct_stem = f"object_direct_lit_geometry_ray_clean_minarea{int(args.min_area):02d}"
    outputs = {
        "object_shadow_clean": f"{rel_base}_masks/{shadow_stem}.png",
        "object_shadow_clean_pad16": f"{rel_base}_masks/{shadow_pad_stem}.png",
        "object_direct_lit_clean": f"{rel_base}_masks/{direct_stem}.png",
    }
    relight.save_gray_png(scene_dir / outputs["object_shadow_clean"], shadow_clean.astype(np.float32))
    relight.save_gray_png(scene_dir / outputs["object_shadow_clean_pad16"], shadow_pad.astype(np.float32))
    relight.save_gray_png(scene_dir / outputs["object_direct_lit_clean"], direct_clean.astype(np.float32))
    np.save(mask_dir / f"{shadow_stem}.npy", shadow_clean.astype(np.uint8))
    np.save(mask_dir / f"{shadow_pad_stem}.npy", shadow_pad.astype(np.uint8))
    np.save(mask_dir / f"{direct_stem}.npy", direct_clean.astype(np.uint8))
    return {
        "white_full_occ": None,
        "white_no_object_shadow": None,
        "masks": outputs,
        "mask_stats": {
            "power_scale": power_scale,
            "white_render_world_energy": None,
            "shadow_threshold_mode": "object_only_geometry_ray",
            "direct_lit_mode": "front_facing_object_only_geometry_ray",
            "shadow_threshold": None,
            "ambient_subtracted_shadow_ratio": False,
            "object_shadow_clean_ratio": float(shadow_clean.mean()),
            "object_shadow_clean_pad16_ratio": float(shadow_pad.mean()),
            "object_direct_lit_clean_ratio": float(direct_clean.mean()),
            "geometry_ray": geometry_stats,
            "direct_lit_geometry_ray": direct_geometry_stats,
            "shadow_cleanup": shadow_stats,
            "direct_cleanup": direct_stats,
        },
    }


def render_ambient_white_reference(relight, scene_dir: Path, config: dict, subject_objects, ambient_source: dict) -> dict:
    white_config = relight.relighting_white_shading_config(config)
    relight.remove_all_lights()
    relight.set_ambient_source_from_meta(ambient_source, config)
    full_output = relight.render_white_diffuse_component(
        scene_dir, "mask_reference/ambient_white/full_occ", config, white_config, shadows_enabled=True
    )
    snapshot = mesh_shadow_snapshot(subject_objects)
    try:
        set_mesh_shadow_visibility(subject_objects, False)
        no_object_output = relight.render_white_diffuse_component(
            scene_dir, "mask_reference/ambient_white/no_object_shadow", config, white_config, shadows_enabled=True
        )
    finally:
        restore_mesh_shadow_visibility(snapshot)
    return {
        "full_occ": full_output,
        "no_object_shadow": no_object_output,
        "full_luminance": load_exr_luminance(component_exr_path(scene_dir, full_output)),
        "no_object_luminance": load_exr_luminance(component_exr_path(scene_dir, no_object_output)),
    }


def render_white_pair_and_masks(
    relight,
    scene_dir: Path,
    rel_base: str,
    config: dict,
    camera,
    subject_objects,
    light: dict,
    object_mask: np.ndarray,
    receiver_mask: np.ndarray,
    ambient_source: dict,
    ambient_white_reference: dict | None,
    args: argparse.Namespace,
) -> dict:
    if args.shadow_mask_mode == "geometry-ray":
        return render_geometry_ray_masks(
            relight,
            scene_dir,
            rel_base,
            config,
            camera,
            subject_objects,
            light,
            object_mask,
            receiver_mask,
            ambient_source,
            args,
        )
    white_config = relight.relighting_white_shading_config(config)
    power_scale = float(light.get("power_scale", args.default_power_scale))
    light_obj = relight.create_point_light(
        f"TL_FinalMask_{rel_base.replace('/', '_')}",
        Vector(light["world_position"]),
        float(light["world_energy"]) * power_scale,
        float(light["world_radius"]),
        [1.0, 1.0, 1.0],
    )
    try:
        relight.set_ambient_source_from_meta(ambient_source, config)
        full_output = relight.render_white_diffuse_component(
            scene_dir, f"{rel_base}_white/full_occ", config, white_config, shadows_enabled=True
        )
        snapshot = mesh_shadow_snapshot(subject_objects)
        try:
            set_mesh_shadow_visibility(subject_objects, False)
            no_object_output = relight.render_white_diffuse_component(
                scene_dir, f"{rel_base}_white/no_object_shadow", config, white_config, shadows_enabled=True
            )
        finally:
            restore_mesh_shadow_visibility(snapshot)
    finally:
        bpy.data.objects.remove(light_obj, do_unlink=True)
        relight.set_ambient_source_from_meta(ambient_source, config)

    full_y = load_exr_luminance(component_exr_path(scene_dir, full_output))
    no_object_y = load_exr_luminance(component_exr_path(scene_dir, no_object_output))
    point_response_scale = None
    if args.ambient_subtracted_shadow_ratio:
        if ambient_white_reference is None:
            raise RuntimeError("Ambient-subtracted shadow ratio requires an ambient white reference pair")
        point_occ = np.maximum(full_y - ambient_white_reference["full_luminance"], 0.0)
        point_no_object = np.maximum(no_object_y - ambient_white_reference["no_object_luminance"], 0.0)
        loss = np.maximum(point_no_object - point_occ, 0.0)
    else:
        point_no_object = None
        loss = np.maximum(no_object_y - full_y, 0.0)

    loss[object_mask] = 0.0
    shadow_scale = positive_percentile(loss, 99.0)
    if args.ambient_subtracted_shadow_ratio:
        point_response_scale = positive_percentile(point_no_object[receiver_mask], 99.0)
        support_floor = point_response_scale * max(float(args.shadow_support_threshold), 0.0)
        support = receiver_mask & (point_no_object > support_floor)
        denominator_floor = max(point_response_scale * 1.0e-6, 1.0e-8)
        shadow_ratio = loss / np.maximum(point_no_object, denominator_floor)
        shadow_raw = support & (shadow_ratio > float(args.shadow_threshold))
        threshold_mode = "ambient_subtracted_local_ratio"
    elif float(args.shadow_threshold) <= 0.0:
        shadow_raw = loss > 0.0
        threshold_mode = "any_positive_loss"
    else:
        shadow_raw = loss / shadow_scale > float(args.shadow_threshold)
        threshold_mode = "normalized_p99"

    shadow_clean, shadow_stats = remove_small_components(shadow_raw & ~object_mask, int(args.min_area))
    shadow_pad = relight.dilate_binary(shadow_clean, int(args.pad_radius)) & ~object_mask
    valid_scene = object_mask | receiver_mask
    direct_scale = positive_percentile(full_y[valid_scene], 95.0)
    direct_raw = object_mask & (full_y / direct_scale >= float(args.direct_lit_threshold))
    direct_clean, direct_stats = remove_small_components(direct_raw, int(args.min_area))

    mask_dir = scene_dir / f"{rel_base}_masks"
    metric = "ratio" if args.ambient_subtracted_shadow_ratio else "loss"
    shadow_stem = f"object_shadow_{metric}_thr{threshold_token(args.shadow_threshold)}_clean_minarea{int(args.min_area):02d}"
    shadow_pad_stem = f"{shadow_stem}_pad{int(args.pad_radius):02d}"
    direct_stem = f"object_direct_lit_thr{threshold_token(args.direct_lit_threshold)}_clean_minarea{int(args.min_area):02d}"
    outputs = {
        "object_shadow_clean": f"{rel_base}_masks/{shadow_stem}.png",
        "object_shadow_clean_pad16": f"{rel_base}_masks/{shadow_pad_stem}.png",
        "object_direct_lit_clean": f"{rel_base}_masks/{direct_stem}.png",
    }
    relight.save_gray_png(scene_dir / outputs["object_shadow_clean"], shadow_clean.astype(np.float32))
    relight.save_gray_png(scene_dir / outputs["object_shadow_clean_pad16"], shadow_pad.astype(np.float32))
    relight.save_gray_png(scene_dir / outputs["object_direct_lit_clean"], direct_clean.astype(np.float32))
    np.save(mask_dir / f"{shadow_stem}.npy", shadow_clean.astype(np.uint8))
    np.save(mask_dir / f"{shadow_pad_stem}.npy", shadow_pad.astype(np.uint8))
    np.save(mask_dir / f"{direct_stem}.npy", direct_clean.astype(np.uint8))
    return {
        "white_full_occ": full_output,
        "white_no_object_shadow": no_object_output,
        "masks": outputs,
        "mask_stats": {
            "power_scale": power_scale,
            "white_render_world_energy": float(light["world_energy"]) * power_scale,
            "white_pair_world_source_type": str(ambient_source.get("type", "unknown")),
            "white_pair_world_source_path": ambient_source.get("path"),
            "white_pair_semantics": "white diffuse point light plus selected scene ambient source",
            "shadow_p99_positive": shadow_scale,
            "shadow_threshold": float(args.shadow_threshold),
            "ambient_subtracted_shadow_ratio": bool(args.ambient_subtracted_shadow_ratio),
            "shadow_threshold_mode": threshold_mode,
            "shadow_support_threshold": float(args.shadow_support_threshold) if args.ambient_subtracted_shadow_ratio else None,
            "point_response_p99_receiver": point_response_scale,
            "direct_p95_positive_valid_scene": direct_scale,
            "object_shadow_clean_ratio": float(shadow_clean.mean()),
            "object_shadow_clean_pad16_ratio": float(shadow_pad.mean()),
            "object_direct_lit_clean_ratio": float(direct_clean.mean()),
            "shadow_cleanup": shadow_stats,
            "direct_cleanup": direct_stats,
        },
    }


def copy_mask_bundle(scene_dir: Path, source_masks: dict, rel_base: str) -> dict:
    dst_dir = scene_dir / f"{rel_base}_masks"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, rel in source_masks.items():
        src = scene_dir / rel
        dst = dst_dir / Path(rel).name
        shutil.copy2(src, dst)
        copied[key] = f"{rel_base}_masks/{dst.name}"
        npy_src = src.with_suffix(".npy")
        if npy_src.exists():
            shutil.copy2(npy_src, dst.with_suffix(".npy"))
    return copied
