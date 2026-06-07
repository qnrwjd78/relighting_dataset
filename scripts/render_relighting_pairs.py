import bpy
import os
import sys
import json
import math
import argparse
from mathutils import Vector, Matrix


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def look_at(obj, target):
    target = Vector(target)
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def safe_set_enum(obj, attr, value):
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


# ============================================================
# Cycles high-quality setup
# ============================================================

def setup_cycles(width=1536, height=1536, samples=2048, use_gpu=True):
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False

    scene.cycles.samples = samples
    scene.cycles.preview_samples = min(256, samples)

    scene.cycles.use_denoising = True
    safe_set_enum(scene.cycles, "denoiser", "OPENIMAGEDENOISE")

    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.002
    scene.cycles.adaptive_min_samples = 256

    scene.cycles.max_bounces = 16
    scene.cycles.diffuse_bounces = 8
    scene.cycles.glossy_bounces = 10
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.volume_bounces = 2

    # 자동차 유리/금속에서 reflection을 살리기 위한 설정
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = False
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 10

    # PNG preview용. EXR은 linear HDR로 저장됨.
    if not safe_set_enum(scene.view_settings, "view_transform", "AgX"):
        safe_set_enum(scene.view_settings, "view_transform", "Filmic")

    safe_set_enum(scene.view_settings, "look", "Medium High Contrast")
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if use_gpu:
        setup_gpu_cycles()


def setup_gpu_cycles():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences

        # Windows NVIDIA면 보통 OPTIX 또는 CUDA
        for device_type in ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]:
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()

                has_gpu = False
                for dev in prefs.devices:
                    if dev.type != "CPU":
                        dev.use = True
                        has_gpu = True
                    else:
                        dev.use = False

                if has_gpu:
                    bpy.context.scene.cycles.device = "GPU"
                    print(f"[INFO] Cycles GPU enabled: {device_type}")
                    return
            except Exception:
                pass

        print("[WARN] No GPU found. Falling back to CPU.")
        bpy.context.scene.cycles.device = "CPU"

    except Exception as e:
        print(f"[WARN] GPU setup failed: {e}")
        bpy.context.scene.cycles.device = "CPU"


# ============================================================
# Asset loading
# ============================================================

def load_asset(asset_path):
    ext = os.path.splitext(asset_path)[1].lower()

    # 가장 중요: .blend는 append가 아니라 open_mainfile로 연다.
    # 그래야 원래 material / modifier / custom normal을 최대한 보존함.
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=asset_path)

        # 원본 파일에 있던 카메라/조명은 제거하고 geometry/material은 유지
        for obj in list(bpy.context.scene.objects):
            if obj.type in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)

        mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not mesh_objs:
            raise RuntimeError("No mesh objects found in blend file.")

        return mesh_objs

    # glb/obj/fbx도 지원은 하지만, 품질은 .blend 직접 로드가 더 좋음.
    before = set(bpy.context.scene.objects)

    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=asset_path)

    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=asset_path)

    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=asset_path)
        else:
            bpy.ops.import_scene.obj(filepath=asset_path)

    else:
        raise ValueError(f"Unsupported asset format: {ext}")

    after = set(bpy.context.scene.objects)
    imported = list(after - before)

    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("No mesh objects found after loading asset.")

    return mesh_objs


# ============================================================
# Bounds / normalization
# ============================================================

def compute_bounds(objs):
    bpy.context.view_layer.update()

    min_corner = Vector((1e9, 1e9, 1e9))
    max_corner = Vector((-1e9, -1e9, -1e9))

    for obj in objs:
        if obj.type != "MESH":
            continue

        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, wc.x)
            min_corner.y = min(min_corner.y, wc.y)
            min_corner.z = min(min_corner.z, wc.z)

            max_corner.x = max(max_corner.x, wc.x)
            max_corner.y = max(max_corner.y, wc.y)
            max_corner.z = max(max_corner.z, wc.z)

    center = (min_corner + max_corner) / 2.0
    size_vec = max_corner - min_corner
    max_size = max(size_vec.x, size_vec.y, size_vec.z)
    radius = size_vec.length / 2.0

    return min_corner, max_corner, center, max_size, radius


def normalize_objects(objs, target_size=3.2):
    """
    자동차처럼 body/wheel/glass가 여러 object로 나뉜 asset은
    object별로 따로 scale하면 부품이 벌어진다.
    그래서 전체 asset을 하나의 그룹처럼 world transform한다.
    """
    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)

    if max_size <= 0:
        raise RuntimeError("Invalid asset bounding box.")

    scale = target_size / max_size
    transform = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)

    for obj in objs:
        obj.matrix_world = transform @ obj.matrix_world

    bpy.context.view_layer.update()


# ============================================================
# Material / geometry enhancement
# ============================================================

def improve_materials(objs, add_bevel=True, add_weighted_normal=True):
    """
    자동차/하드서피스 모델 퀄리티 개선:
    - shade smooth
    - weighted normal
    - micro bevel
    - roughness 보정
    """
    for obj in objs:
        if obj.type != "MESH":
            continue

        try:
            for poly in obj.data.polygons:
                poly.use_smooth = True
        except Exception:
            pass

        if add_bevel:
            if not any(m.type == "BEVEL" and m.name == "HQ_Micro_Bevel" for m in obj.modifiers):
                bevel = obj.modifiers.new("HQ_Micro_Bevel", "BEVEL")
                bevel.width = 0.008
                bevel.segments = 2
                bevel.affect = "EDGES"
                bevel.harden_normals = True

        if add_weighted_normal:
            if not any(m.type == "WEIGHTED_NORMAL" for m in obj.modifiers):
                wn = obj.modifiers.new("HQ_Weighted_Normal", "WEIGHTED_NORMAL")
                wn.keep_sharp = True
                wn.weight = 50

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue

            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                continue

            roughness = bsdf.inputs.get("Roughness")
            if roughness is not None:
                # 너무 거울처럼 검게 죽는 걸 방지하되 원 재질은 최대한 유지
                roughness.default_value = max(0.18, min(0.62, roughness.default_value))

            alpha = bsdf.inputs.get("Alpha")
            if alpha is not None and alpha.default_value < 1.0:
                mat.blend_method = "BLEND"
                mat.use_screen_refraction = True if hasattr(mat, "use_screen_refraction") else False


# ============================================================
# Studio environment
# ============================================================

def make_principled_material(name, base_color, roughness=0.55, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic

    return mat


def create_studio_floor(objs, size=14.0):
    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)

    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(0, 0, min_corner.z - 0.006),
    )
    floor = bpy.context.object
    floor.name = "STUDIO_FLOOR"

    mat = make_principled_material(
        "studio_floor_mid_gray",
        (0.18, 0.18, 0.18, 1.0),
        roughness=0.62,
        metallic=0.0,
    )
    floor.data.materials.append(mat)

    return floor


def setup_world_ambient(strength=0.12, color=(0.05, 0.05, 0.055)):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputWorld")
    bg = nodes.new(type="ShaderNodeBackground")

    bg.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    bg.inputs["Strength"].default_value = strength

    links.new(bg.outputs["Background"], output.inputs["Surface"])


def set_world_envmap(envmap_path, strength=1.0):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputWorld")
    bg = nodes.new(type="ShaderNodeBackground")
    tex = nodes.new(type="ShaderNodeTexEnvironment")

    tex.image = bpy.data.images.load(envmap_path)
    bg.inputs["Strength"].default_value = strength

    links.new(tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])


# ============================================================
# Camera
# ============================================================

def setup_camera(objs, focal_length=75, view="front_3q"):
    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    distance = max_size * 2.25

    if view == "front_3q":
        cam.location = center + Vector((distance * 0.75, -distance * 1.15, distance * 0.42))
    elif view == "side":
        cam.location = center + Vector((distance * 1.35, -distance * 0.05, distance * 0.35))
    elif view == "rear_3q":
        cam.location = center + Vector((-distance * 0.85, distance * 1.05, distance * 0.40))
    else:
        cam.location = center + Vector((distance * 0.75, -distance * 1.15, distance * 0.42))

    look_target = center + Vector((0, 0, max_size * 0.05))
    look_at(cam, look_target)

    cam_data.lens = focal_length
    cam_data.sensor_width = 32
    cam_data.dof.use_dof = False

    bpy.context.scene.camera = cam
    return cam


# ============================================================
# Lights and reflection panels
# ============================================================

def remove_lights_and_reflectors():
    for obj in list(bpy.context.scene.objects):
        if (
            obj.type == "LIGHT"
            or obj.name.startswith("EMISSIVE_PROXY")
            or obj.name.startswith("REFLECT_")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)


def remove_lights_only():
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT" or obj.name.startswith("EMISSIVE_PROXY"):
            bpy.data.objects.remove(obj, do_unlink=True)


def create_area_light(name, location, target, energy, size, color):
    data = bpy.data.lights.new(name, type="AREA")
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)

    obj.location = Vector(location)
    look_at(obj, target)

    data.energy = energy
    data.size = size
    data.color = color

    return obj


def create_point_light(name, location, energy, color, shadow_soft_size=0.35):
    data = bpy.data.lights.new(name, type="POINT")
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)

    obj.location = Vector(location)
    data.energy = energy
    data.color = color
    data.shadow_soft_size = shadow_soft_size

    return obj


def create_reflection_panels(objs):
    """
    자동차/제품 렌더에서 퀄리티를 크게 좌우하는 반사 패널.
    차체와 유리에 밝은 strip highlight를 만들어준다.
    """
    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)

    def add_panel(name, loc, rot, scale, color, strength):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc, rotation=rot)
        panel = bpy.context.object
        panel.name = name
        panel.scale = scale

        mat = bpy.data.materials.new(name + "_mat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out = nodes.new(type="ShaderNodeOutputMaterial")
        em = nodes.new(type="ShaderNodeEmission")
        em.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
        em.inputs["Strength"].default_value = strength
        links.new(em.outputs["Emission"], out.inputs["Surface"])

        panel.data.materials.append(mat)
        return panel

    add_panel(
        "REFLECT_TOP_SOFTBOX",
        loc=center + Vector((0, -max_size * 0.25, max_size * 1.95)),
        rot=(math.radians(90), 0, 0),
        scale=(max_size * 1.9, max_size * 0.9, 1),
        color=(1.0, 0.96, 0.9),
        strength=4.5,
    )

    add_panel(
        "REFLECT_LEFT_STRIP",
        loc=center + Vector((-max_size * 1.65, -max_size * 0.45, max_size * 0.75)),
        rot=(math.radians(82), 0, math.radians(72)),
        scale=(max_size * 1.45, max_size * 0.23, 1),
        color=(1.0, 0.82, 0.58),
        strength=7.0,
    )

    add_panel(
        "REFLECT_RIGHT_STRIP",
        loc=center + Vector((max_size * 1.65, max_size * 0.35, max_size * 0.78)),
        rot=(math.radians(82), 0, math.radians(-72)),
        scale=(max_size * 1.25, max_size * 0.23, 1),
        color=(0.55, 0.70, 1.0),
        strength=5.5,
    )


def create_source_lighting(objs):
    """
    source_image용 기본 조명.
    너무 어둡지 않게 형태가 보이도록 구성.
    """
    remove_lights_only()
    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)

    setup_world_ambient(strength=0.18, color=(0.055, 0.055, 0.06))

    create_area_light(
        "source_large_key_softbox",
        location=center + Vector((-max_size * 1.3, -max_size * 1.7, max_size * 1.5)),
        target=center,
        energy=1700,
        size=max_size * 2.3,
        color=(1.0, 0.94, 0.86),
    )

    create_area_light(
        "source_front_fill_softbox",
        location=center + Vector((max_size * 1.5, -max_size * 1.2, max_size * 0.95)),
        target=center,
        energy=520,
        size=max_size * 2.9,
        color=(0.76, 0.84, 1.0),
    )

    create_area_light(
        "source_top_softbox",
        location=center + Vector((0.0, -max_size * 0.1, max_size * 2.25)),
        target=center,
        energy=700,
        size=max_size * 2.5,
        color=(1.0, 1.0, 1.0),
    )


def create_target_rig(objs):
    """
    target physical rig.
    이걸로 target_rig_render를 만들고,
    같은 rig를 emissive proxy로 바꿔 target_envmap.exr을 bake한다.
    """
    remove_lights_only()

    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)
    setup_world_ambient(strength=0.10, color=(0.04, 0.04, 0.045))

    rig_info = []

    key = create_area_light(
        "target_key_warm_softbox",
        location=center + Vector((-max_size * 1.45, -max_size * 1.55, max_size * 1.25)),
        target=center,
        energy=3000,
        size=max_size * 2.1,
        color=(1.0, 0.76, 0.52),
    )
    rig_info.append({
        "name": key.name,
        "type": "AREA",
        "position": list(key.location),
        "rotation": list(key.rotation_euler),
        "energy": key.data.energy,
        "size": key.data.size,
        "color": list(key.data.color),
    })

    rim = create_area_light(
        "target_cool_rim_strip",
        location=center + Vector((max_size * 1.65, max_size * 1.35, max_size * 1.15)),
        target=center,
        energy=1800,
        size=max_size * 0.9,
        color=(0.46, 0.62, 1.0),
    )
    rig_info.append({
        "name": rim.name,
        "type": "AREA",
        "position": list(rim.location),
        "rotation": list(rim.rotation_euler),
        "energy": rim.data.energy,
        "size": rim.data.size,
        "color": list(rim.data.color),
    })

    top = create_area_light(
        "target_top_soft_fill",
        location=center + Vector((0.0, -max_size * 0.1, max_size * 2.3)),
        target=center,
        energy=850,
        size=max_size * 2.5,
        color=(1.0, 0.96, 0.9),
    )
    rig_info.append({
        "name": top.name,
        "type": "AREA",
        "position": list(top.location),
        "rotation": list(top.rotation_euler),
        "energy": top.data.energy,
        "size": top.data.size,
        "color": list(top.data.color),
    })

    point = create_point_light(
        "target_small_specular_accent",
        location=center + Vector((-max_size * 1.1, -max_size * 0.9, max_size * 0.65)),
        energy=340,
        color=(1.0, 0.9, 0.75),
        shadow_soft_size=max_size * 0.08,
    )
    rig_info.append({
        "name": point.name,
        "type": "POINT",
        "position": list(point.location),
        "energy": point.data.energy,
        "shadow_soft_size": point.data.shadow_soft_size,
        "color": list(point.data.color),
    })

    return rig_info


# ============================================================
# Envmap baking
# ============================================================

def make_emission_material(name, color, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")

    emission.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
    emission.inputs["Strength"].default_value = strength

    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def create_emissive_proxy_from_rig(rig_info):
    proxies = []

    for light in rig_info:
        color = light["color"]

        if light["type"] == "AREA":
            bpy.ops.mesh.primitive_plane_add(
                size=light.get("size", 1.0),
                location=light["position"],
                rotation=light.get("rotation", (0, 0, 0)),
            )
            obj = bpy.context.object
            obj.name = "EMISSIVE_PROXY_area"

            strength = max(1.0, light["energy"] / 16.0)
            obj.data.materials.append(
                make_emission_material("proxy_area_emission", color, strength)
            )
            proxies.append(obj)

        elif light["type"] == "POINT":
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=64,
                ring_count=32,
                radius=max(0.12, light.get("shadow_soft_size", 0.25)),
                location=light["position"],
            )
            obj = bpy.context.object
            obj.name = "EMISSIVE_PROXY_point"

            strength = max(1.0, light["energy"] / 8.0)
            obj.data.materials.append(
                make_emission_material("proxy_point_emission", color, strength)
            )
            proxies.append(obj)

    return proxies


# ============================================================
# Render functions
# ============================================================

def render_image(path, width, height, file_format="PNG"):
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = path

    if file_format.upper() == "PNG":
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "16"
        scene.render.image_settings.compression = 15

    elif file_format.upper() == "EXR":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"

    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    bpy.ops.render.render(write_still=True)


def bake_target_envmap(path, rig_info, env_width=4096, env_height=2048):
    scene = bpy.context.scene
    old_camera = scene.camera
    old_samples = scene.cycles.samples

    proxies = create_emissive_proxy_from_rig(rig_info)

    cam_data = bpy.data.cameras.new("ENV_BAKE_CAMERA")
    cam = bpy.data.objects.new("ENV_BAKE_CAMERA", cam_data)
    bpy.context.collection.objects.link(cam)

    cam.location = (0, 0, 0)
    cam.rotation_euler = (0, 0, 0)
    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"

    scene.camera = cam

    hidden_states = []
    for obj in scene.objects:
        if obj.type == "MESH" and not obj.name.startswith("EMISSIVE_PROXY"):
            hidden_states.append((obj, obj.hide_render))
            obj.hide_render = True

    scene.cycles.samples = min(old_samples, 512)

    render_image(path, env_width, env_height, file_format="EXR")

    scene.cycles.samples = old_samples

    for obj, old_hide in hidden_states:
        obj.hide_render = old_hide

    scene.camera = old_camera

    bpy.data.objects.remove(cam, do_unlink=True)
    for p in proxies:
        bpy.data.objects.remove(p, do_unlink=True)


# ============================================================
# Main
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_path", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--env_width", type=int, default=4096)
    parser.add_argument("--env_height", type=int, default=2048)
    parser.add_argument("--env_strength", type=float, default=1.0)
    parser.add_argument("--target_size", type=float, default=3.2)
    parser.add_argument("--view", type=str, default="front_3q")

    parser.add_argument("--no_bevel", action="store_true")
    parser.add_argument("--no_weighted_normal", action="store_true")

    args = parser.parse_args(argv)

    args.asset_path = os.path.abspath(args.asset_path)
    args.out_dir = os.path.abspath(args.out_dir)
    args.render_width = args.width if args.width is not None else args.resolution
    args.render_height = args.height if args.height is not None else args.resolution

    ensure_dir(args.out_dir)

    # .blend는 open_mainfile로 열기 때문에 clear_scene을 먼저 하지 않는다.
    ext = os.path.splitext(args.asset_path)[1].lower()
    if ext != ".blend":
        clear_scene()

    objs = load_asset(args.asset_path)

    setup_cycles(
        width=args.render_width,
        height=args.render_height,
        samples=args.samples,
        use_gpu=True,
    )

    normalize_objects(objs, target_size=args.target_size)

    improve_materials(
        objs,
        add_bevel=not args.no_bevel,
        add_weighted_normal=not args.no_weighted_normal,
    )

    floor = create_studio_floor(objs)
    create_reflection_panels(objs)
    cam = setup_camera(objs, focal_length=75, view=args.view)

    # 1. source image
    create_source_lighting(objs)
    source_path = os.path.join(args.out_dir, "source_image.png")
    render_image(source_path, args.render_width, args.render_height, file_format="PNG")

    # 2. target physical rig render
    rig_info = create_target_rig(objs)
    target_rig_path = os.path.join(args.out_dir, "target_rig_render.png")
    render_image(target_rig_path, args.render_width, args.render_height, file_format="PNG")

    # 3. target envmap
    target_envmap_path = os.path.join(args.out_dir, "target_envmap.exr")
    bake_target_envmap(
        target_envmap_path,
        rig_info,
        env_width=args.env_width,
        env_height=args.env_height,
    )

    # 4. target envmap render
    remove_lights_only()
    set_world_envmap(target_envmap_path, strength=args.env_strength)
    target_env_path = os.path.join(args.out_dir, "target_env_render.png")
    render_image(target_env_path, args.render_width, args.render_height, file_format="PNG")

    min_corner, max_corner, center, max_size, radius = compute_bounds(objs)

    metadata = {
        "asset_path": args.asset_path,
        "source_image": source_path,
        "target_rig_render": target_rig_path,
        "target_envmap": target_envmap_path,
        "target_env_render": target_env_path,
        "render_settings": {
            "engine": "CYCLES",
            "resolution": args.resolution,
            "width": args.render_width,
            "height": args.render_height,
            "samples": args.samples,
            "env_width": args.env_width,
            "env_height": args.env_height,
            "env_strength": args.env_strength,
            "target_size": args.target_size,
            "view": args.view,
            "view_transform": bpy.context.scene.view_settings.view_transform,
            "look": bpy.context.scene.view_settings.look,
        },
        "camera": {
            "location": list(cam.location),
            "rotation_euler": list(cam.rotation_euler),
            "lens": cam.data.lens,
        },
        "bbox": {
            "min": list(min_corner),
            "max": list(max_corner),
            "center": list(center),
            "max_size": max_size,
            "radius": radius,
        },
        "target_rig": rig_info,
    }

    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[DONE]")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    main(argv)
