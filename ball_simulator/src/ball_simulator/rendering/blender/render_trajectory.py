"""Run only inside Blender: blender -b --factory-startup --python this.py -- --job job.json"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def arguments() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(args)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def material(name: str, color: tuple[float, float, float, float], roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def striped_ball_material(scale: float):
    mat = bpy.data.materials.new("BallTextured")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = scale
    wave.inputs["Distortion"].default_value = 1.2
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (0.025, 0.055, 0.16, 1.0)
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[1].color = (0.88, 0.48, 0.06, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.42
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(wave.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def cube(name: str, location, scale, mat, pass_index: int):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    obj.pass_index = pass_index
    return obj


def build_environment(job: dict):
    render = job["render"]
    extent = float(render["fixed_visual_y_extent"])
    thickness = float(render["wall_thickness"])
    radius = float(job["radius"])
    surfaces = {s["surface_id"]: s for s in job["environment"]["surfaces"]}
    wall_mat = material("Walls", (0.22, 0.28, 0.36, 1.0), 0.72)
    floor_mat = material("Floor", (0.20, 0.17, 0.14, 1.0), 0.82)
    visuals = []
    for index, (surface_id, surface) in enumerate(surfaces.items(), start=2):
        point = np.asarray(surface["point"], dtype=float)
        normal = np.asarray(surface["normal"], dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        if axis == 0:  # vertical x wall
            location = (point[0] - normal[0] * thickness / 2.0, 0.0, 1.25)
            scale = (thickness, 2.0 * extent, 2.5)
            mat = wall_mat
        elif axis == 2:  # floor
            location = (float(job["environment"].get("channel_width") or 2.0) / 2.0, 0.0,
                        point[2] - normal[2] * thickness / 2.0)
            scale = (float(job["environment"].get("channel_width") or 2.0) + 2 * radius,
                     2.0 * extent, thickness)
            mat = floor_mat
        else:
            continue
        visuals.append(cube(surface_id, location, scale, mat, index))
    return visuals


def look_at(camera, target) -> None:
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(config: dict):
    data = bpy.data.cameras.new("Camera")
    data.lens = float(config["focal_length_mm"])
    data.sensor_width = float(config["sensor_width_mm"])
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(camera)
    camera.location = tuple(config["location"])
    look_at(camera, config["target"])
    bpy.context.scene.camera = camera
    return camera


def add_area_light(name: str, location, energy: float, size: float, target):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    direction = Vector(target) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return light


def setup_scene(job: dict):
    cfg = job["render"]
    scene = bpy.context.scene
    requested_engine = cfg["engine"]
    try:
        scene.render.engine = requested_engine
    except TypeError:
        # Blender 4.x uses BLENDER_EEVEE_NEXT; some Blender 5.x builds expose BLENDER_EEVEE.
        fallback = "BLENDER_EEVEE" if requested_engine == "BLENDER_EEVEE_NEXT" else "BLENDER_EEVEE_NEXT"
        scene.render.engine = fallback
    scene.render.resolution_x = int(cfg["width"])
    scene.render.resolution_y = int(cfg["height"])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(cfg["fps"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if cfg["transparent_background"] else "RGB"
    scene.render.film_transparent = bool(cfg["transparent_background"])
    scene.render.use_file_extension = True
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(cfg["samples"])
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(cfg["samples"])

    world = bpy.data.worlds.new("World") if bpy.data.worlds.get("World") is None else bpy.data.worlds["World"]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.035, 0.045, 0.065, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = float(cfg["lighting"]["world_strength"])

    add_camera(cfg["camera"])
    target = cfg["camera"]["target"]
    lighting = cfg["lighting"]
    add_area_light("Key", (4.0, -2.5, 5.0), float(lighting["key_energy"]), float(lighting["key_size"]), target)
    add_area_light("Fill", (-2.0, 1.5, 3.0), float(lighting["fill_energy"]), float(lighting["fill_size"]), target)


def setup_ball(job: dict):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=float(job["radius"]))
    ball = bpy.context.object
    ball.name = "Ball"
    ball.data.materials.append(striped_ball_material(float(job["render"]["texture_scale"])))
    ball.pass_index = 1
    bpy.ops.object.shade_smooth()
    return ball


def render_rgb(scene, output_root: Path, ball, states):
    rgb_dir = output_root / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(states["time"])):
        ball.location = tuple(float(x) for x in states["position"][i])
        x, y, z, w = (float(x) for x in states["quaternion_xyzw"][i])
        ball.rotation_mode = "QUATERNION"
        ball.rotation_quaternion = (w, x, y, z)
        bpy.context.view_layer.update()
        scene.frame_set(i + 1)
        scene.render.filepath = str(rgb_dir / f"{i:06d}.png")
        bpy.ops.render.render(write_still=True)


def render_semantic(scene, output_root: Path, ball, environment_objects, states):
    """Robust semantic masks via flat emission materials, no compositor/version coupling."""
    seg_dir = output_root / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    original_world = scene.world
    hidden_lights = [(obj, obj.hide_render) for obj in bpy.data.objects if obj.type == "LIGHT"]
    originals = {obj.name: list(obj.data.materials) for obj in [ball, *environment_objects]}
    colors = {
        "Ball": (1.0, 0.0, 0.0, 1.0),
        "floor": (0.0, 1.0, 0.0, 1.0),
        "left_wall": (0.0, 0.0, 1.0, 1.0),
        "right_wall": (1.0, 1.0, 0.0, 1.0),
        "wall": (0.0, 0.0, 1.0, 1.0),
    }
    mats = {}
    for name, color in colors.items():
        mat = bpy.data.materials.new(f"Seg_{name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 1.0
        mats[name] = mat
    for obj, _ in hidden_lights:
        obj.hide_render = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    for obj in [ball, *environment_objects]:
        obj.data.materials.clear()
        obj.data.materials.append(mats.get(obj.name, mats.get("wall")))
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    for i in range(len(states["time"])):
        ball.location = tuple(float(x) for x in states["position"][i])
        x, y, z, w = (float(x) for x in states["quaternion_xyzw"][i])
        ball.rotation_quaternion = (w, x, y, z)
        bpy.context.view_layer.update()
        scene.frame_set(i + 1)
        scene.render.filepath = str(seg_dir / f"{i:06d}.png")
        bpy.ops.render.render(write_still=True)
    for obj in [ball, *environment_objects]:
        obj.data.materials.clear()
        for mat in originals[obj.name]:
            obj.data.materials.append(mat)
    for obj, old in hidden_lights:
        obj.hide_render = old
    scene.world = original_world
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


def write_sidecar(job: dict, states, output_root: Path):
    np.savez_compressed(output_root / "states.npz", **{key: states[key] for key in states.files})
    metadata = {
        "schema_version": job["schema_version"],
        "trajectory_id": job["trajectory_id"],
        "environment": job["environment"],
        "render": job["render"],
        "frame_count": int(len(states["time"])),
        "quaternion_input_order": "xyzw",
        "blender_quaternion_order": "wxyz",
        "segmentation_colors_rgb": {
            "background": [0, 0, 0], "ball": [255, 0, 0], "floor": [0, 255, 0],
            "left_wall": [0, 0, 255], "right_wall": [255, 255, 0], "wall": [0, 0, 255]
        },
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_preview(job: dict, output_root: Path):
    if not job["render"]["outputs"]["preview_mp4"] or not job["render"]["outputs"]["rgb"]:
        return
    ffmpeg = "ffmpeg"
    command = [ffmpeg, "-y", "-framerate", str(job["render"]["fps"]),
               "-i", str(output_root / "rgb" / "%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               str(output_root / "preview.mp4")]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"[WARN] Could not create preview MP4: {error}")


def main() -> None:
    job = json.loads(arguments().job.read_text(encoding="utf-8"))
    output_root = Path(job["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    states = np.load(job["state_npz"])
    clear_scene()
    setup_scene(job)
    environment_objects = build_environment(job)
    ball = setup_ball(job)
    scene = bpy.context.scene
    if job["render"]["outputs"]["rgb"]:
        render_rgb(scene, output_root, ball, states)
    if job["render"]["outputs"]["segmentation"]:
        render_semantic(scene, output_root, ball, environment_objects, states)
    # Depth is intentionally deferred to EXR/render-pass implementation after RGB baseline validation.
    if job["render"]["outputs"]["depth"]:
        print("[WARN] Depth requested but not implemented in baseline patch; RGB/segmentation are rendered.")
    write_sidecar(job, states, output_root)
    make_preview(job, output_root)
    print(f"[OK] Rendered {job['trajectory_id']} to {output_root}")


if __name__ == "__main__":
    main()
