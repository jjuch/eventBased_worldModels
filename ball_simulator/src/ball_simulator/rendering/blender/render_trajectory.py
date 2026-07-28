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

TETRAHEDRAL_DIRECTIONS = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=float,
)

TETRAHEDRAL_DIRECTIONS /= np.linalg.norm(
    TETRAHEDRAL_DIRECTIONS,
    axis=1,
    keepdims=True,
)


MARKER_COLORS = (
    (0.90, 0.05, 0.25, 1.0),  # Red
    (0.05, 0.25, 0.95, 1.0),  # Blue
    (0.05, 0.80, 0.20, 1.0),  # Green
    (0.95, 0.95, 0.05, 1.0),  # Yellow
)


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
    bsdf.inputs["Alpha"].default_value = color[3]
    bsdf.inputs["Roughness"].default_value = roughness
    mat.blend_method = 'BLEND'
    # mat.shadow_method = 'HASHED'
    return mat

def clamp_color(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(
        max(0.0, min(1.0, float(component))) for component in color
    )

def lighten_color(color: tuple[float, float, float, float], amount: float) -> tuple[float, float, float, float]:
    red, green, blue, alpha = color
    return clamp_color(
        (red + amount, green + amount, blue + amount, alpha)
    )

def darken_color(color: tuple[float, float, float, float], amount: float) -> tuple[float, float, float, float]:
    red, green, blue, alpha = color
    return clamp_color(
        (red - amount, green - amount, blue - amount, alpha)
    )


def checkerboard_material(
    name: str, base_color: tuple[float, float, float, float],
    contrast: float, cell_counts: tuple[float, float, float],
    roughness: float = 0.75,
):
    material_data = bpy.data.materials.new(name)
    material_data.use_nodes = True

    nodes = material_data.node_tree.nodes
    links = material_data.node_tree.links

    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (520, 0)

    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (260, 0)
    shader.inputs["Alpha"].default_value = base_color[3]
    shader.inputs["Roughness"].default_value = roughness 

    texture_coordinates = nodes.new("ShaderNodeTexCoord")
    texture_coordinates.location = (-620, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-420, 0)
    mapping.inputs["Scale"].default_value = (
        float(cell_counts[0]),
        float(cell_counts[1]),
        float(cell_counts[2]),
    )

    checker = nodes.new("ShaderNodeTexChecker")
    checker.location = (-160, 0)
    checker.inputs["Color1"].default_value = lighten_color(base_color, contrast / 2.0)
    checker.inputs["Color2"].default_value = darken_color(base_color, contrast / 2.0)
    # Mapping already sets the physical frequency
    checker.inputs["Scale"].default_value = 1.0

    links.new(texture_coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(
        mapping.outputs["Vector"], checker.inputs["Vector"]
    )
    links.new(
        checker.outputs["Color"], shader.inputs["Base Color"]
    )
    links.new(
        shader.outputs["BSDF"], output.inputs["Surface"]
    )
    return material_data

def grid_line_material():
    return material(
        "BoundaryGrid",
        color=(0.66, 0.82, 0.92, 1.0),
        roughness=0.00
    )

def transparent_surface_material(
    name: str,
    color: tuple[float, float, float, float],
    alpha: float,
    roughness: float = 0.65,
):
    material_data = bpy.data.materials.new(name)
    material_data.use_nodes = True

    nodes = material_data.node_tree.nodes
    links = material_data.node_tree.links

    nodes.clear()

    output = nodes.new(
        "ShaderNodeOutputMaterial"
    )

    transparent_shader = nodes.new(
        "ShaderNodeBsdfTransparent"
    )

    transparent_shader.inputs[
        "Color"
    ].default_value = (
        color[0],
        color[1],
        color[2],
        1.0,
    )

    principled_shader = nodes.new(
        "ShaderNodeBsdfPrincipled"
    )

    principled_shader.inputs[
        "Base Color"
    ].default_value = color

    principled_shader.inputs[
        "Roughness"
    ].default_value = roughness

    mix_shader = nodes.new(
        "ShaderNodeMixShader"
    )

    # Alpha = 0 means fully transparent.
    # Alpha = 1 means fully opaque.
    mix_shader.inputs[
        "Fac"
    ].default_value = float(alpha)

    links.new(
        transparent_shader.outputs["BSDF"],
        mix_shader.inputs[1],
    )
    links.new(
        principled_shader.outputs["BSDF"],
        mix_shader.inputs[2],
    )
    links.new(
        mix_shader.outputs["Shader"],
        output.inputs["Surface"],
    )

    # Blender 4.2+.
    if hasattr(
        material_data,
        "surface_render_method",
    ):
        try:
            material_data.surface_render_method = (
                "DITHERED"
            )
        except TypeError:
            pass

    # Older Blender fallback.
    if hasattr(
        material_data,
        "blend_method",
    ):
        try:
            material_data.blend_method = "HASHED"
        except TypeError:
            pass

    return material_data


def marker_material(
    index: int,
    color: tuple[float, float, float, float],
):
    return material(
        name=f"BallMarkerMaterial_{index}",
        color=color,
        roughness=0.32,
    )

def add_beam(
        name: str,
        location: tuple[float, float, float],
        dimensions: tuple[float, float, float],
        beam_material, # bpy.material
        pass_index: int = 0,
):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)

    beam = bpy.context.object
    beam.name = name
    beam.dimensions = dimensions

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    beam.data.materials.append(beam_material)
    beam.pass_index = pass_index

    return beam


def add_wall_frame_and_grid(
    surface_id: str,
    plane_x: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    frame_thickness: float,
    grid_spacing: float,
    grid_thickness: float,
    show_frame: bool,
    show_grid: bool,
    frame_material,
) -> list:
    objects = []

    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    y_length = y_max - y_min
    z_height = z_max - z_min
    depth = max(frame_thickness, grid_thickness) 

    if show_frame:
        # Bottom and top horizontal rails
        for name, z_position in (
            ("bottom", z_min), ("top", z_max)
        ):
            objects.append(
                add_beam(
                    name=f"{surface_id}_frame_{name}",
                    location=(plane_x, y_center, z_position),
                    dimensions=(depth, y_length, frame_thickness),
                    beam_material=frame_material,
                )
            )

        # End posts in the visual y-direction
        for name, y_position in (
            ("front", y_min), ("back", y_max)
        ):
            objects.append(
                add_beam(
                    name=f"{surface_id}_frame_{name}",
                    location=(plane_x, y_position, z_center),
                    dimensions=(depth, frame_thickness, z_height),
                    beam_material=frame_material,
                )
            )

    if show_grid:
        # Lines parallel to z at regular y positions
        y_positions = np.arange(y_min + grid_spacing, y_max, grid_spacing)

        for index, y_position in enumerate(y_positions):
            objects.append(
                add_beam(
                    name=f"{surface_id}_grid_vertical_{index:03d}",
                    location=(plane_x, float(y_position), z_center),
                    dimensions=(depth, grid_thickness, z_height),
                    beam_material=frame_material,
                )
            )

        # Lines parallel to y at regular z positions
        z_positions = np.arange(z_min + grid_spacing, z_max, grid_spacing)
        
        for index, z_position in enumerate(z_positions):
            objects.append(
                add_beam(
                    name=f"{surface_id}_grid_horizontal_{index:03d}",
                    location=(plane_x, y_center, float(z_position)),
                    dimensions=(depth, y_length, grid_thickness),
                    beam_material=frame_material,
                )
            )

    return objects


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
    obj.dimensions = scale

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True,
    )
    obj.data.materials.append(mat)
    obj.pass_index = pass_index
    return obj


def build_environment(job: dict):
    render = job["render"]
    environment_render = render["environment"]
    visual_y_extent = float(render["fixed_visual_y_extent"])
    wall_thickness = float(render["wall_thickness"])
    frame_thickness = float(environment_render["frame_thickness"])
    grid_spacing = float(environment_render["grid_spacing"])
    grid_thickness = float(environment_render["grid_thickness"])
    show_frame = float(environment_render["show_near_wall_frame"])
    show_grid = float(environment_render["show_near_wall_grid"])

    radius = float(job["radius"])
    channel_width = float(job["environment"].get("channel_width") or 2.0)
    surfaces = {s["surface_id"]: s for s in job["environment"]["surfaces"]}
    camera_location = np.asarray(
            render["camera"]["location"], dtype=float
        )
    near_wall_id = camera_facing_wall(
        surfaces=surfaces,
        camera_location=camera_location
    )
    representation = environment_render["representation"]
    checker_size = float(environment_render["checker_size"])
    checker_contrast = float(environment_render["checker_contrast"])
    y_length = 2.0 * visual_y_extent
    wall_height = 2.0
    wall_cell_counts = (1.0, max(1.0, y_length / checker_size), max(1.0, wall_height / checker_size)) 
    floor_cell_counts = (max(1.0, channel_width / checker_size), max(1.0, y_length / checker_size), 1.0)

    opaque_wall_material = checkerboard_material("WallCheckerBoar", (0.27, 0.24, 0.43, 1.0),contrast=checker_contrast, cell_counts=wall_cell_counts, roughness=0.72)
    floor_material = checkerboard_material(name="FloorCheckerBoard",
                                           base_color=(0.28, 0.24, 0.20, 1.0), contrast=checker_contrast, cell_counts=floor_cell_counts, roughness=0.82)
    alpha = render["environment"]["near_wall_alpha"]
    transparent_wall_material = transparent_surface_material(
        name="NearWallTransparent",
        color=(0.42, 0.58, 0.68, 1.0),
        alpha=float(alpha),
        roughness=0.0,
    )
    boundary_material = grid_line_material()
    
    visual_objects = []
    
    for pass_index, (surface_id, surface) in enumerate(surfaces.items(), start=2):
        point = np.asarray(surface["point"], dtype=float)
        normal = np.asarray(surface["normal"], dtype=float)
        dominant_axis = int(np.argmax(np.abs(normal)))
        is_near_wall = (surface_id == near_wall_id)

        if dominant_axis == 0:  # vertical x wall
            plane_x = float(point[0])
            location = (plane_x - normal[0] * wall_thickness / 2.0, 0.0, wall_height / 2.0)
            scale = (wall_thickness, y_length, wall_height)
            selected_material = (
                transparent_wall_material if (
                    is_near_wall and representation in ("cutaway", "boundaries_only")
                ) else opaque_wall_material
            )
            visual_objects.append(
                cube(
                    name=surface_id,
                    location=location,
                    scale=scale,
                    mat=selected_material,
                    pass_index=pass_index
                )
            )
            if is_near_wall:
                visual_objects.extend(
                    add_wall_frame_and_grid(
                        surface_id=surface_id,
                        plane_x=plane_x,
                        y_min=-visual_y_extent,
                        y_max=visual_y_extent,
                        z_min=0.0,
                        z_max=wall_height,
                        frame_thickness=frame_thickness,
                        grid_spacing=grid_spacing,
                        grid_thickness=grid_thickness,
                        show_frame=show_frame,
                        show_grid=show_grid,
                        frame_material=boundary_material,
                    )
                )
        elif dominant_axis == 2:  # floor
            floor_z = float(point[2])
            location = (channel_width / 2.0, 0.0,
                        floor_z - normal[2] * wall_thickness / 2.0)
            scale = (channel_width + 2 * radius,
                     y_length, wall_thickness)

            visual_objects.append(
                cube(
                    name=surface_id,
                    location=location,
                    scale=scale,
                    mat=floor_material,
                    pass_index=pass_index
                )
            )


    return visual_objects


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

def camera_facing_wall(
    surfaces: list[dict],
    camera_location: np.ndarray,
) -> str:
    vertical_walls = []
    for surface_key in surfaces.keys():
        surface = surfaces[surface_key]
        normal = np.asarray(
            surface["normal"],
            dtype=float,
        )

        if abs(normal[0]) < 0.9:
            continue

        point = np.asarray(
            surface["point"],
            dtype=float,
        )

        distance = abs(
            np.dot(
                camera_location - point,
                normal,
            )
        )

        vertical_walls.append(
            (
                distance,
                surface["surface_id"],
            )
        )

    if not vertical_walls:
        raise ValueError(
            "No x-normal vertical walls found."
        )

    return min(vertical_walls)[1]


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
    radius = float(job["radius"])
    config = job["render"]["ball"]

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius, location=(0.0, 0.0, 0.0))
    ball = bpy.context.object
    ball.name = "Ball"
    ball.rotation_mode = "QUATERNION"
    ball.data.materials.append(
        material(
            "BallBase",
            tuple(config["base_color"]),
            float(config["roughness"])
        )
    )
    ball.pass_index = 1
    bpy.ops.object.shade_smooth()

    markers = []

    if config["markers_enabled"]:
        angular_radius = math.radians(
            float(config["marker_angular_radius_degree"])
        )
        marker_radius = radius * math.sin(angular_radius)
        center_distance= (
            radius + float(config["marker_surface_offset"])
            - 0.5 * marker_radius
        )

        for index, (direction, color) in enumerate(zip(
            TETRAHEDRAL_DIRECTIONS, MARKER_COLORS, strict=True,
        )):
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=24,
                ring_count=12,
                radius=marker_radius,
                location=(0.0, 0.0, 0.0),
            )
            marker = bpy.context.object
            marker.name = f"BallMarker_{index}"
            marker.parent = ball
            marker.location = tuple(direction * center_distance)
            marker.pass_index = 1
            marker.data.materials.append(marker_material(index, color))

            bpy.ops.object.shade_smooth()
            markers.append(marker)
    return ball, markers


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


def render_semantic(scene, output_root: Path, ball, ball_markers, environment_objects, states):
    """Robust semantic masks via flat emission materials, no compositor/version coupling."""
    seg_dir = output_root / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    original_world = scene.world
    hidden_lights = [(obj, obj.hide_render) for obj in bpy.data.objects if obj.type == "LIGHT"]
    ball_objects = [ball, *ball_markers]
    all_objects = [*ball_objects, *environment_objects]
    originals = {obj.name: list(obj.data.materials) for obj in all_objects}
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
    for obj in all_objects:
        obj.data.materials.clear()

        if obj in ball_objects: obj.data.materials.append(mats["Ball"])
        elif"floor" in obj.name: obj.data.materials.append(mats["floor"])
        elif "left_wall" in obj.name: obj.data.materials.append(mats["left_wall"])
        elif "right_wall" in obj.name: obj.data.materials.append(mats["right_wall"])
        else:
            obj.data.materials.append(mat["wall"])
    
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
    ball, ball_markers = setup_ball(job)
    scene = bpy.context.scene
    if job["render"]["outputs"]["rgb"]:
        render_rgb(scene, output_root, ball, states)
    if job["render"]["outputs"]["segmentation"]:
        render_semantic(scene, output_root, ball, ball_markers, environment_objects, states)
    # Depth is intentionally deferred to EXR/render-pass implementation after RGB baseline validation.
    if job["render"]["outputs"]["depth"]:
        print("[WARN] Depth requested but not implemented in baseline patch; RGB/segmentation are rendered.")
    write_sidecar(job, states, output_root)
    make_preview(job, output_root)
    print(f"[OK] Rendered {job['trajectory_id']} to {output_root}")


if __name__ == "__main__":
    main()
