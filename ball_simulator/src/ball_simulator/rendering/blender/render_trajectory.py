"""Run only inside Blender: blender -b --factory-startup --python this.py -- --job job.json"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import shutils
from dataclasses import dataclass
from typing import Sequence

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

SEMANTIC_INDEX = {
    "background": 0,
    "Ball": 1,
    "BallMarker": 1,
    "floor": 2,
    "left_wall": 3,
    "right_wall": 4,
    "wall": 3,
}

SEMANTIC_COLOR = {
    0: (0.0, 0.0, 0.0, 1.0),
    1: (1.0, 0.0, 0.0, 1.0),
    2: (0.0, 1.0, 0.0, 1.0),
    3: (0.0, 0.0, 1.0, 1.0),
    4: (1.0, 1.0, 0.0, 1.0),
}


def arguments() -> argparse.Namespace:
    args = (
        sys.argv[sys.argv.index("--") + 1:] 
        if "--" in sys.argv 
        else []
    )
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--job",
        type=Path,
        help="A single render-job JSON file.",
    )
    group.add_argument(
        "--batch",
        type=Path,
        help="A batch JSON file containing multiple job paths.",
    )
    return parser.parse_args(args)


def semantic_index_for_object(obj) -> int:
    name = obj.name

    if name == "Ball" or name.startswith("BallMarker_"):
        return SEMANTIC_INDEX["Ball"]
    if name == "floor":
        return SEMANTIC_INDEX["floor"]
    if name == "left_wall":
        return SEMANTIC_INDEX["left_wall"]
    if name == "right_wall":
        return SEMANTIC_INDEX["right_wall"]
    if name == "wall":
        return SEMANTIC_INDEX["wall"]

    # Frames and grid lines belonging to a named wall receive
    # that wall's label.
    if name.startswith("left_wall"):
        return SEMANTIC_INDEX["left_wall"]
    if name.startswith("right_wall"):
        return SEMANTIC_INDEX["right_wall"]
    if name.startswith("wall_"):
        return SEMANTIC_INDEX["wall"]

    return SEMANTIC_INDEX["background"]


def assign_semantic_indices(ball, ball_markers, environment_objects) -> None:
    for obj in [
        ball,
        *ball_markers,
        *environment_objects,
    ]:
        obj.pass_index = semantic_index_for_object(obj)


def compositor_node(nodes, node_type: str, name: str, location: tuple[float, float]):
    node = nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = location
    return node


def create_index_mask(nodes, links, index_socket, class_index: int, y_position: float):
    """
    Return a value socket that equals 1 when the object index is
    class_index and 0 otherwise.
    """
    subtract = compositor_node(
        nodes,
        "CompositorNodeMath",
        f"Substract_{class_index}",
        (-500, y_position),
    )
    subtract.operation = "SUBSTRACT"
    subtract.inputs[1].default_value = float(class_index)
    links.new(index_socket, subtract.inputs[0])

    absolute = compositor_node(
        nodes,
        "CompositorNodeMath",
        f"Absolute_{class_index}",
        (-320, y_position),
    )
    absolute.operation = "ABSOLUTE"
    links.new(subtract.outputs[0], absolute.inputs[0])

    less_than = compositor_node(
        nodes,
        "CompositorNodeMath",
        f"Equal_{class_index}",
        (-140, y_position),
    )
    less_than.operation = "LESS_THAN"
    less_than.inputs[1].default_value = 0.5
    links.new(absolute.outputs[0], less_than.inputs[0])

    return less_than.outputs[0]


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


def configure_rgb_and_segmentation_outputs(
    scene,
    output_root: Path,
    *, 
    write_rgb: bool,
    write_segmentation: bool,
) -> None:
    """
    Configure Blender's compositor to write RGB and semantic
    segmentation from the same render.
    The semantic output is generated from the Object Index pass,
    so no material replacement or second render is required.
    """
    scene.use_nodes = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_object_index = write_segmentation

    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()

    render_layers = compositor_node(
        nodes,
        "CompositorNodeRLayers",
        "Render Layers",
        (-800, 100),
    )
    render_layers.layer = view_layer.name

    # Keep a Composite  output so Blender has a normal final image
    composite = compositor_node(
        nodes,
        "CompositorNodeComposite",
        "Composite",
        (750, 300),
    )
    links.new(
        render_layers.output["Image"],
        composite.inputs["Image"],
    )

    if write_rgb:
        rgb_output = compositor_node(
            nodes,
            "CompositorNodeOutputFile",
            "RGB Output",
            (750, 100),
        )
        rgb_output.base_path = str(output_root / "rgb")
        rgb_output.format.file_format = "PNG"
        rgb_output.format.color_mode = "RGBA" if scene.render.film_transparent else "RGB"
        rgb_output.format.color_depth = "8"
        rgb_output.file_slots[0].path = "frame_"

        links.new(
            render_layers.outputs["Image"],
            rgb_output.inputs[0],
        )


    if write_segmentation:
        segmentation_output = compositor_node(
            nodes,
            "CompositorNodeOutputFile",
            "Segmentation Output",
            (750, -220),
        )
        segmentation_output.base_path = str(output_root / "segmentation")
        segmentation_output.format.file_format = "PNG"
        segmentation_output.format.color_mode = "RGB"
        segmentation_output.format.color_depth = "8"
        segmentation_output.file_slots[0].path = "frame_"

        index_socket = render_layers.outputs.get("IndexOB")
        if index_socket is None:
            raise RuntimeError(
                "The active Blender render engine does not expose "
                "the Object Index compositor pass."
            )

        # Start the segmentation image at black
        segmentation_image = None
        for vertical_index, (class_index, color) in enumerate(
            sorted(SEMANTIC_COLOR.items())
        ):
            if class_index == 0:
                # Unmatched/background pixels remain black
                continue

            mask = create_index_mask(
                nodes, links, 
                index_socket=index_socket, 
                class_index=class_index, 
                y_position=-100.0 - 180.0 * vertical_index
            )

            color_node = compositor_node(
                nodes,
                "CompositorNodeRGB",
                f"SemanticColor_{class_index}",
                (80, -100.0 - 180.0 * vertical_index),
            )
            color_node.outputs[0].default_value = color

            multiply = compositor_node(
                nodes,
                "CompositorNodeMixRGB",
                f"MaskColor_{class_index}",
                (280, -100.0 - 180.0 * vertical_index),
            )
            multiply.blend_type = "MULTIPLY"
            multiply.inputs[0].default_value = 1.0

            links.new(
                color_node.ouputs[0],
                multiply.inputs[1],
            )
            links.new(
                mask,
                multiply.inputs[2],
            )

            if segmentation_image is None:
                segmentation_image = multiply.outputs[0]
            else:
                add = compositor_node(
                    nodes,
                    "CompositorNodeMixRBG",
                    f"AddSemantic_{class_index}",
                    (480, -100.0 - 180.0 * vertical_index),
                )
                add.blend_type = "ADD"
                add.inputs[0].default_value = 1.0

                links.new(
                    segmentation_image,
                    add.inputs[1],
                )
                links.new(
                    multiply.outputs[0],
                    add.inputs[2],
                )

                segmentation_image = add.outputs[0]

        if segmentation_image is None:
            raise RuntimeError(
                "NoSemantic classes were configured."
            )

        links.new(
            segmentation_image,
            segmentation_output.inputs[0]
        )


@dataclass
class RendererContext:
    scene: object
    environment_objects: list
    ball: object
    ball_markers: object
    environment_signature: str
    ball_signature: str


def environment_signature(job: dict) -> str:
    """
    Identify everything requiring rebuilding of camera, lighting,
    materials, and environment geometry.
    """
    payload = {
        "environment": job["environment"],
        "render": {
            "engine": job["render"]["engine"],
            "width": job["render"]["width"],
            "height": job["render"]["height"],
            "fps": job["render"]["fps"],
            "samples": job["render"]["samples"],
            "transparent_background": (
                job["render"]["transparent_background"]
            ),
            "fixed_visual_y_extent": (
                job["render"]["fixed_visual_y_extent"]
            ),
            "wall_thickness": (
                job["render"]["wall_thickness"]
            ),
            "camera": job["render"]["camera"],
            "lighting": job["render"]["lighting"],
            "environment": job["render"]["environment"],
        },
    }

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

def ball_signature(job: dict) -> str:
    payload = {
        "radius": float(job["radius"]),
        "ball": job["render"]["ball"],
    }

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def clean_output_directory(output_root: Path, job: dict) -> None:
    """
    Prevent stale frames from an earlier interrupted render from
    being mistaken for current output.
    """
    outputs = job["render"]["outputs"]

    if outputs["rgb"]:
        shutils.rmtree(output_root / "rgb", ignore_errors=True)

    if outputs["segmentation"]:
        shutils.rmtree(output_root / "segmentation", ignore_errors=True)

    success_marker = output_root / "_SUCCESS"
    if success_marker.exists():
        success_marker.unlink()


def render_one_job(context: RendererContext | None, job: dict) -> RendererContext:
    context = ensure_compatibility_context(context, job)
    output_root = Path(job["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)

    clean_output_directory(output_root, job)

    with np.load(job["state_npz"]) as loaded:
        states = {
            key: np.asarray(loaded[key])
            for key in loaded.files
        }

    outputs = job["render"]["outputs"]

    if outputs["depth"]:
        print(
            "[WARN] Depth requested but not implemented; "
            "RGB and segmentation will be rendered."
        )
    if outputs["rgb"] or outputs["segmentation"]:
        frame_count = render_rgb(
            scene=context.scene,
            output_root=output_root,
            ball=context.ball,
            ball_markers=context.ball_markers,
            environment_objects=context.environment_objects,
            states=states,
            write_rgb=bool(outputs["rgb"]),
            write_segmentation=bool(outputs["segmentation"])
        )
    else:
        frame_count = int(len(states["time"]))

    write_sidecar(job, states, output_root)
    make_preview(job, output_root)

    success_marker = output_root / "_SUCCESS"
    success_marker.write_text(
        json.dumps(
            {
                "trajectory_id": job["trajectory_id"],
                "frame_count": frame_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[OK] Rendered {job['trajectory_id']} "
        f"to {output_root}."
    )

    return context



def initialise_renderer(job:dict) -> RendererContext:
    clear_scene()
    setup_scene(job)

    environment_objects = build_environment(job)
    ball, ball_markers = setup_ball(job)

    return RendererContext(
        scene=bpy.context.scene,
        environment_objects=environment_objects,
        ball=ball,
        ball_markers=ball_markers,
        environment_signature=environment_signature(job),
        ball_signature=ball_signature(job)
    )


def ensure_compatibility_context(context: RendererContext | None, job: dict) -> RendererContext:
    required_environment = environment_signature(job)
    if (
        context is None
        or context.environment_signature != required_environment
    ):
        return initialise_renderer(job)

    required_ball = ball_signature(job)
    if context.ball_signature != required_ball:
        rebuild_ball(context, job)

    return context



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
    floor_margin = float(environment_render["floor_margin"])

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
            scale = (channel_width + 2.0 * floor_margin,
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
    else:
        configured = False
        if hasattr(scene, "eevee"):
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = int(cfg["samples"])
                configured = True

        if not configured:
            print(
                "[WARN] Could not configure EEVEE render "
                "samples through this Blender Python API.")

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


def remove_object_and_mesh(obj) -> None:
    mesh = obj.data if obj.type == "MESH" else None

    bpy.data.objects.remove(obj, do_unlink=True)
    if (
        mesh is not None
        and mesh.users == 0
    ):
        bpy.data.meshes.remove(mesh)


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


def rebuild_ball(context: RendererContext, job: dict) -> None:
    for marker in list(context.ball_markers):
        remove_object_and_mesh(marker)

    remove_object_and_mesh(context.ball)

    ball, ball_markers = setup_ball(job)

    context.ball = ball
    context.ball_markers = ball_markers
    context.ball_signature = ball_signature(job)


def animate_ball(scene, ball, states) -> int: 
    frame_count = len(states["time"])

    if frame_count < 1:
        raise ValueError("Cannot animate a trajectory with no states.")

    ball.animation_data_clear()
    ball.rotation_mode = "QUATERNION"

    for index in range(frame_count):
        frame = index + 1
        px, py, pz = tuple(
            float(x)
            for x in states["position"][index]
        )

        qx, qy, qz, qw = (
            float(x)
            for x in states["quaternion_xyzw"][index]
        )
        ball.location = (px, py, pz)
        ball.rotation_quaternion = (qw, qx, qy, qz)

        ball.keyframe_insert(
            data_path="location",
            frame=frame,
            group="Trajectory",
        )
        ball.keyframe_insert(
            data_path="rotation_quaternion",
            frame=frame,
            group="Trajectory",
        )
    # Simulation samples must not be smoothed between frames.
    # Linear interpolation preserves the supplied resampled states.
    if (
        ball.animation_data is not None
        and ball.animation_data.action is not None
    ):
        for fcurve in ball.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR" # Avoid Bezier interpolation as it focusses on visually smooth, but not physically correct.

    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.frame_step = 1

    # Force the first state to be evaluated before rendering starts.
    scene.frame_set(1)
    bpy.context.view_layer.update()

    return frame_count


def render_rgb(
    scene, 
    output_root: Path, 
    ball, 
    ball_markers, 
    environment_objects, 
    states,
    *,
    write_rgb: bool,
    write_segmentation: bool,
) -> int:
    """
    Animate one trajectory and render RGB and segmentation together.
    RGB and segmentation are emitted by compositor File Output nodes
    during the same Blender render operation.

    Returns
    -------
    int
        Number of rendered frames.
    """
    rgb_dir = output_root / "rgb"
    segmentation_dir = output_root / "segmentation"

    if write_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)
    if write_segmentation:
        segmentation_dir.mkdir(parents=True, exist_ok=True)

    assign_semantic_indices(
        ball=ball,
        ball_markers=ball_markers,
        environment_objects=environment_objects,
    )

    frame_count = animate_ball(scene, ball, states)

    configure_rgb_and_segmentation_outputs(
        scene=scene,
        output_root=output_root,
        write_rgb=write_rgb,
        write_segmentation=write_segmentation,
    )

    # File Output nodes write the actual dataset images.
    # This main filepath is only a harmless fallback.
    scene.render.filepath = str(output_root / "_render_unused_")


    bpy.ops.render.render(animation=True, write_still=False, use_viewport=False)

    return frame_count



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
    command = [ffmpeg, 
               "-y", 
               "-framerate", str(job["render"]["fps"]),
               "-start_number", "1",
               "-i", str(output_root / "rgb" / "frame_%04d.png"),
               "-c:v", "libx264", 
               "-pix_fmt", "yuv420p", 
               "-crf", "18",
               str(output_root / "preview.mp4")]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"[WARN] Could not create preview MP4: {error}")


def load_job(path: Path) -> dict:
    return json.loads(
        path.read_text(path.read_text(encoding="utf-8"))
    )


def load_batch(path: Path) -> list:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    job_paths = payload.get("jobs")

    if not isinstance(job_paths, list):
        raise ValueError(f"Batch file {path} has no 'jobs' list.")

    return [
        Path(job_path).resolve()
        for job_path in job_paths
    ]

def main() -> None:
    args = arguments()

    if args.job is not None:
        job_paths = [args.job.resolve()]
    else:
        job_paths = load_batch(args.batch.resolve())

    if not job_paths:
        print("[OK] No jobs in batch.")
        return

    context: RendererContext | None = None

    for job_path in job_paths:
        job = load_job(job_path)

        try:
            context = render_one_job(context, job)
        except Exception:
            print(
                f"[ERROR] Failed while rendering "
                f"{job_path}",
                file=sys.stderr,
            )
            raise

    print(
        f"[OK] Completed batch containing "
        f"{len(job_paths)} trajectories."
    )


if __name__ == "__main__":
    main()
