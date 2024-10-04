"""
This module includes classes and functions for creating OpenGL graphics and manipulating data.
Some are new basic shapes that follow built-in pyglet.shapes methodology, while others
are custom graphics arrangements bundled with their own behaviour methods, animations etc.
"""

import numpy as np
from pyglet.window import Window
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Batch, Group
from pyglet.gl import GL_LINES, GL_TRIANGLES


def get_gl_lines_vertices_numpy(points) -> np.ndarray:
    """
    Optimized version of get_gl_lines_vertices().
    Creates two trimmed copies of a ((x1, y1), (x2,y2), ...) point sequence
    to create GL_LINES vertex list in a form (x1, y1, x2, y2, x2, y2, x3, y3, ...).
    """
    return np.column_stack((points[:-1], points[1:])).flatten()


def rotate_points(
        points: np.array,
        pitch: float = 0,
        yaw: float = 0,
        roll: float = 0,
        anchor: tuple[float, float, float] = (0, 0, 0)
) -> np.array:
    """
    Rotates an array of vertices in 3D space through specified Oiler angles and around given anchor.
    Pitch, yaw and roll are angles in radians to rotate around x, y, z axes respectively.
    Points should be passed as np.ndarray([[x1, y1, z1], [x2, y2, z2], ...]).
    """

    # prepare anchor vector
    anchor = np.array(anchor)

    # pre-calculate sin/cos terms
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)

    # x-axis rotation matrix (pitch)
    rot_x = np.array([
        [1, 0, 0],
        [0, cx, sx],
        [0, sx, cx]
    ])

    # y-axis rotation matrix (yaw)
    rot_y = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    # z-axis rotation matrix (roll)
    rot_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ])

    rotation_matrix = rot_z @ rot_y @ rot_x

    return (points - anchor) @ rotation_matrix + anchor


def get_lines(
        points: list,
        colors: list,
        program: ShaderProgram,
        batch: Batch = None,
        group: Group = None
):
    """Add a vertex list to a program, from passed GL_LINES data."""
    npoints = len(points) // 3
    return program.vertex_list(npoints, GL_LINES, batch=batch, group=group,
                               position=('f', points),
                               colors=('f', colors)
                               )


def get_grid(
        window: Window,
        program: ShaderProgram,
        batch=None,
        group=None,
        main_color=(92, 92, 92, 255),
        sub_color=(48, 48, 48, 255),
        main_div=100,
        sub_div=20,
):
    assert main_div % sub_div == 0, "Main division must be divisible by sub division."
    width = window.width
    height = window.height
    vlines = round(width // main_div, 1)
    hlines = round(height // main_div, 1)
    main_color = [component / 255 for component in main_color]
    sub_color = [component / 255 for component in sub_color]
    points = []
    colors = []

    # vertical lines
    for i in range(-vlines, vlines + 1):
        points.extend([i * main_div, 0, -hlines * main_div, i * main_div, 0, hlines * main_div, ])
        colors.extend(main_color * 2)
        if i < vlines:
            for k in range(sub_div, main_div, sub_div):
                points.extend(
                    [i * main_div + k, 0, -hlines * main_div, i * main_div + k, 0, hlines * main_div])
                colors.extend(sub_color * 2)

    # horizontal lines
    for j in range(-hlines, hlines + 1):
        points.extend([-vlines * main_div, 0, j * main_div, vlines * main_div, 0, j * main_div])
        colors.extend(main_color * 2)
        if j < hlines:
            for l in range(sub_div, main_div, sub_div):
                points.extend(
                    [-vlines * main_div, 0, j * main_div + l, vlines * main_div, 0, j * main_div + l])
                colors.extend(sub_color * 2)

    return get_lines(points, colors, program, batch, group)
