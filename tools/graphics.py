"""
This module includes classes and functions for creating OpenGL graphics and manipulating data.
Some are new basic shapes that follow built-in pyglet.shapes methodology, while others
are custom graphics arrangements bundled with their own behaviour methods, animations etc.
"""

import numpy as np
import pyglet.image
from pyglet.window import Window
from pyglet.graphics.shader import ShaderProgram, Shader
from pyglet.image import Texture
from pyglet.graphics import Batch, Group
from pyglet.math import Vec3, Vec4
from pyglet.gl import *


def get_gl_lines_vertices_numpy(points) -> np.array:
    """
    Optimized version of get_gl_lines_vertices().
    Creates two trimmed copies of a ((x1, y1, z1), (x2,y2, z2), ...) point sequence
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


def get_gl_triangle_normals(vertices: tuple) -> tuple:
    """
    Calculate normals for given sequence of GL_TRIANGLES vertex data, using the cross product.
    Vertices should be in same format as the "position" attribute when creating VertexList:
    vertices = (x0, y0, z0, x1, y1, z1, x2, y2, z2 ...). Numpy required.
    """
    n_components = len(vertices)
    n_vertices = n_components / 3
    assert n_vertices % 3 == 0, "Expected 3 vertices per triangle and 3 components per vertex."
    triangles = n_components / 3
    array = np.array(vertices).reshape(-1, 3)
    normals = np.zeros((int(triangles), 3), dtype=np.float16)
    for i in range(0, len(array), 3):
        # assigning current triangle vertices
        v1, v2, v3 = array[i:i+3]
        # calculating edge vectors
        e1 = v2 - v1
        e2 = v3 - v1
        # calculating normal vector by cross product
        normal_vector = np.cross(e1, e2)
        # normalizing a normal vector
        magnitude = np.linalg.norm(normal_vector)
        # division by zero failsafe
        if magnitude > 1e-8:
            normal = normal_vector / magnitude
        else:
            normal = np.array([0.0, 0.0, 1.0])
        # assigning normal
        normals[i:i+3] = normal

    return normals.flatten().tolist()


class TextureGroup(Group):
    def __init__(
            self,
            texture: Texture,
            program: ShaderProgram,
            order=0, parent=None
    ):
        """
        A Group that enables and binds a Texture and ShaderProgram.
        TextureGroups are equal if their Texture and ShaderProgram
        are equal.
        :param texture: Texture to bind.
        :param program: Shader program to use.
        :param order: Change the order to render above or below other Groups.
        :param parent: Parent group.
        """
        super().__init__(order, parent)
        self.texture = texture
        self.program = program

    def set_state(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()

    def unset_state(self):
        glDisable(GL_BLEND)
        self.program.stop()

    def __hash__(self):
        return hash((self.texture.target, self.texture.id, self.order, self.parent, self.program))

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)


class TexturedPlane:
    def __init__(
            self,
            position: tuple[float, float, float],
            batch, group, program: ShaderProgram,
            length=300, height=200, rotation=(0, 0, 0),
    ):
        """
        A 2D textured plane in a 3D space.
        Position parameter is lower-left corner of the rectangle.
        Vertical rectangle is assumed at initiation, which is then rotated around lower-left corner
        with rotation parameter, representing a tuple of Oiler angles (pitch, yaw, roll) in radians.
        """
        self.position = position  # lower left corner of the rect
        self.x, self.y, self.z = position
        self.batch = batch
        self.group = group
        self.program = program
        self.length = length
        self.height = height
        self.rotation = rotation  # tuple of pitch, yaw, and roll angles in radians, respectively
        self.pitch, self.yaw, self.roll = rotation

        vertices = np.array((
            (self.x, self.y, self.z),
            (self.x + length, self.y, self.z),
            (self.x + length, self.y + height, self.z),
            (self.x, self.y + height, self.z)
        ), dtype=np.float32)

        vertex_normals = np.array((
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1)
        ), dtype=np.float32)

        # rotation factor
        vertices = rotate_points(vertices, self.pitch, self.yaw, self.roll, anchor=self.position)
        vertex_normals = rotate_points(vertex_normals, self.pitch, self.yaw, self.roll)

        # normalizing rotated normals
        magnitude = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / magnitude

        texture_coordinates = (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        )

        # indexing
        indices = (0, 1, 2, 0, 2, 3)
        gl_triangles_vertices = []
        tex_coords = []
        normals = []
        for idx in indices:
            gl_triangles_vertices.extend(vertices[idx])
            tex_coords.extend(texture_coordinates[idx])
            normals.extend(vertex_normals[idx])

        # creating vertex buffers
        count = len(gl_triangles_vertices) // 3
        self.vertex_list = self.program.vertex_list(
            count, GL_TRIANGLES, batch, group,
            position=('f', gl_triangles_vertices),
            normals=('f', normals),
            tex_coords=('f', tex_coords),
        )


class WireframeCube:
    """Barebone vertex list of a wireframe GL_LINES cube will be attached to a given program."""
    def __init__(
            self,
            program: ShaderProgram,
            batch: Batch,
            group: Group = None,
            color=(255, 255, 255, 255),
            position=Vec3(0, 0, 0),
            length=100
    ):
        self.program = program
        self.batch = batch
        self.group = group
        self.r, self.g, self.b = color[:3]
        self.a = color[3] if len(color) == 4 else 255
        self.color = self.r, self.g, self.b, self.a
        self.position = position
        self.length = length

        self.vertices = self.get_vertices()
        self.gl_line_vertices = self.get_gl_lines_vertices()
        self.count = len(self.gl_line_vertices) // 3
        self.colors = self.color * self.count

        self.vertex_list = self.program.vertex_list(
            self.count,
            GL_LINES,
            self.batch,
            self.group,
            position=('f', self.gl_line_vertices),
            colors=('Bn', self.colors)
        )

    def get_vertices(self):
        x, y, z = self.position.x, self.position.y, self.position.z
        hl = self.length / 2
        return (
            # front face
            (x - hl, y - hl, z + hl),  # 0
            (x + hl, y - hl, z + hl),  # 1
            (x + hl, y + hl, z + hl),  # 2
            (x - hl, y + hl, z + hl),  # 3
            # back face
            (x - hl, y - hl, z - hl),  # 4
            (x + hl, y - hl, z - hl),  # 5
            (x + hl, y + hl, z - hl),  # 6
            (x - hl, y + hl, z - hl),  # 7
        )

    def get_gl_lines_vertices(self):
        indices = [
            0, 1, 1, 2, 2, 3, 3, 0,  # construct front face
            4, 5, 5, 6, 6, 7, 7, 4,  # construct back face
            0, 4, 1, 5, 2, 6, 3, 7  # connect two faces
        ]

        gl_line_vertices = []
        for idx in indices:
            gl_line_vertices.extend(self.vertices[idx])

        return gl_line_vertices


class Cuboid:
        def __init__(
                self,
                program: ShaderProgram,
                batch: Batch,
                position=(0, 0, 0),
                size=(200, 200, 200),
                texture: Texture = None,
                color=None
        ):
            self.program = program
            self.batch = batch
            self.position = position
            self.size = size
            self.texture = texture
            if self.texture:
                self.group = TextureGroup(self.texture, self.program)
            else:
                self.group = None
            self.color = color or (255, 255, 255, 255)

            self.vertices = self._get_vertices()
            self.gl_triangles_vertices = self._get_gl_triangles_vertices()
            self.normals = self._get_normals()
            self.colors = self._get_colors()
            self.tex_coords = self._get_tex_coords()

            self.count = len(self.gl_triangles_vertices) // 3
            self.vertex_list = self.program.vertex_list(
                self.count,
                pyglet.gl.GL_TRIANGLES,
                self.batch,
                self.group,
                position=('f', self.gl_triangles_vertices),
                normals=('f', self.normals),
                colors=('f', self.colors),
                tex_coords=('f', self.tex_coords)
            )

        def _get_vertices(self) -> tuple:
            # center of the cuboid
            x, y, z = self.position[0], self.position[1], self.position[2]
            # half lengths
            lxh, lyh, lzh, = self.size[0] / 2, self.size[1] / 2, self.size[2] / 2
            vertices = (
                # front face
                (x - lxh, y - lyh, z + lzh),  # 0
                (x + lxh, y - lyh, z + lzh),  # 1
                (x + lxh, y + lyh, z + lzh),  # 2
                (x - lxh, y + lyh, z + lzh),  # 3
                # back face
                (x - lxh, y - lyh, z - lzh),  # 4
                (x + lxh, y - lyh, z - lzh),  # 5
                (x + lxh, y + lyh, z - lzh),  # 6
                (x - lxh, y + lyh, z - lzh),  # 7
            )
            return vertices

        def _get_gl_triangles_vertices(self) -> tuple:
            indices = [
                0, 1, 2, 0, 2, 3,  # front face
                5, 4, 7, 5, 7, 6,  # back face
                4, 0, 3, 4, 3, 7,  # left face
                1, 5, 6, 1, 6, 2,  # right face
                3, 2, 6, 3, 6, 7,  # top face
                4, 5, 1, 4, 1, 0,  # bottom face
            ]
            triangle_vertices = ()
            for idx in indices:
                triangle_vertices += self.vertices[idx]
            return triangle_vertices

        def _get_normals(self):
            return get_gl_triangle_normals(self.gl_triangles_vertices)

        @staticmethod
        def _get_tex_coords() -> tuple:
            tex_coords = (
                             0.0, 0.0,
                             1.0, 0.0,
                             1.0, 1.0,
                             0.0, 0.0,
                             1.0, 1.0,
                             0.0, 1.0,
                         ) * 6
            return tex_coords

        def _get_colors(self) -> tuple:
            r, g, b, a = self.color
            return (r / 255, g / 255, b / 255, a / 255) * (len(self.gl_triangles_vertices) // 3)

