import random

import pyglet.window
import numpy as np
import tools.graphics
import tools_old.graphics
from tools.definitions import *
from tools.camera import Camera3D
from pyglet.math import Vec3
from pyglet.image import Texture
from pyglet.graphics import Batch, Group
from pyglet.graphics.vertexdomain import VertexList
from pyglet.graphics.shader import ShaderProgram, Shader
from pyglet.gl import *

settings = {
    "default_mode": True,
    "width": 1280,
    "height": 720,
    "fps": 60,
    "resizable": True
}

_vertex_source = """#version 330 core
    in vec3 position;
    in vec2 tex_coords;
    out vec2 texture_coords;
    in vec4 colors;

    out vec4 frag_colors;

    uniform WindowBlock 
    {                       // This UBO is defined on Window creation, and available
        mat4 projection;    // in all Shaders. You can modify these matrixes with the
        mat4 view;          // Window.view and Window.projection properties.
    } window;  

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1.0);
        texture_coords = tex_coords;
        frag_colors = colors;
    }
"""

_fragment_source = """#version 330 core
    in vec2 texture_coords;
    in vec4 frag_colors;

    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        final_colors = frag_colors;
        final_colors = texture(our_texture, texture_coords);
    }
"""

vert_shader = Shader(_vertex_source, 'vertex')
frag_shader = Shader(_fragment_source, 'fragment')
shader_program = ShaderProgram(vert_shader, frag_shader)


class TextureGroup(Group):
    """A Group that enables and binds a Texture and ShaderProgram.

    TextureGroups are equal if their Texture and ShaderProgram
    are equal.
    """

    def __init__(self, texture, program, order=0, parent=None):
        """Create a TextureGroup.

        :Parameters:
            `texture` : `~pyglet.image.Texture`
                Texture to bind.
            `program` : `~pyglet.graphics.shader.ShaderProgram`
                ShaderProgram to use.
            `order` : int
                Change the order to render above or below other Groups.
            `parent` : `~pyglet.graphics.Group`
                Parent group.
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
        self.colors = self._get_colors()
        self.tex_coords = self._get_tex_coords()

        self.count = len(self.gl_triangles_vertices) // 3
        self.vertex_list = self.program.vertex_list(
            self.count,
            pyglet.gl.GL_TRIANGLES,
            self.batch,
            self.group,
            position=('f', self.gl_triangles_vertices),
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


class TexturedPlane:
    def __init__(self, position: Vec3, batch, group, program: ShaderProgram, texture: Texture, length=500, height=400,
                 rotation=0.0):
        self.position = position
        self.batch = batch
        self.group = group
        self.program = program
        self.texture = texture
        self.length = length
        self.height = height
        self.rotation = rotation

        vertices = np.array([
            (position.x, position.y, position.z),
            (position.x + length, position.y, position.z),
            (position.x + length, position.y + height, position.z),

            (position.x, position.y, position.z),
            (position.x + length, position.y + height, position.z),
            (position.x, position.y + height, position.z),
        ])

        if self.rotation:
            vertices = tools.graphics.rotate_points(vertices, rotation, anchor=self.position.xyz)

        self.normals = self.get_triangle_normals(vertices)

        self.vertices = vertices.flatten()

        self.tex_coords = [
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ]

        self.vertex_list = self._get_vertex_list()

    def _get_vertex_list(self) -> VertexList:
        count = len(self.vertices) // 3
        return self.program.vertex_list(
            count=count, mode=GL_TRIANGLES, batch=self.batch, group=self.group,
            position=('f', self.vertices),
            normals=('f', self.normals),
            tex_coords=('f', self.tex_coords)
        )

    @staticmethod
    def get_triangle_normals(triangle_vertices: list) -> list:
        """
        Calculate normals for list of triangle vertices using cross product.
        These should be passed in a form [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ...].
        TODO: Optimized by avoiding list->np.array->list conversions.
        TODO: How about processing position data in format [x0, y0, z0, x1, y1, z1 ...] directly?
        """
        assert len(triangle_vertices) % 3 == 0, "Check triangle vertices format, expected 3 vertices per triangle."
        assert len(triangle_vertices[0]) % 3 == 0, "Check vertex format, expected 3 components per vertex."

        normals = []
        for i in range(0, len(triangle_vertices), 3):
            # assigning current triangle vertices
            v1 = np.array(triangle_vertices[i])
            v2 = np.array(triangle_vertices[i + 1])
            v3 = np.array(triangle_vertices[i + 2])
            # calculating edge vectors
            e1 = v2 - v1
            e2 = v3 - v1
            # cross product of edges gives normal vector to triangle plane
            cross = np.cross(e1, e2)
            normalized = cross / np.linalg.norm(cross)
            # write normals for 3 vertices of a current triangle
            to_extend = normalized.tolist() * 3
            normals.extend(to_extend)

        return normals


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
        v1, v2, v3 = array[i:i + 3]
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
        normals[i:i + 3] = normal

    return normals.flatten().tolist()


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.program = shader_program
        self.camera = Camera3D(self, z_far=100_000)

        N = 10
        s = 300
        pyglet.resource.path = ['res/textures/']
        pyglet.resource.reindex()
        self.texture1 = pyglet.resource.texture('img_1.png')
        self.texture2 = pyglet.resource.texture('img_2.png')
        self.texture3 = pyglet.resource.texture('img.png')

        self.cubes = [
            Cuboid(self.program, self.batch,
                   texture=random.choice((self.texture1, self.texture2)),
                   color=(255, 50, 50, 0), position=Vec3(0 + i * s, 0 + j * s, 0 + k * s))
            for i in range(N)
            for j in range(N)
            for k in range(N)
        ]

        # self.cube = Cuboid(self.program, self.batch, size=(300, 10, 500), color=(255, 0, 0, 255), texture=self.texture2)

        glEnable(GL_DEPTH_TEST)

        group = TextureGroup(self.texture1, self.program)

    def on_draw(self) -> None:
        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()


if __name__ == '__main__':
    start_app(App, settings)
