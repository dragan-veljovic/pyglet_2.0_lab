"""
A blueprint for wireframe Cube and example of rotation transformation in shader.
"""

from tools.definitions import *
from tools.camera import Camera3D

from pyglet.math import Vec3
from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics import Batch, Group
from pyglet.image import Texture
from pyglet.gl import GL_LINES, GL_TRIANGLES

SETTINGS = {
    # to overrides all other settings, and start in default mode?
    "default_mode": True,
    # alternatively specify a number of custom arguments
    "fps": 60,
    "width": 1920,
    "height": 1280,
    "resizable": True,
    "config": get_config(samples=4)
}


vertex_source = """
#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;

uniform float time;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

mat4 get_rotation_matrix(float theta){
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

void main() {
    mat4 rotation = get_rotation_matrix(time/2);
    vec3 new_position = position;
    gl_Position = window.projection * window.view * rotation * vec4(new_position, 1.0);
    vertex_colors = colors;
}"""

fragment_source = """
#version 330 core
in vec4 vertex_colors;
out vec4 final_colors;

uniform float time;

void main() {
    final_colors = vertex_colors;
}"""


class WireframeCube:
    def __init__(
            self,
            program: ShaderProgram,
            batch: Batch,
            group: Group = None,
            texture: Texture = None,
            color=(255, 255, 255, 255),
            position=Vec3(0, 0, 0),
            length=100
    ):
        self.program = program
        self.batch = batch
        self.group = group
        self.texture = texture
        self.r, self.g, self.b = color[:3]
        self.a = color[3] if len(color) == 4 else 255
        self.color = self.r, self.g, self.b, self.a
        self.position = position
        self.length = length

        self.gl_line_vertices = self.get_gl_lines_vertices()
        self.count = len(self.gl_line_vertices) // 3
        self.colors = self.color * self.count

        self.vertex_list = self.program.vertex_list(
            self.count,
            GL_LINES,
            self.batch,
            position=('f', self.gl_line_vertices),
            colors=('Bn', self.colors)
        )

    def get_gl_lines_vertices(self):
        x, y, z = self.position.x, self.position.y, self.position.z
        hl = self.length / 2
        vertices = (
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

        indices = [
            0, 1, 1, 2, 2, 3, 3, 0,  # construct front face
            4, 5, 5, 6, 6, 7, 7, 4,  # construct back face
            0, 4, 1, 5, 2, 6, 3, 7   # connect two faces
        ]

        gl_line_vertices = []
        for idx in indices:
            gl_line_vertices.extend(vertices[idx])

        return gl_line_vertices


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)
        self.program = ShaderProgram(
            Shader(vertex_source, "vertex"),
            Shader(fragment_source, "fragment")
        )

        self.run = True
        self.time = 0.0
        self.fps = SETTINGS['fps']

        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glLineWidth(2)

        # cube array l x h x d!
        l = 20
        h = 8
        d = 4
        length = 100
        sep = 0

        self.cubes = []
        for i in range(-l//2, l//2):
            for j in range(-h//2, h//2):
                for k in range(-d//2, d//2):
                    self.cubes.append(
                        WireframeCube(
                            self.program, self.batch, color=(255, 128 + j*30, 128 + k*30), length=100,
                            position=Vec3((length + sep)*i, (length + sep)*j, (length + sep)*k),
                        )
                    )

        self.cube = WireframeCube(self.program, self.batch, color=(255, 255, 255))

    def on_draw(self) -> None:
        if self.run:
            self.time += 1/self.fps
            self.program['time'] = self.time
        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run


if __name__ == '__main__':
    start_app(App, SETTINGS)
