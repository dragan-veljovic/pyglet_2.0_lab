"""
A small test to compare (vast) speed difference between CPU and GPU rotation algorithms.
CPU rotation algorithm is not optimized as it include multiple list-nparray conversions,
but nevertheless it shows how much faster shaders are.
"""
from tools.definitions import *
from tools.camera import Camera3D

from pyglet.math import Vec3
from pyglet.graphics.shader import ShaderProgram
from tools.graphics import rotate_points, WireframeCube

SETTINGS = {
    # to overrides all other settings, and start in default mode?
    "default_mode": True,
    # alternatively specify a number of custom arguments
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

        self.run = False
        self.time = 0.0
        self.fps = 60
        self.CPU_rotation = False

        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glLineWidth(2)
        self.label = pyglet.text.Label(
            "Press space to toggle CPU/GPU rotation", 0, 0, 0, batch=self.batch, font_size=25
        )

        # cube array l x h x d!
        l = 30
        h = 20
        d = 20
        length = 100
        sep = 0

        self.cubes = []
        for i in range(-l // 2, l // 2):
            for j in range(-h // 2, h // 2):
                for k in range(-d // 2, d // 2):
                    self.cubes.append(
                        WireframeCube(
                            self.program, self.batch, color=(255, 128 + j * 30, 128 + k * 30), length=100,
                            position=Vec3((length + sep) * i, (length + sep) * j, (length + sep) * k),
                        )
                    )

    def rotate_cubes(self):
        """A rotation of cubes around 0,0,0 with the CPU"""
        for cube in self.cubes:
            cube.vertices = rotate_points(cube.vertices, yaw=1/60)
            cube.vertex_list.position[:] = cube.get_gl_lines_vertices()

    def toggle_rotation_mode(self):
        self.CPU_rotation = not self.CPU_rotation

    def on_draw(self) -> None:
        self.time += 1/self.fps
        if self.CPU_rotation:
            self.rotate_cubes()
        else:
            self.program['time'] = self.time

        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
        if symbol == pyglet.window.key.SPACE:
            self.toggle_rotation_mode()


if __name__ == '__main__':
    start_app(App, SETTINGS)
