import pyglet.math
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    'width': 1280,
    'height': 720,
    'fps': 60
}

vertex_source = """
#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    vertex_colors = colors;
}
"""

fragment_source = """
#version 330 core
in vec4 vertex_colors;
out vec4 final_colors;

void main() {
    final_colors = vertex_colors;
}
"""


class PointLight:
    def __init__(self, position: Vec3(0, 0, 0), color: (1.0, 1.0, 1.0), ambient=0.2, diffuse=1.0, specular=0.5):
        self.position = position
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

    def get_light_view(self):
        return pyglet.math.Mat4.look_at(self.position, Vec3(0, 0, 0), Vec3(0, 1, 0))


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)
        self.camera.look_at(Vec3(-200, 500, 500), Vec3(0, 0, 0))
        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )
        self.light = PointLight(Vec3(0, 0, 1000), color=(255, 255, 255, 255))

        # scene elements
        self.floor = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, rotation=(np.pi / 2, 0, 0),
            color=(150, 150, 150, 255)
        )
        self.back_wall = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, color=(150, 150, 150, 255)
        )
        self.cube = Cuboid(self.program, self.batch, color=(128, 0, 0, 255))

    def on_draw(self) -> None:
        self.clear()
        self.batch.draw()


if __name__ == '__main__':
    start_app(App, settings)
