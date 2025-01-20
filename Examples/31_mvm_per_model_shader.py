import random

import pyglet.math
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    'default_mode': False,
    'width': 1280,
    'height': 720,
    'fps': 60
}

vertex_source = """
#version 330 core

// Input vertex attributes
in vec3 position;
in vec4 colors;

// Outputs to the fragment shader
out vec4 vertex_colors;

// Uniforms
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

// uniform mat4 model; // Optional model matrix for non-rotation transformations (e.g., scaling/translation)
uniform float angle; // Rotation angle around the Y-axis in radians

void main() {
    // Compute the Y-axis rotation matrix
    mat4 rotation_y = mat4(
        vec4(cos(angle), 0.0, sin(angle), 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(-sin(angle), 0.0, cos(angle), 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );

    // Combine the model matrix and rotation matrix
    mat4 final_model = rotation_y;

    // Transform the vertex position
    gl_Position = window.projection * window.view * final_model * vec4(position, 1.0);

    // Pass through vertex colors
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


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self, z_far=10_000)
        self.camera.look_at(Vec3(-200, 500, 500), Vec3(0, 0, 0))
        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )
        self.time = 0.0

        # scene elements
        self.floor = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, rotation=(np.pi / 2, 0, 0),
            color=(150, 150, 150, 128)
        )
        self.back_wall = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, color=(150, 150, 150, 255)
        )

        self.cubes = self.get_cubes(n=40)
        self.params = self.get_params()
        self.axes = self.get_axes()

        glEnable(GL_DEPTH_TEST)

        self.zero_rotation = pyglet.math.Mat4.from_rotation(0, Vec3(0, 1, 0))

    def get_cubes(self, n=10):
        cubes = []
        spacing = 50
        for i in range(n):
            x = i * 200 + spacing
            z = random.randint(-200, 200)
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            for j in range(n):
                y = j * 200 + spacing
                cubes.append(Cuboid(self.program, batch=self.batch, position=(x, y, z), color=(r, g, b, 255)))
        return cubes

    def get_params(self):
        return [random.randrange(-200, 200, 1) * 0.01 for _ in range(len(self.cubes))]

    def get_axes(self):
        return [random.choice([Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)]) for _ in range(len(self.cubes))]

    def on_draw(self) -> None:
        """1) Per model uniform updates method - shader rotation"""
        self.time += 1/60
        self.clear()
        self.program.use()

        # static objects
        self.program['angle'] = 0.0
        self.floor.vertex_list.draw(GL_TRIANGLES)
        self.back_wall.vertex_list.draw(GL_TRIANGLES)

        # rotation of objects
        for idx, cube in enumerate(self.cubes):
            # rotation = pyglet.math.Mat4.from_rotation(self.params[idx]*self.time, self.axes[idx])
            self.program['angle'] = self.time * self.params[idx]
            cube.vertex_list.draw(GL_TRIANGLES)


if __name__ == '__main__':
    start_app(App, settings)
