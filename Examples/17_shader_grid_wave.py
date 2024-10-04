import pyglet

import tools.camera
from tools.definitions import *
from tools.graphics import *
from pyglet.math import Vec3

WIDTH = 1280
HEIGHT = 720
FPS = 60

vert_source = """
#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;

uniform float dip_pos_x;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    // Calculate distance of vertex from dip position on x-axis
    // float distance_from_dip = position.x - dip_pos_x;

    // Calculate the displacement of vertices from the dip position
    vec3 disp_from_dip = position - vec3(dip_pos_x, 0, 0);

    // Calculate the magnitude of the displacement from the dip
    float disp_from_dip_mag = length(disp_from_dip);

    // Modifying position according to displacement from the dip
    vec3 modified_position = position;
    float dip_radius = 500;
    float dip_amplitude = 300;
    // modified_position.y += clamp(disp_from_dip_mag / dip_radius * disp_from_dip_mag, -300, +300);
    // or try this for crazy effect
    modified_position.y -= cos(clamp(disp_from_dip_mag / dip_radius , -300, +300)*2) * dip_amplitude;


    // Apply a sine-based falloff to create a smooth dip effect
    // float falloff = -cos(clamp(distance_from_dip / dip_width, - 3.14, 3.14));

    // Displace y-position based on the falloff (with some amplitude scaling)
    // float dip_displacement = dip_amplitude * falloff;

    // Modify the y-coordinate of the vertex if it's a horizontal grid line
    // vec3 modified_position = position;
    // modified_position.y += dip_displacement;

    gl_Position = window.projection * window.view * vec4(modified_position, 1.0);
    vertex_colors = colors;
}
"""

frag_source = """
#version 330 core
in vec4 vertex_colors;
out vec4 final_colors;

uniform float dip_pos_x;

void main() {
    final_colors = vertex_colors;
}
"""


def get_quantized_grid_gl_line_vertices(span: 2000, pitch=0, yaw=0, roll=0) -> np.array:
    grid_width = span
    grid_height = span
    spacing = 50

    nh = int(grid_height / spacing)
    nv = int(grid_width / spacing)

    points = []
    vertices = []

    for i in range(-nh, nh):
        for j in range(-nv, nv):
            points.append([i * spacing, j * spacing, 0])
        rotated_points = rotate_points(points, pitch, yaw, roll)
        vertices_this_pass = get_gl_lines_vertices_numpy(rotated_points)
        vertices.extend(vertices_this_pass)
        points = []
    #
    for i in range(-nh, nh):
        for j in range(-nv, nv):
            points.append([j * spacing, i * spacing, 0])
        rotated_points = rotate_points(points, pitch, yaw, roll)
        vertices_this_pass = get_gl_lines_vertices_numpy(rotated_points)
        vertices.extend(vertices_this_pass)
        points = []

    return vertices


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = tools.camera.Camera3D(self, z_near=1, z_far=10000)
        self.camera.look_at(Vec3(0, 500, 1000), Vec3(0, 0, 0))
        self.program = ShaderProgram(
            Shader(vert_source, 'vertex'),
            Shader(frag_source, 'fragment')
        )
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)

        self.create_scene()

    def set_dip(self, value):
        self.dip_pos_x = value
        self.circle.position = self.dip_pos_x, 600
        self.program['dip_pos_x'] = self.dip_pos_x

    def create_scene(self):
        self.time = 0.0
        self.dip_speed = 4
        grid_vertices = get_quantized_grid_gl_line_vertices(span=4000, pitch=np.pi / 2)
        self.grid = self.program.vertex_list(len(grid_vertices) // 3, GL_LINES, self.batch,
                                             position=('f', grid_vertices),
                                             colors=('Bn', (92, 92, 92, 255) * (len(grid_vertices) // 3)))
        self.dip_pos_x = -300
        self.circle = pyglet.shapes.Circle(self.dip_pos_x, 600, 100, batch=self.batch)

    def on_draw(self) -> None:
        self.time += 1 / FPS
        self.set_dip(self.dip_pos_x + self.dip_speed * np.sin(self.time))
        self.clear()
        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT | pyglet.gl.GL_DEPTH_BUFFER_BIT)
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.set_dip(0)


if __name__ == '__main__':
    app = App(height=HEIGHT, width=WIDTH, config=get_config(samples=4), resizable=True, vsync=True)
    pyglet.app.run()