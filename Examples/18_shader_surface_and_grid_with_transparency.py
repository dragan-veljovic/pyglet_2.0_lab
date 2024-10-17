"""
Two meshes of GL_LINE and GL_TRIANGLE primitives that is transformed by shaders.
Group subclass allows for toggling visibility of each one.
"""

import pyglet.gl.gl_compat
from pyglet.gl import *
from tools.definitions import *
from tools.camera import Camera3D, Camera2D
from tools.graphics import *
import numpy as np

vert_source = """
#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;
out vec4 final_position;

uniform vec3 dip_position;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    // Calculate distance of vertex from dip position on x-axis
    // float distance_from_dip = position.x - dip_position.x;

    // Calculate the displacement of vertices from the dip position
    vec3 disp_from_dip = position - vec3(dip_position.x, 0, dip_position.z);

    // Calculate the magnitude of the displacement from the dip
    float disp_from_dip_mag = length(disp_from_dip);

    // Modifying position according to displacement from the dip
    vec3 modified_position = position;
    float dip_radius = 500;
    float dip_amplitude = 300;
    // modified_position.y += clamp(disp_from_dip_mag / dip_radius * disp_from_dip_mag, 0, dip_amplitude);
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
    final_position = vec4(modified_position, 1.0);
    vertex_colors = colors;
}
"""

frag_source = """
#version 330 core
in vec4 vertex_colors;
in vec4 final_position;
out vec4 final_colors;

void main() {
    final_colors = vec4(vertex_colors.rg, vertex_colors.b + final_position.y / 300 * 0.4, 1.0);
}
"""


def get_quad_grid_vertices(span, step, rotation=np.pi / 2):
    """Returns GL_TRIANGLE vertices for quad grid of given span and step."""
    assert span % 2 == 0, "Span must be an even integer"
    half_span = span // 2
    vertices = []
    for row in range(-half_span, half_span, step):
        for column in range(-half_span, half_span, step):
            quad_points = np.array((
                (row, column, 0),
                (row + step, column, 0),
                (row + step, column + step, 0),
                (row, column, 0),
                (row + step, column + step, 0),
                (row, column + step, 0),
            ), dtype=np.float64)
            rotated_quad_points = rotate_points(quad_points, rotation)

            vertices.extend(rotated_quad_points.flatten())
    return vertices


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


class BlendGroup(pyglet.graphics.Group):
    def __init__(self, program, order=0, parent=None):
        """
        GL_DEPTH_TEST seems to interfere with VertexList Alpha setting, not allowing for transparency.
        This Group avoids writing depth data if object is set as not visible, allowing Alpha settings to take effect.
        """
        super(BlendGroup, self).__init__(order, parent)
        self.program = program
        self.visible = True

    def set_state(self):
        if not self.visible:
            glDepthMask(GL_FALSE)  # Disable writing to the depth buffer
        self.program.use()

    def unset_state(self):
        if not self.visible:
            glDepthMask(GL_TRUE)  # Re-enable writing to the depth buffer
        self.program.stop()

    def __hash__(self):
        return hash((self.order, self.parent, self.program))

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.gui_batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self, z_far=20_000)
        self.camera2D = Camera2D(self, centered=False)
        self.program = ShaderProgram(
            Shader(vert_source, 'vertex'),
            Shader(frag_source, 'fragment')
        )

        self.grid_program = ShaderProgram(
            Shader(vert_source, 'vertex'),
            pyglet.resource.shader("shaders/default.frag")
        )

        self.create_scene()
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)

    def create_scene(self):
        from pyglet.math import Vec3
        self.time = 0.0
        self.dip_speed = 10
        self.label = pyglet.text.Label("Test label", 100, self.height - 100, batch=self.gui_batch)

        # line grid object
        self.grid_group = BlendGroup(self.grid_program)
        self.grid_vertices = get_quantized_grid_gl_line_vertices(span=5000, pitch=np.pi / 2)
        self.grid_vertices_count = len(self.grid_vertices) // 3
        self.grid_visible_colors = (92 / 255, 92 / 255, 92 / 255, 255 / 255) * self.grid_vertices_count
        self.grid_invisible_colors = (1.0, 0.0, 0.0, 0.0) * self.grid_vertices_count
        self.grid = self.grid_program.vertex_list(self.grid_vertices_count, GL_LINES, self.batch, group=self.grid_group,
                                                  position=('f', self.grid_vertices),
                                                  colors=('f', self.grid_visible_colors))

        # quad grid object
        self.quad_group = BlendGroup(self.program)
        self.quad_vertices = get_quad_grid_vertices(10_000, 100)
        self.quad_vertices_count = len(self.quad_vertices) // 3
        self.quad_visible_colors = (150 / 255, 150 / 255, 150 / 255, 255 / 255) * self.quad_vertices_count
        self.quad_invisible_colors = (0.0, 0.0, 0.0, 0.0) * self.quad_vertices_count
        self.quad_grid = self.program.vertex_list(
            count=self.quad_vertices_count,
            mode=pyglet.gl.GL_TRIANGLES,
            batch=self.batch,
            group=self.quad_group,
            position=('f', self.quad_vertices),
            colors=('f', self.quad_visible_colors),
        )

        self.dip_position = Vec3(0, 500, 0)
        self.circle = pyglet.shapes.Circle(self.dip_position.x, self.dip_position.y, 100, batch=self.batch)

    def update_dip_position(self):
        self.circle.position = self.dip_position.xy
        self.program['dip_position'] = self.dip_position
        self.grid_program['dip_position'] = self.dip_position

    def toggle_grid_visibility(self):
        self.grid_group.visible = not self.grid_group.visible
        if self.grid_group.visible:
            self.grid.colors[:] = self.grid_visible_colors
        else:
            self.grid.colors[:] = self.grid_invisible_colors

    def toggle_quad_visibility(self):
        self.quad_group.visible = not self.quad_group.visible
        if self.quad_group.visible:
            self.quad_grid.colors[:] = self.quad_visible_colors
        else:
            self.quad_grid.colors[:] = self.quad_invisible_colors

    def on_draw(self) -> None:
        self.time += 1 / 60
        self.dip_position.x += self.dip_speed * np.sin(self.time)

        self.update_dip_position()

        self.clear()

        # no effect
        with self.camera2D:
            self.gui_batch.draw()

        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT | pyglet.gl.GL_DEPTH_BUFFER_BIT)
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
        if symbol == pyglet.window.key.G:
            self.toggle_grid_visibility()
        if symbol == pyglet.window.key.F:
            self.toggle_quad_visibility()


if __name__ == '__main__':
    app = App(width=1280, height=720, resizable=True)
    pyglet.app.run(1 / 60)
