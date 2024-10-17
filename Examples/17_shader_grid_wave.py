import tools.camera
from tools.definitions import *
from tools.graphics import *
from pyglet.math import Vec3

SETTINGS = {
    "default_mode": False,
    "fps": 100,
    "width": 1280,
    "height": 720,
    "resizable": True,
    "config": get_config(samples=2)
}

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
    vertex_colors = colors;
    final_position = vec4(modified_position, 1.0);
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
        self.run = True
        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.fps = SETTINGS['fps']

        self.create_scene()

    def update_dip_position(self):
        self.circle.position = self.dip_position.xy
        self.program['dip_position'] = self.dip_position

    def create_scene(self):
        self.time = 0.0
        self.dip_speed = 10
        grid_vertices = get_quantized_grid_gl_line_vertices(span=5000, pitch=np.pi / 2)
        self.grid = self.program.vertex_list(len(grid_vertices) // 3, GL_LINES, self.batch,
                                             position=('f', grid_vertices),
                                             colors=('Bn', (92, 92, 92, 255) * (len(grid_vertices) // 3)))
        self.dip_position = Vec3(0, 500, 0)
        self.circle = pyglet.shapes.Circle(self.dip_position.x, self.dip_position.y, 100, batch=self.batch)

    def on_draw(self) -> None:
        self.check_keys()

        if self.run:
            self.time += 1 / self.fps
            self.dip_position.x += self.dip_speed * np.sin(self.time)

        self.update_dip_position()

        self.clear()
        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT | pyglet.gl.GL_DEPTH_BUFFER_BIT)
        self.batch.draw()

    def check_keys(self):
        if self.keys[pyglet.window.key.RIGHT]:
            self.dip_position.x += self.dip_speed
        if self.keys[pyglet.window.key.LEFT]:
            self.dip_position.x -= self.dip_speed
        if self.keys[pyglet.window.key.UP]:
            self.dip_position.z -= self.dip_speed
        if self.keys[pyglet.window.key.DOWN]:
            self.dip_position.z += self.dip_speed

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run


if __name__ == '__main__':
    start_app(App, SETTINGS)
