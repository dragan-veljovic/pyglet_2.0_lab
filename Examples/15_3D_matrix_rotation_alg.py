import pyglet.gl
import pyglet.window

import tools.camera
import tools.color
from tools.definitions import *
import numpy as np
import tools_old.graphics as gr

WIDTH = 1280
HEIGHT = 720
FPS = 60

def get_lines(points: list,
              colors: list,
              program: pyglet.graphics.shader.ShaderProgram,
              batch=None, group=None
              ):
    npoints = len(points) // 3
    return program.vertex_list(npoints, pyglet.gl.GL_LINES, batch=batch, group=group,
                               position=('f', points),
                               colors=('f', colors)
                               )

def get_grid(
        window: pyglet.window.Window, program,
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
        points.extend([i * main_div, -hlines * main_div, 0, i * main_div, hlines * main_div, 0])
        colors.extend(main_color * 2)
        if i < vlines:
            for k in range(sub_div, main_div, sub_div):
                points.extend(
                    [i * main_div + k, -hlines * main_div, 0, i * main_div + k, hlines * main_div, 0])
                colors.extend(sub_color * 2)

    # horizontal lines
    for j in range(-hlines, hlines + 1):
        points.extend([-vlines * main_div, j * main_div, 0, vlines * main_div, j * main_div, 0])
        colors.extend(main_color * 2)
        if j < hlines:
            for l in range(sub_div, main_div, sub_div):
                points.extend(
                    [-vlines * main_div, j * main_div + l, 0, vlines * main_div, j * main_div + l, 0])
                colors.extend(sub_color * 2)

    return get_lines(points, colors, program, batch, group)


def rotate_points(
        points: np.array,
        pitch: float = 0,
        yaw: float = 0,
        roll: float = 0,
        anchor: tuple[float, float, float] = (0, 0, 0)
) -> np.ndarray:
    """
    3D rotation around passed anchor.
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

# Parameters


# Generate vertices
def get_sine_vertices() -> np.array:
    amplitude = 200
    wavelength = 200
    num_vertices = 100
    cycle_length = 4*wavelength  # One full cycle of the sine wave
    x_values = np.linspace(0, cycle_length, num_vertices)

    vertices = []
    for x in x_values:
        y = amplitude * np.sin(2 * np.pi * x / wavelength)
        vertices.append((x, y, 0))  # z is 0, since the sine wave is in the x-y plane
    return np.array(vertices)


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.camera = tools.camera.Camera3D(self, z_near=1, z_far=3000)
        self.gui_batch = pyglet.graphics.Batch()
        self.batch = pyglet.graphics.Batch()
        self.scene_elements = []
        self.program = pyglet.graphics.shader.ShaderProgram(
            pyglet.resource.shader("shaders/default.vert"),
            pyglet.resource.shader("shaders/default.frag"),
        )

        self.scene()

        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)

    def scene(self):
        self.grid = get_grid(self, self.program, self.gui_batch)
        self.scene_elements.append(pyglet.shapes.Circle(0, 0, 100, batch=self.batch))
        self.scene_elements.append(
            pyglet.shapes.Rectangle(-500, 0, -250, 250, color=tools.color.ORANGE_PEEL, batch=self.batch))
        # self.points = np.array(((-100, -100, 0), (0, 0, 0), (500, -300, 0), (600, 0, 0)))
        # vertices = gr.get_gl_lines_vertices_numpy(self.points)
        # self.line = self.program.vertex_list(len(vertices)//3, pyglet.gl.GL_LINES, self.batch,
        #                                      position=('f', vertices),
        #                                      colors=('Bn', (*tools.color.BLUE_SAPPHIRE, 255) * (len(vertices) // 3)))

        # generating number of sine functions rotated by equal amount
        self.number = 50
        self.sines = []
        self.sine_points = get_sine_vertices()

        for n in range(self.number):
            theta = 2*np.pi/self.number
            sine_vertices = gr.get_gl_lines_vertices_numpy(rotate_points(self.sine_points, pitch=n*theta))
            self.sines.append(self.program.vertex_list(
                len(sine_vertices)//3, pyglet.gl.GL_LINES, self.batch,
                position=('f', sine_vertices),
                colors=('Bn', (255-n*20, *tools.color.ORANGE_PEEL[1:3], 255) * (len(sine_vertices)//3))
            )
        )
        print(len(self.sines))

        self.roll = 0.0
        self.yaw = 0.0
        self.pitch = 0.0


    def on_draw(self):
        self.pitch += 0.02
        self.apply_rotation()

        self.clear()
        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT | pyglet.gl.GL_DEPTH_BUFFER_BIT)

        pyglet.gl.glLineWidth(3)
        self.batch.draw()
        pyglet.gl.glLineWidth(1)
        self.gui_batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.R:
            self.points = rotate_points(self.points, roll=0.1, yaw=0.1, anchor=(0, 0, 0))
            vertices = gr.get_gl_lines_vertices_numpy(self.points)
            self.line.position[:] = vertices

    def apply_rotation(self):
        for n in range(self.number):
            self.sines[n].position[:] = gr.get_gl_lines_vertices_numpy(rotate_points(self.sine_points, roll=self.roll, yaw=self.yaw, pitch=self.pitch * n * 0.01, anchor=(0, 0, 0)))

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        print("mouse dragged")
        if buttons == pyglet.window.mouse.RIGHT:
            print("hello")

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        print("mouse pressed")


if __name__ == '__main__':
    my_app = App(width=WIDTH, height=HEIGHT, config=get_config(), resizable=True, vsync=True)
    pyglet.app.run(1 / FPS)
