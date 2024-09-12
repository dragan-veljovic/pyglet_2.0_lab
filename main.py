import math

from pyglet.graphics.shader import Shader, ShaderProgram
from tools.camera import Camera2D, Camera3D
import pyglet.event
from pyglet.math import *

import tools_old.camera
from tools.definitions import *

# animations
from tools.animation import AnimationManagerV3
import tools.easing

from pyglet.gl import *

# start app in default display mode
START_DEFAULT = False
# or use these settings
WIDTH = 1280
HEIGHT = 720
FULLSCREEN = False
RESIZEABLE = True
FPS = 60


def get_standard_shader_program() -> ShaderProgram:
    vertex_shader = pyglet.resource.shader("shaders/vertex_shader.vert")
    fragment_shader = pyglet.resource.shader("shaders/fragment_shader.frag")
    shader_program = pyglet.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)
    return shader_program


def get_shader_program(*shader_files: str, path="shaders/") -> ShaderProgram:
    shaders = [pyglet.resource.shader(path + file) for file in shader_files]
    return ShaderProgram(*shaders)


def get_test_rectangle(x, y, width, height, program: pyglet.graphics.shader.ShaderProgram, batch=None, group=None):
    # BL, BR, TR, TL are vertices 0, 1, 2, 3 respectively
    x0, y0, x1, y1 = x - width / 2, y - height / 2, x + width / 2, y - height / 2
    x2, y2, x3, y3 = x1, y + height / 2, x0, y + height / 2
    return program.vertex_list_indexed(4, pyglet.gl.GL_TRIANGLES,
                                       [0, 1, 2, 0, 2, 3], batch=batch, group=group,
                                       position=('f', (x0, y0, 0, x1, y1, 0, x2, y2, 0, x3, y3, 0)),
                                       colors=('f', (1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1))
                                       )


def get_test_circle():
    pass

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


def get_box(program: pyglet.graphics.shader.ShaderProgram, batch=None, group=None, x=0, y=0, z=0, side_length=200):
    # Vertices for the edges of the box
    cx, cy, cz = x, y, z
    half_side = side_length / 2
    vertices = [
        # Bottom face
        cx - half_side, cy - half_side, cz - half_side, cx + half_side, cy - half_side, cz - half_side,
        cx + half_side, cy - half_side, cz - half_side, cx + half_side, cy + half_side, cz - half_side,
        cx + half_side, cy + half_side, cz - half_side, cx - half_side, cy + half_side, cz - half_side,
        cx - half_side, cy + half_side, cz - half_side, cx - half_side, cy - half_side, cz - half_side,

        # Top face
        cx - half_side, cy - half_side, cz + half_side, cx + half_side, cy - half_side, cz + half_side,
        cx + half_side, cy - half_side, cz + half_side, cx + half_side, cy + half_side, cz + half_side,
        cx + half_side, cy + half_side, cz + half_side, cx - half_side, cy + half_side, cz + half_side,
        cx - half_side, cy + half_side, cz + half_side, cx - half_side, cy - half_side, cz + half_side,

        # Vertical edges
        cx - half_side, cy - half_side, cz - half_side, cx - half_side, cy - half_side, cz + half_side,
        cx + half_side, cy - half_side, cz - half_side, cx + half_side, cy - half_side, cz + half_side,
        cx + half_side, cy + half_side, cz - half_side, cx + half_side, cy + half_side, cz + half_side,
        cx - half_side, cy + half_side, cz - half_side, cx - half_side, cy + half_side, cz + half_side,

    ]
    colors = [1, 1, 1, 1] * (len(vertices) // 3)

    return program.vertex_list(len(vertices) // 3, pyglet.gl.GL_LINES, batch, group,
                               position=('f', vertices),
                               colors=('f', colors))


def get_sine(
        program: pyglet.graphics.shader.ShaderProgram,
        position=Vec3(0, 0, 0),
        length=600,
        amplitude=300,
        color=(255, 255, 255, 255),
        no_points=200,
        wavelength=400,
        phase=0,
        batch=None, group=None
):
    x0, y0, z0 = position.x, position.y, position.z
    x_step = length / no_points
    previous_point = []
    points = []
    for n in range(no_points):
        dx = n * x_step
        theta = 2 * math.pi * dx/wavelength
        dy = amplitude * math.sin(theta + phase)
        dz = 0

        if previous_point:
            points.extend(previous_point)
        new_point = (x0 + dx, y0 + dy, z0 + dz)
        points.extend(new_point)

        if n == 0: continue
        else: previous_point = new_point

    colors = [component/255 for component in color]*(len(points)//3)

    return program.vertex_list(len(points)//3, pyglet.gl.GL_LINES, batch=batch, group=group,
                               position=('f', points),
                               colors=('f', colors),
                               )


def animate_sine(program, t, position, batch):
    phase = t*0.1
    return get_sine(program, phase=phase, position=position, batch=batch)


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


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color(20, 30, 40)
        self.conditional_program = get_shader_program('conditional.vert', 'per_pixel.frag')
        self.default_program = get_default_shader_program()
        self.program = self.conditional_program
        self.batch = pyglet.graphics.Batch()
        self.gui_batch = pyglet.graphics.Batch()
        self.grid_batch = pyglet.graphics.Batch()

        self.camera2D = Camera2D(self, centered=False)
        self.camera3D = None
        self.camera = self.camera2D

        # self.gui_camera = Camera2D(self, centered=False)
        self.saved_view_2D = self.view

        # key handler
        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)

        # timers
        self.dt = 1 / FPS
        self.time = 0.0
        self.run = True
        self.timer = pyglet.text.Label("0.00", font_size=26, batch=self.gui_batch, x=30, y=self.height - 30)

        # scene elements
        n = 30
        # self.back_rect = get_test_rectangle(self.width // 2, self.height // 2, self.width, self.height,
        #                                     self.default_program, batch=self.gui_batch)
        self.rectangles = [
            get_test_rectangle(self.width / 2 + i * 51, self.height / 2 + j * 50, 30, 30, self.program, self.batch) for
            i in range(n) for j in range(n)]
        self.circle = pyglet.shapes.Circle(0, 0, 100, batch=self.batch)
        self.grid = get_grid(self, self.default_program, self.grid_batch)

        self.sine = get_sine(self.program, Vec3(0, 0, 0), batch=self.batch)

        # animations
        self.manager = AnimationManagerV3()

        # labels
        self.back_rect = pyglet.shapes.Rectangle(20, 20, 200, 100, (0, 0, 0, 128), batch=self.gui_batch)
        self.zoom_label = pyglet.text.Label('zoom: ' + str(self.camera.zoom), font_size=14, batch=self.gui_batch, x=30,
                                            y=80)
        self.mouse_label = pyglet.text.Label('mouse: ' + str(self._mouse_x) + ", " + str(self._mouse_y), font_size=14,
                                             batch=self.gui_batch, x=30, y=60)
        self.pan_label = pyglet.text.Label('pan: ' + str(self.camera.position), font_size=14, batch=self.gui_batch,
                                           x=30, y=40)
        self.pointer = pyglet.shapes.Circle(*self.camera.get_mouse(), radius=10, segments=10, color=[255, 0, 0, 128],
                                            batch=self.batch)

        # 3d object
        self.box = get_box(self.program, self.batch, side_length=300, x=500, y=500, z=0)

    def on_key_press(self, symbol, modifiers):
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run
        elif symbol == pyglet.window.key.R:
            self.switch_camera()

    def switch_camera(self):
        """Switching different perspective cameras, should be implemented as a class """
        if isinstance(self.camera, Camera2D):
            # switching from 2D to 3D
            self.remove_handlers(self.camera)
            if not self.camera3D:
                self.camera3D = Camera3D(self, z_far=100_000, speed=20)
                self.remove_handlers(self.camera3D)
            self.camera = self.camera3D
            #self.set_exclusive_mouse(True)
            self.push_handlers(self.camera)
            self.camera.load_view()
        else:
            # switching from 3D to 2D
            self.remove_handlers(self.camera)
            self.camera.save_view()
            #self.set_exclusive_mouse(False)
            self.camera = Camera2D(self)
            self.push_handlers(self.camera)
            self.view = self.saved_view_2D

        self.dispatch_event("on_resize", self.width, self.height)

    def on_draw(self):
        if self.run:
            self.time += self.dt
            self.timer.text = f"{self.time:0.2f}"
            if self.program == self.conditional_program:
                self.program['time'] = self.time

        self.manager.play()

        self.clear()

        # results in memory leak
        # self.sine = animate_sine(self.program, self.time, Vec3(0, 0, 0), self.batch)

        self.switch_camera()
        if isinstance(self.camera, Camera2D):
            with self.camera:
                # pyglet.gl.glLineWidth(1)
                # self.grid_batch.draw()
                # pyglet.gl.glLineWidth(3)
                # self.batch.draw()
                self.gui_batch.draw()

        self.switch_camera()
        pyglet.gl.glLineWidth(1)
        self.grid_batch.draw()
        pyglet.gl.glLineWidth(3)
        self.batch.draw()


if __name__ == "__main__":
    if START_DEFAULT:
        start_default_display_mode(App)
    else:
        App(width=WIDTH, height=HEIGHT, resizable=RESIZEABLE, fullscreen=FULLSCREEN)
        pyglet.app.run(1/FPS)
