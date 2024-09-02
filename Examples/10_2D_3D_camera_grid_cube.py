import math

from pyglet.graphics.shader import Shader, ShaderProgram
from tools.camera import Camera2D
import pyglet.event
from pyglet.math import *

import tools_old.camera
from tools_old.definitions import *

# animations
from tools_old.animation import AnimationManagerV3
import tools_old.easing

from pyglet.gl import *

WIDTH = 1280
HEIGHT = 720
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


# def rotate_around_point(view: Mat4, angle: float, vector: Vec3, point: Vec3) -> Mat4:
#     """Get a rotation Matrix around a specific point."""
#     translation_to_origin = view.translate(point)
#     rotation = view.rotate(angle, vector)
#     translation_back = view.translate(point)
#
#     # Combine the translations and rotation
#     return translation_back @ rotation @ translation_to_origin


class FPSCamera:
    def __init__(self, window, position):
        self.window = window
        self.position = Vec3(*position)
        self.yaw = 0.0
        self.pitch = 0.0
        self.speed = 10
        self.sensitivity = 0.05
        self.view = Mat4.look_at(self.position, self.position + Vec3(0, 0, -1), Vec3(0, 1, 0))
        self.update_view()
        self.keys = window.keys

    def on_mouse_motion(self, x, y, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch += dy * self.sensitivity

        # Clamp the pitch angle to prevent flipping the camera
        self.pitch = max(-89.0, min(89.0, self.pitch))

        self.update_view()

    def check_keys(self):
        # Calculate the front vector
        front = Vec3(
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ).normalize()

        # Calculate the right vector
        right = Vec3(
            math.sin(math.radians(self.yaw)),
            0,
            -math.cos(math.radians(self.yaw))
        ).normalize()

        # Calculate the up vector
        up = right.cross(front).normalize()

        speed = self.speed

        if self.keys[pyglet.window.key.W]:
            self.position += front * speed
        if self.keys[pyglet.window.key.S]:
            self.position -= front * speed
        if self.keys[pyglet.window.key.A]:
            self.position += right * speed
        if self.keys[pyglet.window.key.D]:
            self.position -= right * speed
        if self.keys[pyglet.window.key.Q]:
            self.position -= up * speed
        if self.keys[pyglet.window.key.E]:
            self.position += up * speed

        self.update_view()

    def update_view(self):
        # Calculate the new target position using spherical coordinates
        front = Vec3(
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        )
        front = front.normalize()

        target = self.position + front
        self.window.view = Mat4.look_at(self.position, target, Vec3(0, 1, 0))



class App(pyglet.window.Window):
    def __init__(self):
        super(App, self).__init__(WIDTH, HEIGHT, resizable=True)
        center_window(self)
        set_background_color(20, 30, 40)
        self.program = get_shader_program('conditional.vert', 'per_pixel.frag')
        self.default_program = get_shader_program('default.vert', 'default.frag')
        self.batch = pyglet.graphics.Batch()
        self.gui_batch = pyglet.graphics.Batch()
        self.grid_batch = pyglet.graphics.Batch()
        self.camera = Camera2D(self)
        self.gui_camera = Camera2D(self, centered=False)
        self.exclusive_mouse = True
        self.set_exclusive_mouse(self.exclusive_mouse)

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
        self.target = Vec3(0, 0, -500)
        self.position = Vec3(0, 0, 500)

        self.FPS_camera = FPSCamera(self, self.position)

    def on_key_press(self, symbol, modifiers):
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run
        elif symbol == pyglet.window.key.R:
            self.camera.reset()

        elif symbol == pyglet.window.key.V:
            self.view = pyglet.math.Mat4.look_at(Vec3(0, 0, 500), Vec3(500, 500, 0), Vec3(0, 1, 0))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:
            self.camera.mouse_pan(dx, dy)
            pan_x, pan_y = self.camera.position
            self.pan_label.text = 'pan: ' + str(round(pan_x, 0)) + ', ' + str(round(pan_y, 0))

        if buttons == pyglet.window.mouse.MIDDLE:
            self.camera.mouse_zoom(dy)
            self.zoom_label.text = "zoom: " + str(round(self.camera.zoom, 2))
        if buttons == pyglet.window.mouse.RIGHT:
            # self.view = self.view.rotate(0.01*dx, pyglet.math.Vec3(0, 0, 1))
            self.view = self.view.translate((0, 0, dy))
            pass

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.MIDDLE:
            # self.camera.camera_reset()
            self.manager.add(
                tools_old.animation.Move(self.camera, 0, 0, duration=0.5, func_handle=tools_old.easing.ease_in_out),
            )
            self.manager.add(tools_old.animation.Color(self.circle, (255, 80, 40), func_handle=tools_old.easing.ease_in_out))
            self.manager.add(tools_old.animation.Color(self.circle, (255, 255, 255), duration=0.3))

        if button == pyglet.window.mouse.RIGHT:
            self.exclusive_mouse = not self.exclusive_mouse
            self.set_exclusive_mouse(self.exclusive_mouse)

    def on_mouse_motion(self, x, y, dx, dy):
        sensitivity = 0.001
        mx, my = self.camera.get_mouse()
        self.mouse_label.text = 'mouse: ' + str(round(mx, 0)) + ', ' + str(round(my, 0))

        # rotation around origin
        # self.view = self.view.rotate(-sensitivity * -dx, (0, 1, 0))
        # self.view = self.view.rotate(-sensitivity * dy, (1, 0, 0))

        # trying FPS camera
        if self.exclusive_mouse:
            self.FPS_camera.on_mouse_motion(x, y, dx, dy)

        # updating trackers
        # self.target.x += dx
        # self.target.y += dy
        #
        # self.update_view()

    def on_resize(self, width, height):
        """To set up projection, default on_resize() must be overridden."""
        pyglet.gl.glViewport(0, 0, *self.get_framebuffer_size())
        self.projection = pyglet.math.Mat4.perspective_projection(self.aspect_ratio, 0.1, 3000)
        return pyglet.event.EVENT_HANDLED



    def on_draw(self):
        if self.run:
            self.time += self.dt
            self.timer.text = f"{self.time:0.2f}"
            self.program['time'] = self.time

        self.manager.play()

        self.FPS_camera.check_keys()

        self.pointer.position = self.camera.get_mouse()

        self.clear()
        with self.camera:
            pyglet.gl.glLineWidth(1)
            self.grid_batch.draw()

            pyglet.gl.glLineWidth(3 * self.camera.zoom)
            self.batch.draw()

        with self.gui_camera:
            self.gui_batch.draw()


def main():
    App()
    pyglet.app.run(1 / FPS)


if __name__ == "__main__":
    main()
