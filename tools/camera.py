
import weakref

import pyglet
from pyglet.math import *
import math


class CameraManager:
    """Store camera instances and saved views, switch between cameras, animate motion."""


class Camera2D:
    """
    2D camera functionality. Set centered to false for GUI cameras, or true for world space view.
    Instantiate, then draw desired batches in on_draw() method using "with" context manager.
    """

    def __init__(
            self,
            window: pyglet.window.Window,
            scroll_speed=1,
            zoom_sensitivity=0.005,
            min_zoom=0.01,
            max_zoom=100,
            centered=True,
            mouse_controls=True
    ):
        assert min_zoom <= max_zoom, "Camera's minimum zoom must be greater than maximum zoom."
        self._window = weakref.proxy(window)
        self.scroll_speed = scroll_speed
        self.zoom_sensitivity = zoom_sensitivity
        self.max_zoom = max_zoom
        self.min_zoom = min_zoom
        self.centered = centered
        self.mouse_controls = mouse_controls
        self.x, self.y = 0, 0
        self.offset_x, self.offset_y = 0, 0
        self._zoom = min(1.0, self.max_zoom)

        if self.mouse_controls:
            self._window.push_handlers(self)

    def on_resize(self, width, height):
        pyglet.gl.glViewport(0, 0, *self._window.get_framebuffer_size())
        self._window.projection = Mat4.orthogonal_projection(
            0, self._window.width, 0, self._window.height, -255, 255
        )

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.MIDDLE:
            self.move(-dx, -dy)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom += scroll_y * self.zoom_sensitivity * self._zoom * 10

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = max(min(value, self.max_zoom), self.min_zoom)

    @property
    def position(self) -> tuple:
        return self.x, self.y

    @position.setter
    def position(self, value: tuple[float, float]):
        self.x, self.y = value

    def move(self, dx, dy):
        self.x += self.scroll_speed * dx / self._zoom
        self.y += self.scroll_speed * dy / self._zoom

    def reset(self):
        self.position = 0, 0
        self._zoom = min(1.0, self.max_zoom)

    def get_mouse(self) -> tuple:
        """Returns transformed mouse coordinates from screen to world space."""
        mx, my = self._window._mouse_x, self._window._mouse_y
        if self.centered:
            world_x = (mx - self._window.width / 2) / self._zoom + self.x
            world_y = (my - self._window.height / 2) / self._zoom + self.y
        else:
            world_x = mx / self._zoom + self.x
            world_y = my / self._zoom + self.y
        return world_x, world_y

    def _begin(self):
        """Setting camera state to draw the scene."""
        # Translating the view matrix
        self.offset_x, self.offset_y = self._get_offset()
        self._window.view = self._window.view.translate((-self.offset_x * self._zoom, -self.offset_y * self._zoom, 0))
        # Scaling the view matrix
        self._window.view = self._window.view.scale((self._zoom, self._zoom, 1))

    def _end(self):
        """Resetting changes to view matrix (in reverse order) to prevent chain multiplication."""
        # reverse scaling
        self._window.view = self._window.view.scale((1 / self._zoom, 1 / self._zoom, 1))
        # reverse translation
        self._window.view = self._window.view.translate((self.offset_x * self._zoom, self.offset_y * self.zoom, 0))

    def _get_offset(self) -> tuple:
        """Offset calculation for centered camera."""
        if self.centered:
            return (self.x - self._window.width / 2 / self._zoom,
                    self.y - self._window.height / 2 / self._zoom)
        else:
            return self.position

    def __enter__(self):
        self._begin()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end()


class Camera3D:
    """3D perspective camera functionality with builtin FPS style controls. Use CTRL+C to toggle controls."""

    def __init__(
            self,
            window: pyglet.window.Window,
            position=Vec3(0, 0, 500),
            target=Vec3(0, 0, -1),
            aspect_ratio=None,
            z_near=0.1,
            z_far=3000,
            fov=60,
            sensitivity=0.001,
            speed=10,
            fps_controls=True,
    ):
        self._window = weakref.proxy(window)
        # camera orientation
        self._position = position
        self._target = target
        # camera view parameters
        self.aspect_ratio = aspect_ratio or window.aspect_ratio
        self.z_near = z_near
        self.z_far = z_far
        self.fov = fov
        self.sensitivity = sensitivity
        self.speed = speed
        self._fps_controls = not fps_controls
        self.toggle_fps_controls()

        # normalized direction vectors and Oiler angles
        self._up = Vec3(0, 1, 0)
        self._front = (self._target - self.position).normalize()
        self._right = self._front.cross(self._up)
        self._yaw = 0
        self._pitch = 0
        self._roll = 0
        self._pitch_clamp = math.radians(89)

        # set Oiler angles to match passed orientation
        self._update_angles()

        # input controls
        self.keys = pyglet.window.key.KeyStateHandler()
        self._window.push_handlers(self)
        self._window.push_handlers(self.keys)

        # saved view
        self._saved_view = {}

        self.on_resize(window.width, window.height)
        self._update_view()

    def save_view(self):
        self._saved_view = {
            "front": self._front, "target": self._target, "position": self._position,
            "yaw": self._yaw, "pitch": self._pitch, "roll": self._roll
        }

    def load_view(self):
        if self._saved_view:
            self._front, self._target, self._position, self._yaw, self._pitch, self._roll = [value for value in self._saved_view.values()]
            self._update_view()

    def toggle_fps_controls(self):
        self._fps_controls = not self._fps_controls
        self._window.set_exclusive_mouse(self._fps_controls)

    def _update_view(self):
        """For FPS view always translate target in front of position vector."""
        self._target = self._position + self._front
        self._window.view = Mat4.look_at(self._position, self._target, self._up)

    def _update_angles(self):
        """If target or position are set, update angles to match this orientation."""
        self._yaw = -math.acos(self._front.x) * self._position.z / abs(self.position.z)
        self._pitch = math.asin(self._front.y)

    def _update_front(self):
        """Update front unit vector. Determined by yaw and pitch Oiler angles."""
        self._front = Vec3(
            math.cos(self._yaw) * math.cos(self._pitch),
            math.sin(self._pitch),
            math.sin(self._yaw) * math.cos(self._pitch)
        ).normalize()

    def _update_up(self):
        """Update up unit vector. Determined by roll Oiler angle."""
        self._up = Vec3(math.sin(self._roll), math.cos(self._roll), 0).normalize()

    def _update_right(self):
        """Update right unit vector. Determined by front and up unit vectors."""
        self._right = self._front.cross(self._up).normalize()

    def look_at(self, position: Vec3 = None, target: Vec3 = None):
        if position:
            self._position = position
        if target:
            self._target = target

        self._front = (self._target - self._position).normalize()
        self._update_angles()
        self._window.view = Mat4.look_at(self._position, self._target, self._up)

    def on_resize(self, width, height):
        """To set up custom projection matrix, default on_resize() must be overridden."""
        pyglet.gl.glViewport(0, 0, *self._window.get_framebuffer_size())
        self._window.projection = pyglet.math.Mat4.perspective_projection(
            self.aspect_ratio, 0.1, 3000, self.fov
        )
        return pyglet.event.EVENT_HANDLED

    def on_mouse_motion(self, x, y, dx, dy):
        if self._fps_controls:
            self._yaw += dx * self.sensitivity
            self._pitch += dy * self.sensitivity
            # Clamp the pitch angle to prevent flipping the camera
            self._pitch = max(-self._pitch_clamp, min(self._pitch_clamp, self._pitch))

            self._update_front()
            self._update_right()
            self._update_view()

    def on_key_press(self, symbol, modifiers):
        """Toggle camera's FPS style controls with CTRL+C"""
        if symbol == pyglet.window.key.C and modifiers & pyglet.window.key.MOD_CTRL:
            self.toggle_fps_controls()

    def _check_keys(self):
        if self.keys[pyglet.window.key.W]:
            self._position += self._front * self.speed
        if self.keys[pyglet.window.key.S]:
            self._position -= self._front * self.speed
        if self.keys[pyglet.window.key.A]:
            self._position -= self._right * self.speed
        if self.keys[pyglet.window.key.D]:
            self._position += self._right * self.speed
        if self.keys[pyglet.window.key.Q]:
            self._position += self._up * self.speed
        if self.keys[pyglet.window.key.E]:
            self._position -= self._up * self.speed

        # experimental roll features
        # if self.keys[pyglet.window.key.Z]:
        #     self._roll += self.speed * self.sensitivity
        #     self._update_up()
        #     self._update_right()
        # if self.keys[pyglet.window.key.C]:
        #     self._roll -= self.speed * self.sensitivity
        #     self._update_up()
        #     self._update_right()

        self._update_view()

    def on_draw(self):
        if self._fps_controls:
            self._check_keys()

    @property
    def target(self) -> Vec3:
        return self._target

    @target.setter
    def target(self, target: Vec3):
        self.look_at(target=target)

    @property
    def position(self) -> Vec3:
        return self._position

    @position.setter
    def position(self, position: Vec3):
        self.look_at(position=position)
