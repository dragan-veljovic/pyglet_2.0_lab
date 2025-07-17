from tools.camera import Camera3D
from pyglet.gl import *
from pyglet.math import Vec3, Vec4, Mat4
from pyglet.window import Window
from pyglet.graphics import Batch, Group
from pyglet.graphics.shader import Shader, ShaderProgram
from tools.definitions import get_default_shaders_program


class Ray:
    def __init__(self, direction: Vec3, origin=Vec3(0, 0, 0), length=100.0,
                 program: ShaderProgram = None, batch: Batch = None, group: Group = None):
        """A vector from origin in a given direction, and of specified length."""
        self._direction = direction
        self._origin = origin
        self._length = length

        self.vertex_list = None
        self._program = program
        self._batch = batch
        self._group = group

        self._target = self._origin + self._direction * self._length

        self.colors = (1.0, 1.0, 1.0, 1.0) * 2

    def draw(self):
        """Show visual representation of the ray."""
        positions = (self._origin.x, self._origin.y, self._origin.z, self._target.x, self._target.y, self._target.z)
        if not self._program:
            self._program = get_default_shaders_program()
        if not self._batch:
            self._batch = Batch()
        if self.vertex_list:
            self.vertex_list.position[:] = positions
        else:
            self._program.vertex_list(
                2, GL_LINES, self._batch, self._group,
                position=('f', positions),
                color=('f', self.colors)
            )

    def delete(self):
        self.vertex_list.delete()


class BoundingBox:
    def __init__(self, min_corner: Vec3, max_corner: Vec3):
        self.min = min_corner
        self.max = max_corner

    def center(self) -> Vec3:
        return (self.min + self.max) * 0.5

    def size(self) -> Vec3:
        return self.max - self.min

    def __contains__(self, point: Vec3):
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(3))

    # For testing purposes only
    def draw(self, program=None, batch=None, group=None):
        ray = Ray(self.size(), self.min, 1.0, program, batch, group)
        ray.draw()


class MousePicker:
    def __init__(self, window: Window, camera: Camera3D, program=None, batch=None, group=None):
        """Create a :py:class:`Ray` object from mouse (x,y) position, directed into the screen."""
        self.window = window
        self.view = window.view
        self.proj = window.projection
        self.camera = camera
        self.program = program
        self.batch = batch
        self.group = group

    def get_mouse_ray(self) -> Ray:
        x, y = self.window._mouse_x, self.window._mouse_y
        width, height = self.window.width, self.window.height

        # convert from viewport space to normalized device coordinates
        ndc_x = (2.0 * x) / width - 1.0
        ndc_y = (2.0 * y) / height - 1.0

        # convert from NDC to clip coordinates (z=-1 for vector pointing in, and adding w)
        clip_near = Vec4(ndc_x, ndc_y, -1.0, 1.0)
        clip_far = Vec4(ndc_x, ndc_y, 1.0, 1.0)

        # convert from clip to eye space by inverse proj, then to world space by inverse view matrix
        inv_vp = (self.window.projection @ self.window.view).__invert__()
        world_near = inv_vp @ clip_near
        world_far = inv_vp @ clip_far

        ray_origin = world_near.xyz / world_near.w
        ray_direction = world_far.xyz / world_far.w - ray_origin
        ray_direction.normalize()

        return Ray(ray_direction, self.camera.position, self.camera.z_far,
                   self.program, self.batch, self.group)
