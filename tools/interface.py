from tools.camera import Camera3D
from pyglet.gl import *
from pyglet.math import Vec3, Vec4, Mat4
from pyglet.window import Window
from pyglet.graphics import Batch, Group
from pyglet.graphics.shader import Shader, ShaderProgram
from tools.definitions import get_default_shaders_program


class Ray:
    def __init__(
            self, direction: Vec3, origin=Vec3(0, 0, 0), length=100.0,
            program: ShaderProgram = None, batch: Batch = None, group: Group = None, color=(1.0, 1.0, 1.0, 1.0)
    ):
        """A vector from origin in a given direction, and of specified length."""
        self._direction = direction
        self._origin = origin
        self._length = length

        self.vertex_list = None
        self._program = program
        self._batch = batch
        self._group = group

        self._target = self._origin + self._direction * self._length

        self.colors = color * 2

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
    """
    Simple Axis-aligned bounding box (aabb).
    Normally you would use :py:meth:`~model.Mesh.get_bounding_box()` to create BoundingBox instance.
    """
    def __init__(self, min_corner: Vec3, max_corner: Vec3):
        self.min = min_corner
        self.max = max_corner

        self.vertex_list = None

    def center(self) -> Vec3:
        return (self.min + self.max) * 0.5

    def size(self) -> Vec3:
        return self.max - self.min

    def __contains__(self, point: Vec3):
        return all(self.min[i] <= point[i] <= self.max[i] for i in range(3))

    def draw(self, program=None, batch=None, group=None, color=(1.0, 1.0, 1.0, 1.0)):
        """Visual representation of a bounding box - a wireframe cuboid."""
        vertices = (
            # front face
            (self.min.x, self.min.y, self.min.z),
            (self.max.x, self.min.y, self.min.z),
            (self.max.x, self.max.y, self.min.z),
            (self.min.x, self.max.y, self.min.z),

            # back face
            (self.min.x, self.min.y, self.max.z),
            (self.max.x, self.min.y, self.max.z),
            (self.max.x, self.max.y, self.max.z),
            (self.min.x, self.max.y, self.max.z)
        )

        indices = [
            0, 1, 1, 2, 2, 3, 3, 0,  # construct front face
            4, 5, 5, 6, 6, 7, 7, 4,  # construct back face
            0, 4, 1, 5, 2, 6, 3, 7  # connect two faces
        ]

        positions = [coord for idx in indices for coord in vertices[idx]]
        count = len(positions)//3
        if not program:
            program = get_default_shaders_program()
        if not batch:
            batch = Batch()
        if self.vertex_list:
            self.vertex_list.position[:] = positions
        else:
            self.vertex_list = program.vertex_list(
                count, GL_LINES, batch, group,
                position=('f', positions),
                color=('f', color * count)
            )

    def delete(self):
        self.vertex_list.delete()


class MousePicker:
    def __init__(self, window: Window, camera: Camera3D, program=None, batch=None, group=None):
        """
        Helper class with method to create a :py:class:`Ray` object from mouse (x,y) position,
        directed into the screen.
        """
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


def ray_intersects_aabb(ray_origin: Vec3, ray_dir: Vec3, bbox: BoundingBox) -> tuple[bool, float]:
    tmin = -float('inf')
    tmax = float('inf')

    for i in range(3):
        origin = ray_origin[i]
        direction = ray_dir[i]
        min_bound = bbox.min[i]
        max_bound = bbox.max[i]

        if abs(direction) < 1e-8:
            # Ray is parallel to slab. If origin not within slab, no hit.
            if origin < min_bound or origin > max_bound:
                return False, 0.0
        else:
            inv_d = 1.0 / direction
            t1 = (min_bound - origin) * inv_d
            t2 = (max_bound - origin) * inv_d

            if t1 > t2:
                t1, t2 = t2, t1  # swap

            tmin = max(tmin, t1)
            tmax = min(tmax, t2)

            if tmin > tmax:
                return False, 0.0

    # Optional: you can reject if tmax < 0 (box is behind ray)
    if tmax < 0:
        return False, 0.0

    return True, tmin if tmin > 0 else tmax






