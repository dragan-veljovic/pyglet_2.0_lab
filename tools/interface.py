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

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    @property
    def target(self):
        return self._target


class Selectable:
    """Base class for all selectable objects."""

    def __init__(self):
        self._selected = False
        self._bounding_box = self.get_bounding_box()

        self.update_bounding_box()

    @property
    def matrix(self) -> Mat4:
        """Return the current world transformation matrix."""
        raise NotImplementedError

    def get_bounding_box(self):
        """Return an AABB or other bounding shape in world space."""
        raise NotImplementedError

    def update_bounding_box(self):
        """Optional method to update cached bounding box."""
        # self._bounding_box = self.get_bounding_box()
        # minimum = self._bounding_box._min
        # maximum = self._bounding_box._max
        # self._bounding_box._min = self.matrix @ Vec4(minimum.x, minimum.y, minimum.z, 1)
        # self._bounding_box._max = self.matrix @ Vec4(maximum.x, maximum.y, maximum.z, 1)
        raise NotImplementedError

    def ray_intersects_aabb(self, ray: Ray) -> tuple[bool, float]:
        """Check if :py:class:`Ray` intersects :py:class:`BindingBox`"""

        # Inefficient! Create once then update with matrix!
        return self._bounding_box.intersects_ray(ray)

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        self._selected = value

    @property
    def bounding_box(self):
        return self._bounding_box


class BoundingBox:
    """
    Simple Axis-aligned bounding box (aabb).
    Normally you would use :py:meth:`~model.Mesh.get_bounding_box()` to create BoundingBox instance.
    """
    def __init__(
            self,
            min_corner: Vec3,
            max_corner: Vec3,
            program: ShaderProgram = None,
            batch: Batch = None,
            group: Group = None,
            color=(1.0, 1.0, 1.0, 1.0)
    ):
        self._min = min_corner
        self._max = max_corner

        self._program = program
        self._batch = batch
        self._group = group
        self._color = color

        self._vertex_list = None

    def center(self) -> Vec3:
        return (self._min + self._max) * 0.5

    def size(self) -> Vec3:
        return self._max - self._min

    def __contains__(self, point: Vec3):
        return all(self._min[i] <= point[i] <= self._max[i] for i in range(3))

    def intersects_ray(self, ray: Ray) -> tuple[bool, float]:
        tmin = -float('inf')
        tmax = float('inf')

        for i in range(3):
            origin = ray.origin[i]
            direction = ray.direction[i]
            min_bound = self._min[i]
            max_bound = self._max[i]

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

        # Optional reject if tmax < 0 (box is behind ray)
        if tmax < 0:
            return False, 0.0

        return True, tmin if tmin > 0 else tmax

    def _create_vertex_list(self):
        """Visual representation of a bounding box - a wireframe cuboid."""
        vertices = (
            # front face
            (self._min.x, self._min.y, self._min.z),
            (self._max.x, self._min.y, self._min.z),
            (self._max.x, self._max.y, self._min.z),
            (self._min.x, self._max.y, self._min.z),

            # back face
            (self._min.x, self._min.y, self._max.z),
            (self._max.x, self._min.y, self._max.z),
            (self._max.x, self._max.y, self._max.z),
            (self._min.x, self._max.y, self._max.z)
        )

        indices = [
            0, 1, 1, 2, 2, 3, 3, 0,  # construct front face
            4, 5, 5, 6, 6, 7, 7, 4,  # construct back face
            0, 4, 1, 5, 2, 6, 3, 7  # connect two faces
        ]

        positions = [coord for idx in indices for coord in vertices[idx]]
        count = len(positions) // 3
        if not self._program:
            self._program = get_default_shaders_program()
        if not self._batch:
            self._batch = Batch()

        return self._program.vertex_list(
            count, GL_LINES, self._batch, self._group,
            position=('f', positions),
            color=('f', self._color * count)
        )

    def delete(self):
        if self._vertex_list is not None:
            self._vertex_list.delete()
            self._vertex_list = None


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
        min_bound = bbox._min[i]
        max_bound = bbox._max[i]

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
