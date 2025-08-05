from tools.camera import Camera3D
from pyglet.gl import *
from pyglet.math import Vec3, Vec4, Mat4
from pyglet.window import Window
from pyglet.graphics import Batch, Group
from pyglet.graphics.shader import Shader, ShaderProgram
from tools.definitions import get_default_shader_program

_interface_shader_program = None


def get_interface_shader_program():
    """Return the singleton ShaderProgram for interface objects."""
    global _interface_shader_program
    if _interface_shader_program is None:
        # Only compile the shader program once
        default_vert_content = """
            #version 330 core
            in vec3 position;
            in vec4 color;
            
            out vec4 frag_color;

            uniform WindowBlock
                {
                    mat4 projection;
                    mat4 view;
                } window;

            uniform mat4 model_precalc;
            uniform bool rendering_dynamic_object = false;

            void main() {
                mat4 model = rendering_dynamic_object ? model_precalc : mat4(1.0);
                gl_Position = window.projection * window.view * model * vec4(position, 1.0);
                frag_color = color;
            }
        """

        default_frag_content = """
        #version 330 core
        in vec4 frag_color;
        
        out vec4 final_color;

        void main() {
            final_color = frag_color;
        }
        """

        _interface_shader_program = ShaderProgram(
            Shader(default_vert_content, 'vertex'),
            Shader(default_frag_content, 'fragment')
        )

    return _interface_shader_program


class Ray:
    def __init__(
            self, direction: Vec3, origin=Vec3(0, 0, 0), length=100.0,
            program: ShaderProgram = None, batch: Batch = None, group: Group = None, color=(1.0, 1.0, 1.0, 1.0)
    ):
        """
        Represents a ray from an origin in a given direction and of specified length.

        If ray is to be constructed from mouse inputs, as used for selection interface in 3D scene,
        use :py:class:`MousePicker` helper to construct it.

        Visual representation of the ray can be toggled on demand through
        :py:meth:`create_vertex_list` and :py:meth:`delete`.
        """
        self._direction = direction
        self._origin = origin
        self._length = length
        self._target = self._origin + self._direction * self._length

        self._vertex_list = None
        self._program = program
        self._batch = batch
        self._group = group
        self._color = color

    def create_vertex_list(self):
        """Creates visual representation of the ray."""
        positions = (self._origin.x, self._origin.y, self._origin.z, self._target.x, self._target.y, self._target.z)
        if not self._program:
            self._program = get_default_shader_program()
        if not self._batch:
            self._batch = Batch()
        else:
            self._vertex_list = self._program.vertex_list(
                2, GL_LINES, self._batch, self._group,
                position=('f', positions),
                color=('f', self._color * 2)
            )

    def delete(self):
        """Deletes the visual representation of the ray."""
        if self._vertex_list:
            self._vertex_list.delete()

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    @property
    def target(self):
        return self._target


class BoundingBox:
    """
    Simple Axis-aligned bounding box (aabb). Can be used for selection or collision mechanisms.

    Normally, :py:class:`Selectable` object would create and manage its own instance of a bounding box.
    Visualisation of a bounding box can be controlled on demand with :py:meth:`create_vertex_list`
    and :py:meth:`delete`, which is more efficient as usually few bounding boxes are visible in the scene.

    TODO: BoundingBox visuals currently only correctly match the actual shape in scale and position, not rotation!
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
        self._min_original = min_corner
        self._max_original = max_corner
        self._min = min_corner
        self._max = max_corner
        self._program = program
        self._batch = batch
        self._group = group
        self._color = color

        # buffer data for faster updates
        self.corners = [
            # front face
            Vec3(self._min_original.x, self._min_original.y, self._min_original.z),
            Vec3(self._max_original.x, self._min_original.y, self._min_original.z),
            Vec3(self._max_original.x, self._max_original.y, self._min_original.z),
            Vec3(self._min_original.x, self._max_original.y, self._min_original.z),

            # back face
            Vec3(self._min_original.x, self._min_original.y, self._max_original.z),
            Vec3(self._max_original.x, self._min_original.y, self._max_original.z),
            Vec3(self._max_original.x, self._max_original.y, self._max_original.z),
            Vec3(self._min_original.x, self._max_original.y, self._max_original.z),
        ]

        indices = [
            0, 1, 1, 2, 2, 3, 3, 0,  # construct front face
            4, 5, 5, 6, 6, 7, 7, 4,  # construct back face
            0, 4, 1, 5, 2, 6, 3, 7  # connect two faces
        ]

        self._positions = [coord for idx in indices for coord in self.corners[idx]]
        self._count = len(self._positions) // 3

        # visual representation controlled externally on demand
        self._vertex_list = None

    def center(self) -> Vec3:
        return (self._min + self._max) * 0.5

    def size(self) -> Vec3:
        return self._max - self._min

    def __contains__(self, point: Vec3):
        return all(self._min[i] <= point[i] <= self._max[i] for i in range(3))

    def intersects_ray(self, ray: Ray) -> tuple[bool, float]:
        """
        Check if passed :py:class:`Ray` intersects this bounding box instance.
        Returns bool statement of the intersection, as well as length of the intersection.
        """
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

    def update(self, matrix: Mat4):
        """
        Transform original box vertices with model matrix.
        If :py:class:`Selectable` object is moved, its BoundingBox should be updated before checking intersections.
        TODO: losing fit with rotation
        """
        # Transform each corner to world space
        transformed = [matrix @ Vec4(corner.x, corner.y, corner.z, 1) for corner in self.corners]
        # Extract x, y, z from transformed Vec4s
        xs = []
        ys = []
        zs = []
        for vector in transformed:
            xs.append(vector.x)
            ys.append(vector.y)
            zs.append(vector.z)

        # Recompute new AABB in world space
        self._min = Vec3(min(xs), min(ys), min(zs))
        self._max = Vec3(max(xs), max(ys), max(zs))

    def create_vertex_list(self):
        """Visual representation of a bounding box - a wireframe cuboid."""
        if self._vertex_list:
            self.delete()

        self._vertex_list = self._program.vertex_list(
            self._count, GL_LINES, self._batch, self._group,
            position=('f', self._positions),
            color=('f', (1.0, 1.0, 1.0, 1.0) * self._count)
        )

    def delete(self):
        if self._vertex_list is not None:
            self._vertex_list.delete()
            self._vertex_list = None


class Selectable:
    """
    Base class for all selectable objects.
    Creates :py:class:`BoundingBox` instance that matches its own dimensions.

    Selectable subclass must implement :py:meth:`get_aabb_min_max` in order to create the bounding box.
    If selectable is transformed, bounding box should also be transformed through :py:meth:`update_bounding_box`,
    before checking any intersections. For this purpose, `matrix` getter method should be provided in the subclass.

    Pass a :py:class:`Batch` if a visual representation of the bounding box is needed.
    Visuals will be toggled automatically based on selection status.
    """
    def __init__(self, batch: Batch = None):
        self._batch = batch
        self._group = None
        self._program = None
        self._selected = False

        self._min, self._max = self.get_aabb_min_max()

        # if batch is passed, visual representation is needed
        if self._batch:
            self._program = get_interface_shader_program()
            self._group = InterfaceDynamicGroup(self, self._program)

        self._bounding_box = self._get_bounding_box()

    @property
    def matrix(self) -> Mat4:
        """Return the current world transformation matrix."""
        raise NotImplementedError("Selectable needs access to model matrix in order to transform bounding box.")

    def get_aabb_min_max(self) -> tuple[Vec3, Vec3]:
        """Return an AABB in world space."""
        raise NotImplementedError("Selectable must implement this method in order to create its bounding box.")

    def select(self):
        if not self.selected:
            if self._batch:
                self._bounding_box.delete()
                self._bounding_box.create_vertex_list()
            self._selected = True

    def deselect(self):
        if self.selected:
            if self._bounding_box:
                self._bounding_box.delete()
            self._selected = False

    def update_bounding_box(self):
        self._bounding_box.update(self.matrix)

    def ray_intersects_aabb(self, ray: Ray) -> tuple[bool, float]:
        """Check if :py:class:`Ray` intersects :py:class:`BindingBox`"""
        return self._bounding_box.intersects_ray(ray)

    def _get_bounding_box(self) -> BoundingBox:
        return BoundingBox(self._min, self._max, program=self._program, batch=self._batch, group=self._group)

    @property
    def selected(self) -> bool:
        return self._selected

    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box


class InterfaceDynamicGroup(Group):
    """For dynamic updates of interface objects, such as bounding boxes."""
    def __init__(self, selectable: Selectable, program: ShaderProgram, order=0, parent: Group = None):
        super().__init__(order, parent)
        self.program = program
        self.selectable = selectable

    def set_state(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()
        self.program['rendering_dynamic_object'] = True
        self.program['model_precalc'] = self.selectable.matrix

    def unset_state(self) -> None:
        self.program['rendering_dynamic_object'] = False

    def __hash__(self):
        return hash((self.program, self.selectable, self.order, self.parent))

    def __eq__(self, other: "InterfaceDynamicGroup"):
        return (
                self.__class__ == other.__class__ and
                self.selectable is other.selectable and
                self.program is other.program and
                self.parent == other.parent and
                self.order == other.order
        )


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
        """Get :py:class:`Ray` instance from current mouse (x, y) position."""
        x, y = self.window._mouse_x, self.window._mouse_y
        width, height = self.window.width, self.window.height

        # convert from viewport space to normalized device coordinates (NDC)
        ndc_x = (2.0 * x) / width - 1.0
        ndc_y = (2.0 * y) / height - 1.0

        # convert from NDC to clip coordinates (z=-1 for vector pointing in, and adding w)
        clip_near = Vec4(ndc_x, ndc_y, -1.0, 1.0)
        clip_far = Vec4(ndc_x, ndc_y, 1.0, 1.0)

        # convert from clip to eye space by inverting projection, then to world space by inverting view matrix
        inv_vp = (self.window.projection @ self.window.view).__invert__()
        world_near = inv_vp @ clip_near
        world_far = inv_vp @ clip_far

        ray_origin = world_near.xyz / world_near.w
        ray_direction = world_far.xyz / world_far.w - ray_origin
        ray_direction.normalize()

        return Ray(ray_direction, self.camera.position, self.camera.z_far,
                   self.program, self.batch, self.group)

