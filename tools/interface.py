from tools.camera import Camera3D
import pyglet
from pyglet.gl import *
from pyglet.math import Vec3, Vec4, Mat4
from pyglet.window import Window
from pyglet.graphics import Batch, Group, ShaderGroup
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

        If ray is to be constructed from mouse inputs (selection interface in a 3D scene, for example),
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
            self._vertex_list = None

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
    Initially, creates an axis-aligned bounding box (AABB) from provided min, max corner vectors.
    It can be used for selection or collision mechanisms.

    Bounding box can later be oriented through :py:meth:`update`, and whether :py:class:`Ray` is passing through
    its volume can be checked with :py:meth:`intersects_ray` OBB algorithm.

    Normally, :py:class:`Selectable` would create and manage updates of its own bounding box instance.
    Visualisation of a bounding box can be toggled on demand with :py:meth:`create_vertex_list` and :py:meth:`delete`.
    This is more efficient than keeping all boxes in the GPU buffer, as usually only few are visible in the scene.
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

        # buffer data for faster updates
        self._original_corners = [
            # front face
            Vec3(self._min.x, self._min.y, self._min.z),
            Vec3(self._max.x, self._min.y, self._min.z),
            Vec3(self._max.x, self._max.y, self._min.z),
            Vec3(self._min.x, self._max.y, self._min.z),

            # back face
            Vec3(self._min.x, self._min.y, self._max.z),
            Vec3(self._max.x, self._min.y, self._max.z),
            Vec3(self._max.x, self._max.y, self._max.z),
            Vec3(self._min.x, self._max.y, self._max.z),
        ]

        indices = [
            0, 1, 1, 2, 2, 3, 3, 0,  # construct front face
            4, 5, 5, 6, 6, 7, 7, 4,  # construct back face
            0, 4, 1, 5, 2, 6, 3, 7  # connect two faces
        ]

        self._positions = [coord for idx in indices for coord in self._original_corners[idx]]
        self._count = len(self._positions) // 3

        # visual representation controlled externally on demand
        self._vertex_list = None

        # for updates and OBB algorithm
        self._corners = self._original_corners.copy()
        self._center_point = None
        self._axes = []
        self._half_lengths = []

    def get_center(self) -> Vec3:
        return sum(self._corners, Vec3(0, 0, 0)) * (1.0 / 8)

    def update(self, matrix: Mat4):
        """
        Transform original box corners into world space using the model matrix.
        OBB keeps these corners for later intersection tests.
        """
        self._corners = [
            (matrix @ Vec4(corner.x, corner.y, corner.z, 1)).xyz
            for corner in self._original_corners
        ]

        self._extract_obb_data()

    def _extract_obb_data(self):
        """
        Recalculates center, axes, and half-lengths from transformed corners.
        Assumes `self._corners` are up-to-date.
        """
        # Center of the box
        self._center_point = self.get_center()

        # Local axes: from corner 0 to corners 1, 3, 4 (right, up, forward)
        axis_x = (self._corners[1] - self._corners[0]).normalize()
        axis_y = (self._corners[3] - self._corners[0]).normalize()
        axis_z = (self._corners[4] - self._corners[0]).normalize()

        self._axes = [axis_x, axis_y, axis_z]
        self._half_lengths = [
            (self._corners[1] - self._corners[0]).length() / 2,
            (self._corners[3] - self._corners[0]).length() / 2,
            (self._corners[4] - self._corners[0]).length() / 2,
        ]

    def intersects_ray(self, ray: Ray) -> tuple[bool, float]:
        """
        Ray vs OBB intersection using the slab method (SAT).
        Returns (hit: bool, distance: float)
        """
        p = self._center_point - ray.origin
        tmin = -float("inf")
        tmax = float("inf")

        for i in range(3):
            axis = self._axes[i]
            e = axis.dot(p)
            f = ray.direction.dot(axis)

            if abs(f) > 1e-6:
                t1 = (e + self._half_lengths[i]) / f
                t2 = (e - self._half_lengths[i]) / f
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False, 0.0
            else:
                # Ray is parallel to the slab
                if -e - self._half_lengths[i] > 0 or -e + self._half_lengths[i] < 0:
                    return False, 0.0

        return True, tmin if tmin > 0 else tmax

    def create_vertex_list(self):
        """Visual representation of a bounding box - a wireframe cuboid."""
        self.delete()

        self._vertex_list = self._program.vertex_list(
            self._count, GL_LINES, self._batch, self._group,
            position=('f', self._positions),
            color=('f', (1.0, 1.0, 1.0, 1.0) * self._count)
        )

    def delete(self):
        if self._vertex_list:
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

    Note that while visual representation of a bounding box is updated on every draw call through
    :py:class:`InterfaceDynamicGroup`, to reduce CPU load, its actual orientation used for
    intersection calculations will only be updated after :py:meth:`update_bounding_box` is called.
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
            self.program_group = ShaderGroup(self._program)
            self._group = InterfaceDynamicGroup(self, self._program, parent=self.program_group)

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


class SelectionManager:
    """
    Manager that tracks selection of :py:class:`Selectable` objects in passed :py:attr:`selectables` set.

    Can be toggled on and off by setting :py:attr:`active` boolean.

    If manager is active, it will handle mouse inputs automatically:
        - select individual selectable with LEFT mouse click
        - add to selection with CTRL+LEFT mouse click
        - remove from selection with CTRL+ALT+LEFT mouse click

    **Example usage:**

        class App(pyglet.window.Window):
            def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.camera = Camera3D(self)
            self.selectables = set()
            self.selection = SelectionManager(self, self.camera, self.selectables, self.batch)

            # ... add some `Selectable` to the set

            def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
                if self.selection:
                    for selected_item in self.selection:
                        selected_item.position += Vec3(dx, 0,0, dy)

    TODO: Batch passed to manager controls visibility of bounding boxes, remove this burden from `Selectable`
    """
    def __init__(
            self,
            window: Window,
            camera: Camera3D,
            selectables: set[Selectable],
            batch: Batch | None = None,
            active=True,
    ):
        self._window = window
        self._camera = camera
        self._selectables = selectables
        self._batch = batch

        self._active = active
        self._selected = set()
        self._mouse_picker = MousePicker(self._window, self._camera)
        self._mouse_pressed = False
        self._mouse_dragging = False
        if active:
            self._window.push_handlers(self)

    def deselect_all(self):
        for item in self._selected:
            item.deselect()
        self._selected = set()
        # Issue detected - creating vertex lists slows down rendering, but slowdown persists even after its deletion.
        # should call this periodically, and is a must after large number of vertex lists has been deleted!
        if self._batch:
            self._batch.invalidate()

    def select_all(self):
        for item in self._selectables:
            item.select()
            self._selected.add(item)

    def select(self, modifiers: int):
        min_dist = float('inf')
        closest = None

        for item in self._selectables:
            # Update position of binding box before selection
            item.update_bounding_box()
            ray = self._mouse_picker.get_mouse_ray()
            hit, dist = item.ray_intersects_aabb(ray)
            if hit and dist < min_dist:
                closest = item
                min_dist = dist

        if closest:
            # removal of already selected object
            if modifiers == 6:  # CTRL + ALT mods pressed
                self._selected.discard(closest)
                closest.deselect()

            # adding new object to selection
            elif modifiers == 2:  # CTRL pressed
                self._selected.add(closest)
                closest.select()
            else:
                # single selection, deselect everything else
                for item in self._selected:
                    item.deselect()
                closest.select()
                self._selected = {closest}
        else:
            # empty space clicked, deselect everything
            self.deselect_all()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        if not self._window.gui_interaction():
            self._mouse_pressed = True
            self._mouse_dragging = False

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        if not self._window.gui_interaction():
            if self._mouse_pressed and not self._mouse_dragging:
                if button == pyglet.window.mouse.LEFT:
                    self.select(modifiers)

                if button == pyglet.window.mouse.RIGHT:
                    self.deselect_all()
            self._mouse_pressed = False
            self._mouse_dragging = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        self._mouse_dragging = True

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.A and modifiers & pyglet.window.key.MOD_CTRL:
            self.select_all()

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        if value:
            if not self._active:
                self._window.push_handlers(self)
                self._active = True
        else:
            if self._active:
                self._window.remove_handlers(self)
                self._active = False

    def __iter__(self):
        return iter(self._selected)

    def __len__(self):
        return len(self._selected)

    def __contains__(self, item):
        return item in self._selected

    def __bool__(self):
        return bool(self._selected)


class InterfaceDynamicGroup(Group):
    """For dynamic updates of interface objects, such as bounding boxes."""
    def __init__(self, selectable: Selectable, program: ShaderProgram, order=0, parent: Group = None):
        super().__init__(order, parent)
        self.program = program
        self.selectable = selectable

    def set_state(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
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




