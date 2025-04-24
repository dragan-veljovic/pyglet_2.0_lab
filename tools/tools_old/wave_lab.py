import numpy as np

from tools_old.graphics import *


class Ripples:
    """Concentric circles that expand from x,y centre and are added/removed
    in a periodic manner.
    """

    def __init__(self,
                 center_x, center_y,
                 source,
                 wavelength=200, radius_limit=200, phase_rad=0,
                 color=(255, 255, 255), delta_opacity=3,
                 variable_segments=True, update_source=False):

        # attribute from arguments
        self._x = center_x
        self._y = center_y
        self._source = source
        self._wavelength = wavelength
        self._radius_limit = radius_limit
        self._phase = phase_rad
        self._rgb = color
        self._delta_opacity = delta_opacity
        self._variable_segments = variable_segments
        self._update_source = update_source
        self._segments = 50  ## TO IMPLEMENT!

        self._batch = self._source._batch
        self._group = self._source._group

        # operating attributes
        self._step = 1  # how about wavelength / fps??
        self._ripples = []

        # testing of my method (erase if use isclose)
        self._previous_theta = self._source.theta
        self._difference = 0.0

    def update_position(self):
        # auto source update if assigned
        if self._update_source:
            self._source.update_position()

        # produce ripple on every cycle but use "is close" method to keep sync
        # if np.isclose((self._source.theta - self._phase) % (2 * np.pi), 0, atol=0.1 * abs(self._source.frequency)):

        # my method 10% more efficient by less robust
        threshold = 2 * np.pi / 12
        delta_theta = self._source.theta - self._previous_theta
        if delta_theta >= threshold:
            # compensating for remainder
            self._previous_theta = self._source.theta - (delta_theta - threshold)

            # self._ripples.append(
            #     CircleOutlineNumpy(self._x, self._y, 0, color=self._rgb, segments=self._segments,
            #                        batch=self._batch, group=self._group))

            self._ripples.append(
                CircleOutline(self._x, self._y, 0, color=self._rgb, opacity=int(255 * np.sin(self._previous_theta)),
                              segments=self._segments, batch=self._batch, group=self._group, thickness=10))

        # updating ripples radii
        for ripple in self._ripples:
            ripple.radius += self._step
            # fade ripples beyond limit
            if ripple.radius >= self._radius_limit:
                ripple.opacity -= self._delta_opacity
                # remove fade-out ripple
                if ripple.opacity <= self._delta_opacity:
                    self._ripples.remove(ripple)  # remove from window list
                    ripple.delete()  # remove from graphics batch

    def update_colors(self, value):
        for ripple in self._ripples:
            ripple.color = value

    @property
    def position(self):
        return self._x, self._y

    @position.setter
    def position(self, value: tuple[float, float]):
        self._x, self._y = value

    @property
    def color(self):
        return self._rgb

    @color.setter
    def color(self, value: tuple[int, int, int]):
        self._rgb = value
        self.update_colors()

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value: int):
        self._step = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value


class WaveSource:
    """
    A source for a ProgressiveWave or Ripple class.
    One can be passed to these upon instantiation.
    Same source can drive multiple of these waves.
    WaveSource just updates motion of a single point through passed x, y functions
    and the internal counter based on window's FPS.
    """

    def __init__(self, pos_x, pos_y, batch, group,
                 frequency=1, amplitude=100, show_marker=True,
                 function='sine', custom_func=None,
                 rotation_rad=0, phase_rad=0, fps=60,
                 marker_radius=5, marker_segments=None, marker_color=(255, 255, 255),
                 ):

        # attributes from parameters
        self._pos_x = pos_x
        self._pos_y = pos_y
        self._frequency = frequency
        self._amplitude = amplitude
        self._show_marker = show_marker
        self._phase = phase_rad
        self._function = function
        self._fps = fps
        self._batch = batch or pyglet.graphics.Batch()
        self._group = pyglet.shapes._ShapeGroup(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA, group)

        self._rotation = rotation_rad
        self._anchor_x = self._pos_x
        self._anchor_y = self._pos_y

        self._radius = marker_radius
        self._segments = marker_segments or max(8, int(self._radius / 1.25))
        self._color = marker_color

        if custom_func:
            if callable(custom_func):
                self._custom_func = custom_func
            else:
                raise TypeError("Passed custom_func argument must be a handle to callable function.")

        if self._show_marker:
            self._marker = GradientCircle(self._pos_x, self._pos_y, self._radius, self._segments,
                                          batch=self._batch, group=self._group, outer_color=self._color)

        # operating parameters
        self._theta = self._phase
        self._delta_theta = self._frequency * 2 * np.pi / self._fps
        # current absolute values
        self._x = self._pos_x
        self._y = self._pos_y
        # equilibrium position
        self._pos_array = np.array((self._pos_x, self._pos_y))

    def update_position(self):
        # determine new absolute value (displacement + equilibrium position)
        new_value = self._get_value() + self._pos_array

        # check if rotation
        if self._rotation:
            new_value_rotated = rotate_points_matrix(new_value, self._rotation, self._anchor_x, self._anchor_y)
            self._x, self._y = new_value_rotated
        else:
            self._x, self._y = new_value

        # increment counter
        self._theta += self._delta_theta

        # update marker
        if self._show_marker:
            self._marker.position = self._x, self._y

    def _get_value(self) -> np.ndarray:
        """Get current x,y displacement from equilibrium position."""
        if self._function == 'sine':
            return 0, self._amplitude * np.sin(self._theta + self._phase)
        elif self._function == 'cosine':
            return 0, self._amplitude * np.cos(self._theta + self._phase)
        elif self._function == 'square':
            return 0, self._amplitude * np.sign(np.sin(self._theta + self._phase))
        elif self._function == 'triangle':
            return 0, self._amplitude * (
                        2 * np.abs(2 * (0.16 * (self._theta + self._phase) * self._frequency - np.floor(
                    0.5 + 0.16 * self._theta * self._frequency))) - 1)
        elif self._function == 'sawtooth':
            return 0, self._amplitude * (2 * (0.16 * (self._theta + self._phase) - np.floor(
                0.5 + 0.16 * self._theta)))
        elif self._function == 'circular':
            return self._amplitude * np.cos(self._theta + self._phase), self._amplitude * np.sin(
                self._theta + self._phase)
        elif self._function == 'custom':
            if hasattr(self, "_custom_func"):
                return self._custom_func(self)
            else:
                raise AttributeError(
                    "To use custom function, pass a function handle upon WaveSource instantiation."
                )

    @property
    def value(self):
        return self._x, self._y

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value

    @property
    def anchor(self):
        return self._anchor_x, self._anchor_y

    @anchor.setter
    def anchor(self, value: tuple[float, float]):
        self._anchor_x, self._anchor_y = value

    @property
    def position(self):
        return np.array((self._pos_x, self._pos_y))

    @position.setter
    def position(self, value: tuple[float, float]):
        self._pos_x, self._pos_y = value
        self._pos_array = np.array((self._pos_x, self._pos_y))

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._delta_theta = self._frequency * 2 * np.pi / self._fps

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    @property
    def theta(self):
        return self._theta

    @property
    def fps(self):
        return self._fps


class ProgressiveWave:
    """ Progressive sine graph.
        Make sure to update WaveSource to see changes.
        If source is updated elsewhere make sure to set update_source=False.
        Negative wavelength changes direction of propagation.

        To implement!!
        change speed should not change the length!
        Implemented as hiding the points that exceeds the length
        (but as a bug also hiding those less than length)
        But 1000 points still are present and wave takes full space just not drawn.
        This negative effect the superposition which still is independent of speed
     """

    def __init__(self,
                 # all above can be taken from the source
                 source: WaveSource,
                 # shape
                 length: int = 1000, wavelength: int = 200,
                 # position
                 pos_x=None, pos_y=None,
                 # curve parameters
                 rotation_rad=None, step: int = 1,
                 # options
                 update_source=True,
                 # colors
                 color=(255, 255, 255), opacity=255,
                 ):

        # initialization from parameters
        self._source = source
        self._x = pos_x or self._source.position[0]
        self._y = pos_y or self._source.position[1]
        self._anchor_x, self._anchor_y = self._source.anchor
        self._batch = self._source._batch
        self._group = self._source._group

        self._wavelength = wavelength
        self._amplitude = self._source.amplitude
        self._frequency = self._source.frequency
        self._length = length
        self._step = step
        self._rgb = color
        self._opacity = opacity
        self._rotation = rotation_rad or self._source.rotation
        self._update_source = update_source

        # wave's velocity
        self._speed = self._wavelength * self._source.frequency
        self._speed_reference = self._speed
        self.direction_factor = self._speed / abs(self.speed)

        # programming attributes
        self.animate = True
        self._shift_increment = self._step * self.direction_factor
        self._shift_increment_components = self._get_shift_increment_components()

        # points
        self.points = self._generate_initial_points()
        # saving initial points for superposition
        self.initial_points_x = self.points[::, 0]
        self.adjusted_points = None

        # vertex list
        self.num_vertices = 2 * (len(self.points)) - 2
        self.vertex_list = self._batch.add(self.num_vertices, pyglet.gl.GL_LINES, self._group, 'v2f', 'c4B')

        # build
        self.update_position()
        self.update_color()

    def _generate_initial_points(self):
        """ Make initial straight line of undisturbed positions, horizontal or rotated. """
        x = np.arange(self._x, self._x + self._length * self.direction_factor, self._shift_increment)
        y = np.full_like(x, self._y)
        points = np.column_stack((x, y))

        if self._rotation:
            return rotate_points_numpy(points, self._rotation, self._anchor_x, self._anchor_y)
        else:
            return points

    def _get_shift_increment_components(self):
        """Calculates components of shift increment after rotation has been applied."""
        return np.array([self._shift_increment * np.cos(self._rotation),
                         self._shift_increment * np.sin(self._rotation)])

    def update_position(self):
        """
        Update position that includes correction for rotation. Call this to animate the graph.
        By rolling numpy array, we can re-use existing points and only re-calculate the first one.
        """
        if self.animate:
            if self._update_source:
                self._source.update_position()

            self.points += self._shift_increment_components

            # shift all points by moving last point (two coordinates) to first position
            self.points = np.roll(self.points, 2)

            # value that first point should have, before applying rotation
            new_point = self._source.value

            if not new_point:
                # failsafe in case custom function doesn't produce output
                new_point = self._x, self._y

            self.points[0] = new_point

            # trying to implement speed effect, resampling is done by interpolating
            # causes poor results with sharp corners
            # if self._step != 1:  # if step not initial step??
            #     points_to_interpolate = self.points[:len(self.points) // abs(self._step)]
            #     old_x = points_to_interpolate[::, 0]
            #     old_y = points_to_interpolate[::, 1]
            #
            #     if self.direction_factor < 0:
            #         # interp assumes ascending lists, have to reverse
            #         old_x = old_x[::-1]
            #         old_y = old_y[::-1]
            #
            #     new_x = np.linspace(old_x[0], old_x[-1], self._length)
            #     new_y = np.interp(new_x, old_x, old_y)
            #
            #     self.adjusted_points = np.column_stack([new_x, new_y])
            #     self.vertex_list.vertices[:] = get_gl_lines_vertices_numpy(self.adjusted_points)
            #
            # else:
            #     # get GL_LINES vertices and update vertex list
            #     self.vertex_list.vertices[:] = get_gl_lines_vertices_numpy(self.points)

            self.vertex_list.vertices[:] = get_gl_lines_vertices_numpy(self.points)

    # def update_position_no_rotation(self):
    #     """ Update without rotation parameter (but can use crazy shapes!)"""
    #     self.points[::, [0]] += self.shift_increment
    #     self.points = np.roll(self.points, 2)
    #     self.points[0] = self._x, self._amplitude * np.sin(self.theta) + self._y
    #     self.theta += self.delta_angle
    #
    #     # # testing rotation - crazy many shapes effect if points are rotated here on every frame
    #     # if self.rotation:
    #     #     self.points = rotate_points(self.points, self._rotation, self._anchor_x, self._anchor_y)
    #
    #     self.vertices = get_gl_lines_vertices_numpy(self.points)
    #     self.vertex_list.vertices[:] = self.vertices

    def update_color(self):
        self.vertex_list.colors[:] = [*self._rgb, self._opacity] * self.num_vertices

    def move_in_place(self, dx, dy):
        """ Move entire arrangement by passed dx, dy offset. """
        translation_matrix = np.array((dx, dy))
        self._x += dx
        self._y += dy
        self.points += translation_matrix
        self._source.position += translation_matrix

    # def amplitude_in_place(self, da):
    #     """ Increase amplitude of entire arrangement by given value. """
    #     self._amplitude += da
    #     thetas = np.arange(0, self.cycles * 2 * np.pi, self.delta_angle)
    #     s = self._get_value(thetas)
    #     self.points[::, 1] = self._y + s * np.cos(self._rotation)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def position(self):
        return self._x, self.y

    @position.setter
    def position(self, value: tuple):
        dx, dy = value[0] - self._x, value[1] - self._y
        self.move_in_place(dx, dy)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value: int):
        # implementation for different speed effect
        # what if step == 1 and value is < 1?
        self._step = value
        self._shift_increment = self._step * self.direction_factor
        self._shift_increment_components = self._get_shift_increment_components()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value
        self._wavelength = self._speed / self._source.frequency
        self._shift_increment = self._speed / self._speed_reference
        self._shift_increment_components = self._get_shift_increment_components()

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def color(self):
        return self._rgb

    @color.setter
    def color(self, value):
        self._rgb = value
        self.update_color()

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value: int):
        self._opacity = value
        self.update_color()

    @property
    def anchor(self):
        """Anchor is used as rotation center."""
        return self._anchor_x, self._anchor_y

    @anchor.setter
    def anchor(self, value):
        self._anchor_x, self._anchor_y = value

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, theta_rad):
        delta_theta = theta_rad - self._rotation
        self._rotation = theta_rad
        self.points = rotate_points_matrix(self.points, delta_theta, self._anchor_x, self._anchor_y)
        self._shift_increment_components = self._get_shift_increment_components()

    @property
    def amplitude(self):
        return self._source.amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._source.amplitude = value

    @property
    def frequency(self):
        return self._source.frequency

    @frequency.setter
    def frequency(self, value):
        self._source.frequency = value

    @property
    def update_source(self):
        return self._update_source

    @update_source.setter
    def update_source(self, value: bool):
        self._update_source = value

    def delete(self):
        self.vertex_list.delete()
        self.vertex_list = None
        # But is wave's source out of scope?


class Superposition:
    """
    Edit: attempt to implement using np.sum(ndarray_list) doesn't work.
    Also, when progressive wave is negative need some way subtract, currently using np.flipud
    Finally Progressive wave's speed is not reflected in the superposition.
    """

    def __init__(self,
                 waves: list[ProgressiveWave, ...],
                 pos_x=None, pos_y=None, step=None,
                 batch=None, group=None,
                 color=(255, 255, 255), opacity=255):
        # initializing attributes from parameters
        self._waves = waves
        self._waves_points = np.array([wave.points for wave in self._waves])
        self._rgb = color
        self._opacity = opacity
        self._batch = batch or pyglet.graphics.Batch()
        self._group = pyglet.shapes._ShapeGroup(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA, group)

        # arrangement left side coordinates
        self._x = pos_x or self._waves[0].x
        self._y = pos_y or self._waves[0].y
        self._step = step or self._waves[0].step

        self._initial_points = self._generate_initial_points()
        self._points = np.empty_like(self._initial_points)
        self._validate_wave_lengths()

        # vertex list
        self.num_vertices = 2 * (len(self._initial_points) - 1)
        self.vertex_list = self._batch.add(self.num_vertices, pyglet.gl.GL_LINES, self._group, 'v2f', 'c4B')

        self.update_position()
        self.update_color()

    def _validate_wave_lengths(self):
        pass
        # validation_reference = len(self._initial_points)
        # for wave in self._waves:
        #     points = wave.adjusted_points if wave.adjusted_points is not None else wave.points
        #     if len(points) != validation_reference:
        #         raise ValueError("Length of the points array must be equal for all waves.")

    def _generate_initial_points(self):
        """
        Generate initial straight line of undisturbed positions and allocate point number for vertex list.
        These values will be used as zero reference upon which other displacement will be added on each update.
        """
        x_list = np.arange(self._x, self._x + len(self._waves[0].points), self._step, dtype='float64')
        y_list = np.full_like(x_list, self._y)
        return np.column_stack((x_list, y_list))

    def update_color(self):
        self.vertex_list.colors[:] = [*self._rgb, self._opacity] * self.num_vertices

    def update_position(self):
        # can sum be used instead of for loop?
        # resultant = np.sum(self._waves_points, axis=0)

        # reset the sum to zero level
        self._points = np.copy(self._initial_points)
        for wave in self._waves:
            # implementation of speed effect
            if wave.adjusted_points is not None:
                points = wave.adjusted_points
            else:
                points = wave.points

            if wave.speed < 0:
                # flip points array for a wave that is moving in opposite direction
                self._points[::, 1] += np.flipud(points[::, 1] - wave.y)
                self._points[::, 0] += np.flipud(points[::, 0] - wave.initial_points_x)
            else:
                # get wave's relative displacements and add to sum
                self._points[::, 1] += points[::, 1] - wave.y
                self._points[::, 0] += points[::, 0] - wave.initial_points_x

        # get vertices and update vertex list
        self.vertex_list.vertices[:] = get_gl_lines_vertices_numpy(self._points)

    @property
    def points(self):
        return self._points

    @property
    def step(self):
        return self._step
