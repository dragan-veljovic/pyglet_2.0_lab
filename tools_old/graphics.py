"""
This module includes classes and functions for creating and drawing new graphics.
Some are new basic shapes that follow built-in pyglet.shapes methodology, while others
are custom graphics arrangements bundled with their own behaviour methods, animations etc.
"""
import math

import numpy as np
import pyglet


def get_gl_triangles_strip_vertices(points: list[tuple[float, float], ...], thickness=2, closed=False) -> list:
    """
    Processes ordinary sequence of (x,y) coordinates and returns arranged
    GL_TRIANGLES vertex list for creating triangle strip, effectively drawing a thick line
    through passed points. Corners are merged for smooth segment connection.
    """
    triangle_vertices = []
    hw = thickness / 2
    prev_x3y3x4y4 = []

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        # calculate dx and dy sides of the triangle constructed with p1 and p2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        # finding normal by inverting deltas
        nx, ny = -dy, dx
        # calculating the length of the normal
        normal_length = (nx ** 2 + ny ** 2) ** 0.5
        if normal_length > 0:
            # normalizing the length of the normal deltas
            nx, ny = nx / normal_length, ny / normal_length

            # Merge corners by assigning previous x3,y3 and x4,y4 to new x1,y1 and x2,y2 respectively
            if prev_x3y3x4y4:
                x1, y1 = prev_x3y3x4y4[0], prev_x3y3x4y4[1]
                x2, y2 = prev_x3y3x4y4[2], prev_x3y3x4y4[3]
            else:
                # if not available, it's first point in range, so calculate new x1,y1 and x2,y2 instead
                x1, y1 = p1[0] + nx * hw, p1[1] + ny * hw
                x2, y2 = p1[0] - nx * hw, p1[1] - ny * hw

            # if this is the final segment, close the shape
            if i == len(points) - 2 and closed:
                x3, y3 = triangle_vertices[0], triangle_vertices[1]
                x4, y4 = triangle_vertices[2], triangle_vertices[3]
            else:
                # calculate new x3,y3 and x4,y4
                x3, y3 = p2[0] + nx * hw, p2[1] + ny * hw
                x4, y4 = p2[0] - nx * hw, p2[1] - ny * hw

            # appending coordinates for two triangles
            triangle_vertices.extend((x1, y1, x2, y2, x3, y3))
            triangle_vertices.extend((x2, y2, x3, y3, x4, y4))

            # saving previous x3,y3 and x4,y4
            prev_x3y3x4y4 = [x3, y3, x4, y4]

        else:
            # p1 aligns with p2, fill data to avoid vertex list length error
            triangle_vertices.extend(p1 * 6)

    return triangle_vertices


def get_double_quad_strip_vertices(points: list[tuple[float, float], ...], thickness=4, closed=False):
    """
    Processes ordinary sequence of (x,y) coordinates and returns arranged
    GL_TRIANGLES vertex list used to add a quad-strip on each side of a line segment
    between two points. Corners are merged for smooth segment connection.
    Set closed=True to connect last segment with the first one for closed shapes.

    This effectively draws a thick line through passed points, but also gives
    your shape an option to have three different colors along the thickness.

    Example how to properly arrange these colors in 'c4B' form:

    def update_color(self):
        colors = (midline_rgba * 2 + first_edge_rgba * 3
                  + midline_rgba * 3 + second_edge_rgba * 3
                  + midline_rgba) * shape_segments
        vertex_list.colors[:] = colors
    """

    vertices = []
    half_thickness = thickness / 2
    segments = len(points) - 1
    prev_p3 = None
    prev_p5 = None
    init_p4 = None
    init_p6 = None

    for i in range(segments):
        p1 = points[i]
        p2 = points[i + 1]
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        length = (delta_x ** 2 + delta_y ** 2) ** 0.5
        # calculating normalized components of the normal to the segment
        nx, ny = -delta_y / length, delta_x / length
        # thickness increments
        dx, dy = nx * half_thickness, ny * half_thickness

        if i == 0:
            # if first segment, calculate everything and save initial corners
            p3 = p2[0] + dx, p2[1] + dy
            p4 = p1[0] + dx, p1[1] + dy
            p5 = p2[0] - dx, p2[1] - dy
            p6 = p1[0] - dx, p1[1] - dy
            init_p4, init_p6 = p4, p6

        elif i == segments - 1 and closed:
            # if last segment, merge final and initial corners to close the shape
            p3 = init_p4
            p4 = prev_p3
            p5 = init_p6
            p6 = prev_p5

        else:
            # if other segments, merge starting corners with the previous end corners
            p3 = p2[0] + dx, p2[1] + dy
            p4 = prev_p3
            p5 = p2[0] - dx, p2[1] - dy
            p6 = prev_p5

        # save end corners
        prev_p3, prev_p5 = p3, p5

        # append vertices of two double-triangles, forming quads on each side of a segment
        vertices.extend((*p1, *p2, *p3, *p3, *p4, *p1))
        vertices.extend((*p1, *p2, *p5, *p5, *p6, *p1))

    return vertices


def get_gl_lines_vertices(points: list[tuple[float, float], ...], closed=False) -> list:
    """
    Processes ordinary sequence of (x, y) coordinates are returns arranged
    GL_LINES vertex list for creating a line strip through passed points.
    Set closed=True to connect last segment with the first one for closed shapes.
    """
    line_vertices = []
    for i in range(len(points) - 1):
        line_points = *points[i], *points[i + 1]
        line_vertices.extend(line_points)

    # manually combining last and first point to close a circle
    if closed:
        last_pair = *points[-1], *points[0]
        line_vertices.extend(last_pair)

    return line_vertices


def get_gl_lines_vertices_numpy(points) -> np.ndarray:
    """
    Optimized version of get_gl_lines_vertices().
    Creates two trimmed copies of a ((x1, y1), (x2,y2), ...) point sequence
    to create GL_LINES vertex list in a form (x1, y1, x2, y2, x2, y2, x3, y3, ...).
    """
    return np.column_stack((points[:-1], points[1:])).flatten()


def rotate_points(points, theta_rad, anchor_x, anchor_y) -> list:
    """
    For testing purposes only - use numpy matrix version instead!
    Rotates each point from the points list around anchor by theta radians.
    2x slower pyglet.shapes._rotate(), as uses geometry rather than matrix calc.
    """
    rotated_points = []
    for point in points:
        x, y = point[0] - anchor_x, point[1] - anchor_y
        # displacement from anchor
        s = math.sqrt(x**2 + y**2)
        # current angle to horizontal
        alpha = math.atan2(y, x)
        # add components of rotated displacement vector to anchor
        x_new = anchor_x + s * math.cos(alpha + theta_rad)
        y_new = anchor_y + s * math.sin(alpha + theta_rad)
        # append rotated coordinates
        rotated_points.append((x_new, y_new))
    return rotated_points


def rotate_points_numpy(points, theta_rad, anchor_x, anchor_y) -> np.ndarray:
    """
    Optimized numpy version of the pyglet.shapes._rotate() algorithm.
    60 calls for a 1000 point array yield around 10x faster execution time.
    Points should be passed as np.ndarray([[x1, y1], [x2, y2], ...]).
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    x_array = points[:, 0] - anchor_x
    y_array = points[:, 1] - anchor_y
    displacements = np.hypot(x_array, y_array)
    alphas = np.arctan2(y_array, x_array)

    x_new_array = anchor_x + displacements * np.cos(alphas + theta_rad)
    y_new_array = anchor_y + displacements * np.sin(alphas + theta_rad)

    return np.column_stack((x_new_array, y_new_array))


def rotate_points_matrix(points, theta_rad, anchor_x, anchor_y) -> np.ndarray:
    """
    Highly optimized version of pyglet.shapes._rotate() algorithm, using matrices.
    60 calls for a 1000 point list yield 20x faster execution time.
    Points should be passed as np.ndarray([[x1, y1], [x2, y2], ...]).
    """
    # prepare anchor matrix
    anchor = np.array((anchor_x, anchor_y))
    # pre-calculate sin/cos terms
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    # form rotation matrix
    rotation_matrix = np.array([[c, s],
                                [-s,  c]])
    # multiply matrices, compensating for anchor
    return np.dot(points - anchor, rotation_matrix) + anchor


def scale_points(points, anchor_x, anchor_y, scale_x, scale_y=None):
    """
    Scales each point of the passed points list by multiplying
    its displacement from anchor by the scale factor.
    """
    scaled_points = []
    scale_y = scale_y or scale_x
    for point in points:
        x, y = point[0], point[1]
        dx, dy = x - anchor_x, y - anchor_y
        x_new = anchor_x + dx * scale_x
        y_new = anchor_y + dy * scale_y
        scaled_points.append((x_new, y_new))
    return scaled_points


def scale_points_numpy(points, anchor_x, anchor_y, scale_x, scale_y=None):
    """
    Numpy version of scale_points algorithm. The fastest scale algorithm.
    10x faster execution (60 calls for 1000 vertices) and increases for larger sets.
    Points should be passed as np.ndarray([[x1, y1], [x2, y2], ...]).
    """
    scale_y = scale_y or scale_x
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    x_new = points[:, [0]] * scale_x + anchor_x * (1 - scale_x)
    y_new = points[:, [1]] * scale_y + anchor_y * (1 - scale_y)
    return np.column_stack((x_new, y_new))


def scale_points_matrix(points, anchor_x, anchor_y, scale_x, scale_y=None):
    """
    Scaling of points using matrices. Execution speed is comparable to scale_points_numpy()
    for small data sets, but gets proportionally longer for larger data sets.
    Points should be passed as np.ndarray([[x1, y1], [x2, y2], ...]).
    """
    # if scale_y is not passed, take it same as scale_x
    scale_y = scale_y or scale_x
    # prepare anchor matrix
    anchor_matrix = np.array([anchor_x, anchor_y])
    # form scale matrix
    scale_matrix = np.array([[scale_x, 0],
                             [0, scale_y]])
    # multiply matrices, compensating for anchor
    return np.dot(points - anchor_matrix, scale_matrix) + anchor_matrix


class CircleOutline:
    def __init__(self, x, y, radius=50, thickness=None, segments=None, color=(255, 255, 255), batch=None, group=None,
                 opacity=255, closed=True):
        """
        Create circle outline with, or without thickness.
        If thickness is specified it needs to get_gl_triangles_strip_vertices() on every update,
        increasing CPU usage.
        """

        # initialized attributes
        self._x = x
        self._y = y
        self._radius = radius
        self._thickness = thickness
        self._rgb = color
        self.segments = segments or max(14, int(radius / 1.25))
        self._opacity = opacity
        self.closed = closed

        # batch and group
        self.batch = batch or pyglet.graphics.Batch()
        self.group = pyglet.shapes._ShapeGroup(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA, group)

        # vertex list
        if self._thickness:
            self.num_triangle_vertices = self.segments * 6
            self.vertex_list = self.batch.add(self.num_triangle_vertices, pyglet.gl.GL_TRIANGLES, self.group, 'v2f',
                                              'c4B')
        else:
            self.num_triangle_vertices = self.segments * 2 + 2 if closed else 0
            self.vertex_list = self.batch.add(self.num_triangle_vertices, pyglet.gl.GL_LINES, self.group, 'v2f', 'c4B')

        # build
        self.update_position()
        self.update_color()

    def update_position(self):
        theta = 0
        dtheta = 2 * np.pi / self.segments
        points = []

        for n in range(self.segments + 1):
            x = self._x + self._radius * np.cos(theta)
            y = self._y + self._radius * np.sin(theta)
            theta += dtheta

            points.append((x, y))

        if self._thickness:
            """ Prepare vertices for GL_TRIANGLES, to simulate thickness with triangle strip. """
            self.vertex_list.vertices[:] = get_gl_triangles_strip_vertices(points, thickness=self._thickness,
                                                                           closed=self.closed)

        else:
            """ Prepare vertices for GL_LINES, for zero thickness line strip."""
            self.vertex_list.vertices[:] = get_gl_lines_vertices(points, closed=self.closed)

    def update_color(self):
        self.vertex_list.colors[:] = [*self._rgb, self._opacity] * self.num_triangle_vertices

    def delete(self):
        self.vertex_list.delete()
        self.vertex_list = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.update_position()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.update_position()

    @property
    def position(self):
        return self._x, self._y

    @position.setter
    def position(self, values):
        self._x, self._y = values
        self.update_position()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.update_position()

    @property
    def color(self):
        return self._rgb

    @color.setter
    def color(self, value: tuple[int, int, int]):
        self._rgb = value
        self.update_color()

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        if value in range(0, 256):
            self._opacity = value
            self.update_color()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        if isinstance(value, (int, float)):
            self._thickness = value
            self.update_position()


class CircleOutlineNumpy:
    def __init__(self, x, y, radius=50, segments=None,
                 color=(255, 255, 255), opacity=255, thickness=None,
                 batch=None, group=None, closed=True):
        """
        More efficient version of CircleOutline, for larger segment counts.
        Uses Numpy vectorized operations to speed-up update_position method.
        For segments < 20, pure python version is equal or faster.
        For segments=50 this version is 5x faster.
        For segments=100 this version is 10x faster, and so on (up to 50x).
        """
        self._x = x
        self._y = y
        self._radius = radius
        self._rgb = color
        self._opacity = opacity
        self._thickness = thickness
        self._segments = segments or max(14, int(radius / 1.25))
        self._closed = closed

        # batch and group
        self.batch = batch or pyglet.graphics.Batch()
        self.group = pyglet.shapes._ShapeGroup(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA, group)

        # vertex list
        self.num_vertices = self._segments * 2
        self.vertex_list = self.batch.add(self.num_vertices, pyglet.gl.GL_LINES, self.group, 'v2f', 'c4B')

        # build
        self.update_position()
        self.update_color()

    def update_position(self):
        theta = np.linspace(0, 2 * np.pi, self._segments + 1, dtype=np.float32)
        x = self._x + self._radius * np.cos(theta)
        y = self._y + self._radius * np.sin(theta)
        points = np.column_stack((x, y)).astype(np.float32)

        start_points = points[:-1]
        end_points = points[1:]

        line_vertices = np.column_stack((start_points, end_points)).flatten()

        # if self._closed:
        #     last_pair = np.column_stack((points[-1], points[0])).flatten()
        #     line_vertices = np.concatenate((line_vertices, last_pair))

        self.vertex_list.vertices[:] = line_vertices

    def update_color(self):
        self.vertex_list.colors[:] = [*self._rgb, self._opacity] * self.num_vertices

    def delete(self):
        self.vertex_list.delete()
        self.vertex_list = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.update_position()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.update_position()

    @property
    def position(self):
        return self._x, self._y

    @position.setter
    def position(self, values):
        self._x, self._y = values
        self.update_position()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.update_position()

    @property
    def color(self):
        return self._rgb

    @color.setter
    def color(self, value: tuple[int, int, int]):
        self._rgb = value
        self.update_color()

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        if value in range(0, 256):
            self._opacity = value
            self.update_color()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        if isinstance(value, (int, float)):
            self._thickness = value
            self.update_position()


class Dashedline:
    def __init__(self, x, y, x1, y1, batch=None, group=None, color=(255, 255, 255, 255), step: int = 20):
        self._x = x
        self._y = y
        self._x1 = x1
        self._y1 = y1
        self._batch = batch or pyglet.graphics.Batch()
        self._group = pyglet.shapes._ShapeGroup(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA, group)
        self._color = color
        self._step = step

        self._vertices = None
        self._colors = None
        self._vertex_list = self._get_vertex_list()

    def _get_vertices(self):
        """ Problem when x = x1 (dx = 0) """
        dx, dy = self._x1 - self._x, self._y1 - self._y
        angle = np.arctan2(dy, dx)
        x_coords, y_coords = None, None
        if dx:
            x_coords = np.arange(self._x, self._x1, self._step*np.cos(angle))
        if dy:
            y_coords = np.arange(self._y, self._y1, self._step*np.sin(angle))

        if x_coords is not None and y_coords is not None:
            return np.column_stack((x_coords, y_coords)).flatten()
        else:
            return None

    def _get_colors(self):
        return self._color * (len(self._vertices) // 2)

    def _get_vertex_list(self):
        new_vertices = self._get_vertices()
        if new_vertices is not None:
            self._vertices = self._get_vertices()
        self._colors = self._get_colors()
        count = len(self._vertices) // 2
        return self._batch.add(
            count, pyglet.gl.GL_LINES, self._group,
            ('v2f', self._vertices),
            ('c4B', self._colors),
        )

    def _update_position(self):
        self._vertex_list.vertices[:] = self._get_vertices()

    def _update_color(self):
        self._vertex_list.colors[:] = self._get_colors()

    def set_start_end(self,
                      start: tuple[float, float],
                      end: tuple[float, float]):
        self._x, self._y = start
        self._x1, self._y1 = end
        self._vertex_list.delete()
        self._vertex_list = self._get_vertex_list()




class CircleRingThreeColored:
    """ Circular ring with thickness, with separate color parameters for
        inner, center and outer regions.
    """

    def __init__(self, center_x, center_y, radius, thickness: int = 2, segments: int = None,
                 inner_rgba=None, center_rgba=(255, 255, 255, 255), outer_rgba=None,
                 batch=None, group=None):

        # attributes from arguments
        self._x = center_x
        self._y = center_y
        self._radius = radius
        self._thickness = thickness
        self._segments = segments or max(12, int(self._radius / 1.25))
        self._cen_rgba = center_rgba
        self._inn_rgba = inner_rgba or self._cen_rgba
        self._out_rgba = outer_rgba or self._cen_rgba

        # graphics
        self._points = self._generate_points()
        self._num_vertices = (len(self._points) - 1) * 12
        self._batch = batch or pyglet.graphics.Batch()
        self._group = pyglet.shapes._ShapeGroup(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA, group)
        self._vertex_list = self._batch.add(self._num_vertices, pyglet.gl.GL_TRIANGLES, self._group,
                                            'v2f', 'c4B')
        # build
        self.update_color()
        self.update_position()

    def update_position(self):
        self._vertex_list.vertices[:] = get_double_quad_strip_vertices(self._points, self._thickness, closed=True)

    def update_color(self):
        colors = (self._cen_rgba * 2 + self._inn_rgba * 3 + self._cen_rgba * 3 + self._out_rgba * 3 +
                  self._cen_rgba) * self._segments
        self._vertex_list.colors[:] = colors

    def _generate_points(self):
        thetas = np.linspace(0, 2 * np.pi, self._segments + 1)
        return np.column_stack(
            (self._x + self._radius * np.cos(thetas),
             self._y + self._radius * np.sin(thetas))
        )

    @property
    def center_rgba(self):
        return self._cen_rgba

    @center_rgba.setter
    def center_rgba(self, value: tuple[int, int, int, int]):
        self._cen_rgba = value
        self.update_color()

    @property
    def inner_rgba(self):
        return self._inn_rgba

    @inner_rgba.setter
    def inner_rgba(self, value: tuple[int, int, int, int]):
        self._inn_rgba = value
        self.update_color()

    @property
    def outer_rgba(self):
        return self._cen_rgba

    @outer_rgba.setter
    def outer_rgba(self, value: tuple[int, int, int, int]):
        self._cen_rgba = value
        self.update_color()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self._points = self._generate_points()
        self.update_position()


# for testing purposes
def thick_line_trig(points: tuple[[float, float], ...], width=1) -> list:
    """ Process series of simple x,y coordinates to make line segments thick using GL triangles.
        Uses trigonometric functions to get angle and extension deltas for triangle points.

        For test purposes only!
        Use get_gl_triangles_strip_vertices() instead.
    """
    triangle_vertices = []
    hw = width / 2
    for n in range(len(points) - 1):
        cur = points[n]
        nxt = points[n + 1]
        # get angle to horizontal to "create width"
        alpha = np.atan2(cur[0] - nxt[0], nxt[1] - cur[1])
        # get extension deltas in each direction
        dx = hw * np.cos(alpha)
        dy = hw * np.sin(alpha)

        to_extend = cur[0] - dx, cur[1] - dy, cur[0] + dx, cur[1] + dy, nxt[0] - dx, nxt[1] - dy
        triangle_vertices.extend(to_extend)
        to_extend = cur[0] + dx, cur[1] + dy, nxt[0] - dx, nxt[1] - dy, nxt[0] + dx, nxt[1] + dy
        triangle_vertices.extend(to_extend)

    return triangle_vertices


class GradientCircle(pyglet.shapes.Circle):
    def __init__(self, x, y, radius,
                 segments=None, batch=None, group=None,
                 inner_color=(255, 255, 255), outer_color=(255, 255, 255)):
        super().__init__(x, y, radius, segments, batch=batch, group=group)
        self.inner_color = inner_color
        self.outer_color = outer_color
        self.inner_opacity = 255
        self.outer_opacity = 255
        self.new_update_color()

    def new_update_color(self):
        self._vertex_list.colors[:] = [*self.inner_color, int(self.inner_opacity),
                                       *self.outer_color, int(self.outer_opacity),
                                       *self.outer_color, int(self.outer_opacity)] * self._segments
