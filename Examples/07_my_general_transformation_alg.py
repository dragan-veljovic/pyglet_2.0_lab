""" Exploring ways to apply any vector transformation and optimize algorithm."""
import pyglet
from tools_old.definitions import *
import numpy as np
from tools_old.color import *
import math

WIDTH = 1280
HEIGHT = 720


def linear_transform(points: np.ndarray,
                     i_t=np.array([1, 0, 0]),
                     j_t=np.array([0, 1, 0]),
                     k_t=np.array([0, 0, 1]),
                     anchor=np.array([0, 0, 0])
                     ):
    """
    General method for applying linear transformation to a set of
    np.array[(x1, y1, z1), (x2, y2, z2), ...] coordinates.
    This is achieved by passing arrays i_t, j_t, k_t that contain components
    of transformed unit vectors. Default arguments do not apply any transformation.

    Example: apply 0.5 radians 2D rotation to passed points
    i_t=np.array([np.cos(0.5), np.sin(0.5), 0]),
    j_t=np.array([-np.sin(0.5), np.cos(0.5), 0]),

    Example: apply horizontal shear to passed points
    j_t=np.array([0.5, 1, 0]),
    anchor=np.array((screen_width // 2, screen_height // 2, 0)),

    TO DO:
    2) stack + reset stack on points could just be reverse-stacking ijk_t?
    3) linear_transform vertex_list vertices directly?

    """

    # form transformation matrix by column-stacking transformed unit vector components
    ijk_t = np.column_stack((i_t, j_t, k_t))

    # compensate for anchor
    points = points - anchor

    # stack coordinates vertically
    points = np.column_stack(points)

    # multiply transformation matrix with coordinates
    points = np.dot(ijk_t, points)

    # reset stacking to original shape
    points = np.column_stack(points)

    # add back the anchor
    points = points + anchor

    return points


def linear_transform_vertices(vertices: np.array,
                              it=np.array([1, 0, 0]),
                              jt=np.array([0, 1, 0]),
                              kt=np.array([0, 0, 1]),
                              anchor=np.array([0, 0, 0])
                              ):
    """Pass GL_LINES vertex position list directly."""
    # transformation matrix
    tm = np.stack([it, jt, kt])

    # form point matrix, stacking coordinates vertically
    points = (vertices.reshape(len(vertices)//3, 3) - anchor).T

    # apply transformation
    points = np.dot(tm, points)

    # reverse shape to fit GL_LINES vertex position list
    points = (points.T + anchor).flatten()

    return points


def normalize_rgba(r, g, b, a=255):
    """Convenience method for converting c4B values into normalized 0-1 values for opengl."""
    return r / 255, g / 255, b / 255, a / 255


class App(pyglet.window.Window):
    def __init__(self):
        super(App, self).__init__(WIDTH, HEIGHT, resizable=True, config=get_config())
        center_window(self)
        set_background_color(20, 30, 40, 255)
        self.batch = pyglet.graphics.Batch()
        self.zoom = 1.0
        self.program = pyglet.graphics.get_default_shader()
        self.vertex_list = None
        pyglet.gl.glLineWidth(2)

        self.line_points = []
        self.draw_screen_grid()

        # parameters for different linear_transformations
        self.theta = 0.1  # rotation angle
        self.ix, self.iy, self.jx, self.jy = 1, 0, 0, 1


    def draw_screen_grid(self, separation=3, color=(255, 255, 255, 255)):
        points = []
        colors = []
        for x in range(0, self.width, separation):
            start = x, 0, 0
            end = x, self.height, 0
            points.extend((*start, *end))
            colors.extend(normalize_rgba(*FRESH_AIR) * 2)

        for y in range(0, self.height, separation):
            start = 0, y, 0
            end = self.width, y, 0
            points.extend((*start, *end))
            colors.extend(normalize_rgba(*ORANGE_PEEL) * 2)

        self.line_points = np.array(points)
        self.vertex_list = self.program.vertex_list(len(self.line_points) // 3, pyglet.gl.GL_LINES, self.batch,
                                                    position=('f', points),
                                                    colors=('f', colors))

    def rotate_screen_grid(self, theta):
        points = np.array(self.line_points).reshape((len(self.line_points) // 3, 3))
        self.line_points = linear_transform(points=points,
                                            i_t=np.array([np.cos(theta), np.sin(theta), 0]),
                                            j_t=np.array([-np.sin(theta), np.cos(theta), 0]),
                                            anchor=np.array((self.width / 2, self.height / 2, 0)),
                                            ).flatten()

        self.vertex_list.pip_position[:] = self.line_points

    def rotate_screen_grid_vertices(self, theta):
        self.line_points = linear_transform_vertices(self.line_points,
                                                     it=np.array([np.cos(theta), -np.sin(theta), 0]),
                                                     jt=np.array([np.sin(theta), np.cos(theta), 0]),
                                                     anchor=np.array((self.width / 2, self.height / 2, 0)),
                                                     )
        self.vertex_list.pip_position[:] = self.line_points

    def shear_screen_grid(self, jx):
        points = np.array(self.line_points).reshape((len(self.line_points) // 3, 3))
        self.line_points = linear_transform(points,
                                            i_t=np.array([1, 0, 0]),
                                            j_t=np.array([jx, 1, 0]),
                                            anchor=np.array((self.width // 2, self.height // 2, 0)),
                                            ).flatten()

        self.vertex_list.pip_position[:] = self.line_points

    def test_screen_grid(self, ix, iy, jx, jy):
        points = np.array(self.line_points).reshape((len(self.line_points) // 3, 3))
        self.line_points = linear_transform(points,
                                            i_t=np.array([ix, iy, 0]),
                                            j_t=np.array([jx, jy, 0]),
                                            anchor=np.array((self.width // 2, self.height // 2, 0)),
                                            ).flatten()

        self.vertex_list.pip_position[:] = self.line_points

    def twist_screen_grid(self, angle):
        points = np.array(self.line_points).reshape((len(self.line_points) // 3, 3))
        self.line_points = np.array(twist(points, (self.width // 2, self.height // 2, 0), angle)).flatten()

        self.vertex_list.pip_position[:] = self.line_points

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def update(self):
        self.rotate_screen_grid_vertices(self.theta)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
        if symbol == pyglet.window.key.UP:
            self.rotate_screen_grid_vertices(np.radians(self.theta))
        if symbol == pyglet.window.key.DOWN:
            self.rotate_screen_grid(np.radians(-self.theta))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:

            # self.theta = np.radians(dy*10)
            self.jx = 0.01 * dy
            self.iy = 0.01 * dx
            self.test_screen_grid(self.ix, self.iy, self.jx, self.jy)
        elif buttons == pyglet.window.mouse.MIDDLE:
            self.view = self.view.translate((dx, dy, 0))
        elif buttons == pyglet.window.mouse.RIGHT:
            self.rotate_screen_grid_vertices(np.radians(dy*0.1))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.zoom = 1.0
        self.zoom += 0.03 * scroll_y
        self.view = self.view.scale((self.zoom, self.zoom, 1))


if __name__ == "__main__":
    app = App()
    pyglet.app.run()
