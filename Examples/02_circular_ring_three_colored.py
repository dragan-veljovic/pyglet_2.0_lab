import pyglet
import numpy as np


class CircleRingThreeColored:
    """ Circular ring with thickness, with separate color parameters for
        inner, center and outer regions.
    """

    def __init__(self, center_x, center_y, radius, thickness: int = 2, segments: int = None,
                 inner_rgba=None, center_rgba=(255, 255, 255, 255), outer_rgba=None,
                 batch=None, group=None):
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

        # # GL_LINE_STRIP - can use points directly
        # self._points = self._points.tolist()
        # self._vertex_list = self._batch.add(len(self._points) // 2, pyglet.gl.GL_LINE_STRIP, self._group,
        #                                     ('v2f', self._points))

        # # GL_LINES - change points to _gl_line_vertices
        # self._lines_vertices = self._generate_lines_vertices()
        # self._vertex_list = self._batch.add(len(self._lines_vertices)//2, pyglet.gl.GL_LINES, self._group,
        #                                     ('v2f', self._lines_vertices))

        # GL_TRIANGLES - change points to _gl_triangle_vertices
        self._vertex_list = self._batch.add(self._num_vertices, pyglet.gl.GL_TRIANGLES, self._group,
                                            'v2f', 'c4B')
        self._generate_triangle_vertices()
        self.update_colors()

    def _generate_triangle_vertices(self):
        points = self._points
        vertices = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            delta_x = p2[0] - p1[0]
            delta_y = p2[1] - p1[1]
            length = (delta_x**2 + delta_y**2)**0.5
            # calculating normalized components of the normal to the segment
            nx, ny = -delta_y / length, delta_x / length
            dx, dy = nx * self._thickness, ny * self._thickness

            p3 = p2[0] + dx, p2[1] + dy
            p4 = p1[0] + dx, p1[1] + dy
            p5 = p2[0] - dx, p2[1] - dy
            p6 = p1[0] - dx, p1[1] - dy

            vertices.extend((*p1, *p2, *p3, *p3, *p4, *p1))
            vertices.extend((*p1, *p2, *p5, *p5, *p6, *p1))

        print("vertices: ", len(vertices))
        self._vertex_list.vertices[:] = vertices

    def _generate_points(self):
        thetas = np.linspace(0, 2 * np.pi, self._segments + 1)
        return np.column_stack(
            (self._x + self._radius * np.cos(thetas),
             self._y + self._radius * np.sin(thetas))
        )

    def update_colors(self):
        colors = (self._cen_rgba*2 + self._out_rgba*3 + self._cen_rgba*3 + self._inn_rgba*3 +
                  self._cen_rgba) * self._segments

        print("colors:", len(colors))
        print(colors)
        self._vertex_list.colors[:] = colors

    def _generate_lines_vertices(self):
        # pure python
        # vertices = []
        # for i in range(len(self._points) - 1):
        #     p1, p2 = self._points[i], self._points[i + 1]
        #     vertices.extend((p1[0], p1[1], p2[0], p2[1]))
        # return vertices

        # Numpy
        return np.column_stack((self._points[:-1], self._points[1:])).flatten()


class App(pyglet.window.Window):
    def __init__(self):
        super(App, self).__init__(1280, 720, "Test app")
        self.batch = pyglet.graphics.Batch()
        # self.circle = pyglet.shapes.Circle(500, 500, 100, batch=self.batch)
        self.ring = CircleRingThreeColored(self.width / 2, self.height / 2, 200, batch=self.batch,
                                           thickness=30,
                                           outer_rgba=(0, 0, 0, 0), inner_rgba=(0, 0, 0, 0))

        self.ring = CircleRingThreeColored(self.width / 2, self.height / 2, 200, batch=self.batch,
                                           thickness=30,
                                           outer_rgba=(255, 0, 0, 100), inner_rgba=(0, 0, 0, 0), center_rgba=(255, 255 ,255 ,255))
        #pyglet.gl.glClearColor(1, 1, 1, 1)

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()


if __name__ == "__main__":
    app = App()
    pyglet.app.run()
