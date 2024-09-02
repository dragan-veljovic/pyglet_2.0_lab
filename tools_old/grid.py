import pyglet
from tools_old.defaults import *


class Grid:
    def __init__(self, window_width: int, window_height: int, batch: pyglet.graphics.Batch,
                 group: pyglet.graphics.Group = None, width: int = None, height: int = None,
                 main_grid=True, fine_grid=True,
                 labels=True, label_color=(255, 255, 255, 255), label_stripe_color=(255, 255, 255),
                 main_div: int = 100, main_color=(92, 92, 92),
                 fine_div: int = 20, fine_color=(48, 48, 48),
                 grid_offset=False):
        """
        Create a graph-paper style grid, with optional main and fine divisions.
        Made to work with CenteredCamera class, so window width and height required for correction.
        Entire grid size is centered on whatever is the camera center.
        Update labels with pan_label_positions() after pan, but draw_labels() after zoom button is released.

        :param window_width: int, pyglet window width
        :param window_height: int, pyglet window height
        :param batch: pyglet.graphics.Batch, pyglet window batch
        :param group: pyglet.graphics.Group, Optional, whether attach grid to camera group
        :param width: int, Optional, width of entire grid (double window_width used if None)
        :param height: int, Optional, height of entire grid (double window_height used if None)
        :param main_grid: bool, Optional, whether to draw main grid lines
        :param fine_grid: bool, Optional, whether to draw fine grid lines
        :param labels: bool, Optional, whether to include labels (can be off for performance gain)
        :param grid_offset: bool, Optional, offset lines by half fine_div in both directions
                in order for snap to happen at fine square center instead of vertices
        :param main_div: int, Optional, specify division of main grid in pixels
        :param main_color: tuple, Optional, specify color of main grid lines ex: '(255, 255, 255)'
        :param fine_div: int, Optional, specify division of fine grid in pixels
        :param fine_color: tuple, Optional, specify color of fine grid lines ex: '(255, 255, 255)'
        """
        self.window_width, self.window_height = window_width, window_height
        self.batch = batch
        self.group = group
        self.main_grid, self.fine_grid = main_grid, fine_grid
        self.labels = labels
        self.label_color = label_color
        self.label_stripe_color = label_stripe_color
        self.height = height if height else round(self.window_height, -3) * 2
        self.width = width if width else round(self.window_width, -3) * 2
        self.mdiv, self.fdiv = main_div, fine_div
        self.offset = self.fdiv // 2 if grid_offset else 0
        self.mcolor, self.fcolor = main_color, fine_color
        # reserving memory for vertex lists
        self.main_vertex_list = pyglet.graphics.vertex_list(128, 'v2f', 'c3B')
        self.fine_vertex_list = pyglet.graphics.vertex_list(256, 'v2f', 'c3B')
        # number of divisions
        self.hfdivs = self.width // self.fdiv
        self.vfdivs = self.height // self.fdiv
        self.hmdivs = self.width // self.mdiv
        self.vmdivs = self.height // self.mdiv
        # labels for main grid
        self.label_size = 12.0
        self.label_font = pyglet.font.load("Segoe UI", size=18.0)
        self.hlabel_points = []
        self.hlabels = []
        self.vlabel_points = []
        self.vlabels = []
        self.hlabel_line = pyglet.shapes.Rectangle(5, 5, self.width, 20, batch=self.batch,
                                                   color=self.label_stripe_color)
        self.hlabel_line.opacity = 20
        self.vlabel_line = pyglet.shapes.Rectangle(5, 5, 40, self.height, batch=self.batch,
                                                   color=self.label_stripe_color)
        self.vlabel_line.opacity = 20

        self.draw_grid()

    def draw_grid(self):
        if self.fine_grid:
            self.draw_fine_grid()
        if self.main_grid:
            self.draw_main_grid()
        if self.labels:
            self.draw_labels()

    # following is the manual way of inserting vertices for all lines in the grid into batch
    def draw_fine_grid(self):
        # fine grid lines
        points = []
        colors = []
        # horizontal lines
        x = [-self.width // 4, self.width // 4 * 3]
        y = [j * self.fdiv for j in range(-self.vfdivs // 4, self.vfdivs // 4 * 3)]
        for j in range(self.vfdivs):
            for i in range(2):
                points.append(x[i])
                points.append((y[j] - self.offset))
                colors.extend(self.fcolor)
        # vertical lines
        x = [i * self.fdiv for i in range(-self.hfdivs // 4, self.hfdivs // 4 * 3)]
        y = [-self.height // 4, self.height // 4 * 3]
        for i in range(self.hfdivs):
            for j in range(2):
                points.append((x[i] - self.offset))
                points.append(y[j])
                colors.extend(self.fcolor)

        # add vertices to batch
        self.fine_vertex_list = self.batch.add(len(points) // 2,
                                               pyglet.gl.GL_LINES,
                                               self.group,
                                               ('v2i', points),
                                               ('c3B', colors))

    def draw_main_grid(self):
        points = []
        colors = []
        # horizontal lines
        x = [-self.width // 4, self.width // 4 * 3]
        y = [j * self.mdiv for j in range(-self.vmdivs // 4, self.vmdivs // 4 * 3)]
        self.vlabel_points = [[5, yy] for yy in y]
        for j in range(self.vmdivs):
            for i in range(2):
                points.append(x[i])
                points.append((y[j] - self.offset))
                colors.extend(self.mcolor)

        # add last horizontal line
        points.append(-self.width // 4)
        points.append(self.height // 4 * 3)
        colors.extend(self.mcolor)
        points.append(self.width // 4 * 3)
        points.append(self.height // 4 * 3)
        colors.extend(self.mcolor)

        # vertical lines
        x = [i * self.mdiv for i in range(-self.hmdivs // 4, self.hmdivs // 4 * 3)]
        y = [-self.height // 4, self.height // 4 * 3]
        self.hlabel_points = [[xx, 10] for xx in x]
        for i in range(self.hmdivs):
            for j in range(2):
                points.append((x[i] - self.offset))
                points.append(y[j])
                colors.extend(self.mcolor)

        # add last vertical line
        points.append(self.width // 4 * 3)
        points.append(-self.height // 4)
        colors.extend(self.mcolor)
        points.append(self.width // 4 * 3)
        points.append(self.height // 4 * 3)
        colors.extend(self.mcolor)

        # add vertices to batch
        self.main_vertex_list = self.batch.add(len(points) // 2,
                                               pyglet.gl.GL_LINES,
                                               self.group,
                                               ('v2i', points),
                                               ('c3B', colors))

    # slow re-draw, use on button release
    def draw_labels(self, zoom_level=1.0, camera_x=0, camera_y=0):
        delta_corr_x = self.window_width // 2 - camera_x * zoom_level - self.label_size // 2 * 3 / 1.25
        delta_corr_y = self.window_height // 2 - camera_y * zoom_level - self.label_size // 2
        delta = self.mdiv * zoom_level
        # clear existing labels from batch and list
        if self.vlabels:
            for vlabel in self.vlabels:
                vlabel.delete()
            self.vlabels = []

        if self.hlabels:
            for hlabel in self.hlabels:
                hlabel.delete()
            self.hlabels = []

        for j in range(-self.hmdivs // 4, self.hmdivs // 4 * 3):
            self.hlabels.append(pyglet.text.Label(f"{j * self.mdiv}", font_name="Segoe UI", font_size=self.label_size,
                                                  x=j * delta + delta_corr_x, y=10, batch=self.batch,
                                                  color=self.label_color))
        for i in range(-self.vmdivs // 4, self.vmdivs // 4 * 3):
            self.vlabels.append(pyglet.text.Label(f"{i * self.mdiv}", font_name="Segoe UI", font_size=self.label_size,
                                                  x=0, y=i * delta + delta_corr_y, width=40, align='right',
                                                  batch=self.batch, color=self.label_color))

    # fast position update, can be used realtime
    def pan_label_positions(self, dx, dy):
        for hlabel in self.hlabels:
            hlabel.x += dx
        for vlabel in self.vlabels:
            vlabel.y += dy


class NewGrid:
    """Attempt to create a dynamically expanding grid."""

    def __init__(
            self,
            window: pyglet.window.Window,
            camera: pyglet.graphics.Group,
            batch: pyglet.graphics.Batch,
            major_spacing=100,
            minor_spacing=20,
    ):
        # attributes from parameters
        self._window = window
        self._camera = camera
        self._batch = batch
        self._major = major_spacing
        self._minor = minor_spacing

        # operating attributes
        self._vertex_list = None
        self._vertex_count = 0
        self._hline_vertices = []
        self._vline_vertices = []
        self._vertices = []
        self._pan_dx = 0
        self._pan_dy = 0

        self.create_vertical_lines()
        self.create_horizontal_lines()

    def create_vertical_lines(self):
        width, height = self._window.width, self._window.height
        vertical_lines = width // self._major
        for i in range(vertical_lines):
            self._vline_vertices += i * self._major, 0, i * self._major, height
        self._vertex_list = self._batch.add(
            len(self.vertices) // 2,
            pyglet.gl.GL_LINES,
            self._camera,
            ('v2i', self.vertices)
        )

    @property
    def vertices(self) -> list:
        self._vertices[:] = self._hline_vertices + self._vline_vertices
        return self._vertices

    def create_horizontal_lines(self):
        width, height = self._window.width, self._window.height
        horizontal_lines = height // self._major
        for j in range(horizontal_lines):
            self._hline_vertices += 0, j*self._major, width, j*self._major
        self._vertex_list = self._batch.add(
            len(self.vertices) // 2,
            pyglet.gl.GL_LINES,
            self._camera,
            ('v2i', self.vertices)
        )

    def pan_grid(self, dx, dy):
        self._pan_dx += dx
        self._pan_dy += dy

        if self._pan_dx <= -self._major:
            self.add_vertical_right()
            self._pan_dx = 0
        elif self._pan_dx >= self._major:
            self.add_vertical_left()
            self._pan_dx = 0

        if self._pan_dy <= -self._major:
            self.add_horizontal_up()
            self._pan_dy = 0
        elif self._pan_dy >= self._major:
            self.add_horizontal_down()
            self._pan_dy = 0

    def add_horizontal_down(self):
        old_y = self._hline_vertices[1]
        new_y = old_y - self._major
        new_vertices = [0, new_y, self._window.width, new_y]
        self._hline_vertices = new_vertices + self._hline_vertices[:-4]
        self._vertex_list.vertices[:] = self.vertices

    def add_horizontal_up(self):
        old_y = self._hline_vertices[-1]
        new_y = old_y + self._major
        new_vertices = [0, new_y, self._window.width, new_y]
        self._hline_vertices = self._hline_vertices[4:] + new_vertices
        self._vertex_list.vertices[:] = self.vertices

    def add_vertical_right(self):
        old_x = self._vline_vertices[-2]
        new_x = old_x + self._major
        self._vline_vertices += new_x, 0, new_x, self._window.height
        self._vline_vertices = self._vline_vertices[4:]
        self._vertex_list.vertices[:] = self.vertices

    def add_vertical_left(self):
        old_x = self._vline_vertices[0]
        new_x = old_x - self._major
        new_vertices = [new_x, 0, new_x, self._window.height]
        self._vline_vertices = new_vertices + self._vline_vertices[:-4]
        self._vertex_list.vertices[:] = self.vertices

    def pan_label_positions(self, *args):
        pass

    def draw_labels(self, *args):
        pass


class Snap:
    # could be made to work without prior grid
    def __init__(self, grid: Grid, batch: pyglet.graphics.Batch, group=None, mode: str = 'off'):
        self.grid = grid
        self.batch = batch
        self.group = group
        self.mode = mode
        self.snap_amount = self.grid.mdiv if self.mode == 'main-grid' else self.grid.fdiv if self.mode == 'fine-grid' else 0
        self.mode_counter = 0
        self.marker_size = 20
        self.snap_marker = pyglet.shapes.Rectangle(0, 0, self.marker_size, self.marker_size, palette['yellow'][:3],
                                                   batch=self.batch, group=self.group)
        self.snap_marker.anchor_x, self.snap_marker.anchor_y = self.marker_size // 2, self.marker_size // 2
        self.snap_marker.opacity = 100
        self.snap_marker.visible = False

    def snap_round(self, x):
        return round(x / self.snap_amount) * self.snap_amount

    def update_snap_marker(self, mx, my):
        self.snap_marker.position = self.snap_round(mx), self.snap_round(my)

    def toggle_snap_mode(self):
        if self.mode_counter >= 2:
            self.mode_counter = 0
        else:
            self.mode_counter += 1

        if self.mode_counter == 0:
            self.mode = 'off'
            self.snap_marker.visible = False
        elif self.mode_counter == 2:
            self.mode = 'main-grid'
            self.snap_amount = self.grid.mdiv
            self.snap_marker.visible = True
        elif self.mode_counter == 1:
            self.mode = 'fine-grid'
            self.snap_amount = self.grid.fdiv // 2
            self.snap_marker.visible = True
