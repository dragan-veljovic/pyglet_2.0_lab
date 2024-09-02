import pyglet
from pyglet.graphics import Group
from pyglet.math import Vec2


class CenteredCamera(Group):
    def __init__(self, window: pyglet.window.Window,
                 x=None, y=None, zoom: float = 1.0,
                 parent=None, zoom_min_max: tuple = None):
        super(CenteredCamera, self).__init__(parent)
        self.window = window
        self.x = x or window.width / 2
        self.y = y or window.height / 2
        self._zoom = zoom
        self._zoom_range = zoom_min_max
        self.parent = parent

    @property
    def position(self) -> Vec2:
        """Query the current offset."""
        return Vec2(self.x, self.y)

    @position.setter
    def position(self, new_position: Vec2):
        """Set the scroll offset directly."""
        self.x, self.y = new_position

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        if self._zoom_range:
            if self._zoom_range[0] <= value <= self._zoom_range[1]:
                self._zoom = value
        else:
            self._zoom = value

    def set_state(self):
        # translation (pan)
        pyglet.gl.glTranslatef(-self.x * self._zoom + self.window.width//2,
                               -self.y * self._zoom + self.window.height//2, 0)
        # scaling (zoom)
        pyglet.gl.glScalef(self._zoom, self._zoom, 1)

    # above method changes the matrix, and these changes should be reversed back
    # otherwise it'll keep multiplying offsets on every draw update pushing away from the screen
    def unset_state(self):  # signs, ratios and order is reversed here
        # un-scale
        pyglet.gl.glScalef(1/self._zoom, 1/self._zoom, 1)
        # un-translate
        pyglet.gl.glTranslatef(self.x * self._zoom - self.window.width//2,
                               self.y * self._zoom - self.window.height//2, 0)


class Camera(Group):
    """ Graphics group emulating the behaviour of a camera in 2D space. """

    def __init__(
        self,
        x: float, y: float,
        zoom: float = 1.0,
        parent=None
    ):
        super().__init__(parent)
        self.x, self.y = x, y
        self.zoom = zoom

    @property
    def position(self) -> Vec2:
        """Query the current offset."""
        return Vec2(self.x, self.y)

    @position.setter
    def position(self, new_position: Vec2):
        """Set the scroll offset directly."""
        self.x, self.y = new_position

    def set_state(self):
        """ Apply zoom and camera offset to view matrix. """
        pyglet.gl.glTranslatef(
            -self.x * self.zoom,
            -self.y * self.zoom,
            0
        )

        # Scale with zoom
        pyglet.gl.glScalef(self.zoom, self.zoom, 1)

    def unset_state(self):
        """ Revert zoom and camera offset from view matrix. """
        # Since this is a matrix, you will need to reverse the translate after rendering otherwise
        # it will multiply the current offset every draw update pushing it further and further away.

        # Use inverse zoom to reverse zoom
        pyglet.gl.glScalef(1 / self.zoom, 1 / self.zoom, 1)
        # Reverse the translation
        pyglet.gl.glTranslatef(
            self.x * self.zoom,
            self.y * self.zoom,
            0
        )