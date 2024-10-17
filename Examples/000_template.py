from tools.definitions import *
from tools.camera import Camera3D

SETTINGS = {
    # to overrides all other settings, and start in default mode?
    "default_mode": False,
    # alternatively specify a number of custom arguments
    "fps": 60,
    "width": 1920,
    "height": 1280,
    "resizable": True,
    "config": get_config(samples=4)
}


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)

    def on_draw(self) -> None:
        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()


if __name__ == '__main__':
    start_app(App, SETTINGS)
