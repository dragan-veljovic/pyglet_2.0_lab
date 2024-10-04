import random

import tools.camera
from tools.definitions import *

# start app in default display mode
DEFAULT_DISPLAY_MODE = True
# or use these settings
WIDTH = 1280
HEIGHT = 720
FULLSCREEN = False
RESIZEABLE = True
FPS = 60


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.gui_batch = pyglet.graphics.Batch()
        self.program = get_default_shader_program()
        self.camera = tools.camera.Camera3D(self)
        self.clock = pyglet.clock.Clock()
        self.time = 0.0

        # scene
        self.graphs = []

        self.time_label = pyglet.text.Label(str(self.time), 50, 50, batch=self.gui_batch, font_size=20)

        pyglet.gl.glLineWidth(3)

    def generate_graph(self):
        vertices = []
        mx, my = random.randint(0,1000), random.randint(0,1000)
        for n in range(-500, 500):
            vertices.extend([n + mx, my, 0])

        self.graphs.append(self.program.vertex_list(
            count=len(vertices)//3, mode=pyglet.gl.GL_LINE_STRIP, batch=self.batch,
            position=('f', vertices),
            colors=('Bn', (255, 0, 0, 255)*(len(vertices)//3))
            )
        )

    def on_draw(self) -> None:
        self.clear()
        self.time += self.clock.update_time()
        self.time_label.text = "Time: " + str(round(self.time, 2))
        self.program['time'] = self.time

        self.batch.draw()

        self.gui_batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.time = 0.0

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == pyglet.window.mouse.LEFT:
            self.generate_graph()


if __name__ == "__main__":
    if DEFAULT_DISPLAY_MODE:
        start_in_default_display_mode(App, config=get_config(), fullscreen=True)
    else:
        App(width=WIDTH, height=HEIGHT, resizable=RESIZEABLE, fullscreen=FULLSCREEN)
        pyglet.app.run(1/FPS)
