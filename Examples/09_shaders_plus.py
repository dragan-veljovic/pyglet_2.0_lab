import pyglet.shapes
from pyglet.graphics.shader import Shader, ShaderProgram

import tools_old.camera
from tools_old.definitions import *

WIDTH = 1280
HEIGHT = 720
FPS = 60


def get_standard_shader_program() -> ShaderProgram:
    vertex_shader = pyglet.resource.shader("shaders/vertex_shader.vert")
    fragment_shader = pyglet.resource.shader("shaders/fragment_shader.frag")
    shader_program = pyglet.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)
    return shader_program


def get_shader_program(*shader_files: str, path="shaders/") -> ShaderProgram:
    shaders = [pyglet.resource.shader(path+file) for file in shader_files]
    return ShaderProgram(*shaders)


def get_test_rectangle(x, y, width, height, program: pyglet.graphics.shader.ShaderProgram, batch=None, group=None):
    # BL, BR, TR, TL are vertices 0, 1, 2, 3 respectively
    x0, y0, x1, y1 = x - width/2, y - height/2, x + width/2, y - height/2
    x2, y2, x3, y3 = x1, y + height/2, x0, y + height/2
    return program.vertex_list_indexed(4, pyglet.gl.GL_TRIANGLES,
                                       [0, 1, 2, 0, 2, 3], batch=batch, group=group,
                                       position=('f', (x0, y0, 0, x1, y1, 0, x2, y2, 0, x3, y3, 0)),
                                       colors=('f', (1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1))
                                       )


def get_test_circle():
    pass


class App(pyglet.window.Window):
    def __init__(self):
        super(App, self).__init__(WIDTH, HEIGHT, resizable=True)
        center_window(self)
        set_background_color(20, 30, 40)
        self.program = get_shader_program('conditional.vert', 'per_pixel.frag')
        self.default_program = get_shader_program('default.vert', 'default.frag')
        self.batch = pyglet.graphics.Batch()
        self.gui_batch = pyglet.graphics.Batch()

        # timers
        self.dt = 1/FPS
        self.time = 0.0
        self.run = True
        self.timer = pyglet.text.Label("0.00", font_size=26, batch=self.gui_batch, x=30, y=self.height - 30)

        # scene elements
        n = 30
        self.back_rect = get_test_rectangle(self.width//2, self.height//2, self.width, self.height, self.default_program, batch=self.batch)
        self.rectangles = [get_test_rectangle(self.width/2 + i*51, self.height/2 + j*50, 30, 30, self.program, self.batch) for i in range(n) for j in range(n)]
        self.circle = pyglet.shapes.Circle(0, 0, 100, batch=self.batch)

    def on_draw(self):
        if self.run:
            self.time += self.dt
            self.timer.text = f"{self.time:0.2f}"
            self.program['time'] = self.time
        self.clear()
        self.batch.draw()
        self.gui_batch.draw()

    def on_key_press(self, symbol, modifiers):
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:
            self.view = self.view.translate((dx, dy, 0))
        if buttons == pyglet.window.mouse.RIGHT:
            self.view = self.view.scale((1 + 0.01*dy, 1 + 0.01*dy, 1))


def main():
    App()
    pyglet.app.run(1/FPS)


if __name__ == "__main__":
    main()
