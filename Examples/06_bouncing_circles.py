import random

import pyglet
""" Weird problem of sprites working too slow 300 compared to 1500!!!"""
import tools.camera
from tools.definitions import *
from tools.color import *

WIDTH = 1270
HEIGHT = 720
FPS = 60


class CustomSprite(pyglet.sprite.Sprite):
    # pyglet.resource.path = ["assets/ball.png"]
    # pyglet.resource.reindex()
    # ball_image = pyglet.resource.image('assets/ball.png')

    def __init__(self, batch):
        radius = 16
        x = random.randrange(0 + radius, WIDTH - radius)
        y = random.randrange(0 + radius, HEIGHT - radius)
        image = pyglet.image.load("../assets/ball.png")
        super().__init__(image, x=x, y=y, batch=batch)
        self.x_pos = x
        self.y_pos = y
        self.radius = self.width/2
        self.vx = random.randrange(-50, 50) * 0.1
        self.vy = random.randrange(-50, 50) * 0.1

    def update_motion(self):
        if 0 - self.radius > self.x or self.x > WIDTH - self.radius:
            self.vx *= -1
        if 0 - self.radius > self.y or self.y > HEIGHT - self.radius:
            self.vy *= - 1

        # doing it with self.x then self.y would waste resources
        # self.x_pos += self.vx
        # self.y_pos += self.vy
        #
        # self.position = self.x_pos, self.y_pos
        self.x += self.vx
        self.y += self.vy


class CustomCircle(pyglet.shapes.Circle):
    def __init__(self, batch, color=(0, 0, 0)):
        radius = 5
        x = random.randrange(0 + radius, WIDTH - radius)
        y = random.randrange(0 + radius, HEIGHT - radius)
        super().__init__(x, y, radius, batch=batch, color=color)
        self.x_pos = x
        self.y_pos = y
        self.vx = random.randrange(-50, 50) * 0.1
        self.vy = random.randrange(-50, 50) * 0.1

    def update_motion(self):
        if 0 - self.radius > self.x or self.x > WIDTH - self.radius:
            self.vx *= -1
        if 0 - self.radius > self.y or self.y > HEIGHT - self.radius:
            self.vy *= - 1

        # doing it with self.x then self.y would waste resources
        self.x_pos += self.vx
        self.y_pos += self.vy

        self.position = self.x_pos, self.y_pos


class App(pyglet.window.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Bouncing circles")
        center_window(self)
        set_background_color(*AIR_SUPERIORITY_BLUE)
        self.batch = pyglet.graphics.Batch()
        self.group = tools_old.camera.CenteredCamera(self, 0, 0)
        self.count = pyglet.text.Label("Hello", font_size=20, x=WIDTH-100, y=HEIGHT-100, color=(*DARK_BROWN, 255), batch=self.batch)
        self.label = pyglet.text.Label("LMB add circles\n RMB add sprites", font_size=20, x=WIDTH-200, y=HEIGHT-130, batch=self.batch, multiline=True, width=300, color=(*BLACK, 255))
        self.circles = []

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.circles.clear()

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            for i in range(100):
                self.circles.append(CustomCircle(self.batch))
            self.count.text = str(len(self.circles))
        elif button == pyglet.window.mouse.RIGHT:
            for i in range(100):
                self.circles.append(CustomSprite(self.batch))
            self.count.text = str(len(self.circles))

    def update(self, dt):
        for circle in self.circles:
            circle.update_motion()

    def on_draw(self):
        self.clear()
        self.batch.draw()


if __name__ == "__main__":
    app = App()
    pyglet.clock.schedule_interval(app.update, 1/FPS)
    pyglet.app.run()