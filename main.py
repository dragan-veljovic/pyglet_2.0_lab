import random

import pyglet
import pymunk
from tools.definitions import *
from tools.camera import Camera2D
from tools.physics import CircleBody

WIDTH = 1280
HEIGHT = 720
FPS = 60
MAX_VEL = 300


class MyApp(pyglet.window.Window):
    def __init__(self):
        super(MyApp, self).__init__(height=HEIGHT, width=WIDTH, resizable=True, config=get_config())
        center_window(self)
        set_background_color()
        self.camera = Camera2D(self)
        self.batch = pyglet.graphics.Batch()

        # run toggle
        self.run = False
        self.toggle_run()

        # Create a Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)  # Gravity pointing downward

        # # Create a static ground line
        # ground = pymunk.Segment(self.space.static_body, (0, 100), (self.width, 100), 5)
        # ground.friction = 1.0
        # ground.elasticity = 1.0
        # self.space.add(ground)

        # create a static kinematic line
        self.platform = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        shape = pymunk.Segment(self.platform, (0, 100), (self.width, 100), 50)
        self.space.add(self.platform, shape)

        # scene attributes
        self.release_bodies = False
        self.bodies = []
        self.kinematic_body = None

    def toggle_run(self):
        if self.run:
            pyglet.clock.unschedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1/FPS)
        self.run = not self.run

    def add_body(self):
        radius = random.randint(3, 10)
        color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255
        velocity = random.randint(-MAX_VEL, MAX_VEL), random.randint(-MAX_VEL, MAX_VEL)
        body = CircleBody(
            self.space, self.batch, position=self.camera.get_mouse(), radius=radius, color=color, velocity=velocity
        )
        self.bodies.append(body)

    def update(self, dt):
        self.space.step(dt)  # Step the Pymunk physics simulation

        if self.release_bodies:
            self.add_body()

        if self.kinematic_body:
            self.kinematic_body.update()

        for body in self.bodies[:]:
            if body.position.y < -2000:
                body.delete()
                self.bodies.remove(body)
            else:
                body.update()

    def on_draw(self) -> None:
        self.clear()
        with self.camera:
            self.batch.draw()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == pyglet.window.mouse.LEFT:
            self.release_bodies = True
        if button == pyglet.window.mouse.RIGHT:
            if not self.kinematic_body:
                self.kinematic_body = CircleBody(
                    self.space, self.batch, radius=15, mass=150, position=self.camera.get_mouse(),
                    body_type=pymunk.Body.KINEMATIC
                )
            else:
                self.kinematic_body.position = self.camera.get_mouse()

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == pyglet.window.mouse.LEFT:
            self.release_bodies = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if buttons == pyglet.window.mouse.RIGHT:
            if modifiers & pyglet.window.key.MOD_CTRL:
                self.platform.angle += dy * 0.005
            if self.kinematic_body:
                self.kinematic_body.position = self.camera.get_mouse()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(MyApp, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.toggle_run()

        if symbol == pyglet.window.key.R:
            self.platform.angle += -0.01


if __name__ == "__main__":
    app = MyApp()
    pyglet.app.run()
