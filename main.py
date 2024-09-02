import random

import pyglet
import pymunk
from tools.definitions import *
from tools.camera import Camera2D

WIDTH = 1280
HEIGHT = 720
FPS = 60


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

        # Create a static ground line
        ground = pymunk.Segment(self.space.static_body, (0, 100), (self.width, 100), 5)
        ground.friction = 1.0
        ground.elasticity = 1.0
        self.space.add(ground)

        # scene attributes
        self.release_bodies = False
        self.circles = []
        self.bodies = []

    def toggle_run(self):
        if self.run:
            pyglet.clock.unschedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1/FPS)
        self.run = not self.run

    def add_body(self):
        mass = 1
        radius = random.randrange(5, 15)
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (self.camera.get_mouse())
        body.velocity = (random.randrange(500), random.randrange(500))
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        shape.elasticity = 0.9
        color = random.randrange(255), random.randrange(255), random.randrange(255), 255
        self.space.add(body, shape)

        self.bodies.append(body)
        self.circles.append(pyglet.shapes.Circle(body.position.x, body.position.y, radius, color=color, batch=self.batch))

    def update(self, dt):
        self.space.step(dt)  # Step the Pymunk physics simulation

        if self.release_bodies:
            self.add_body()

        idx_to_remove = []
        for i in range(len(self.circles)):
            circle = self.circles[i]
            body = self.bodies[i]

            # clear up out-of-bounds bodies
            if body.position.y < -2000:
                idx_to_remove.append(i)
            else:
                circle.position = int(body.position.x), int(body.position.y)

        # following doesn't work, it seems circles are bodies are not matched after some time
        # a class should unify the body, shape and pyglet.circle into attributes of one instance
        for idx in idx_to_remove:
            self.circles.pop(idx)
            self.bodies.pop(idx)

    def on_draw(self) -> None:
        self.clear()
        with self.camera:
            self.batch.draw()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == pyglet.window.mouse.LEFT:
            self.release_bodies = True

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == pyglet.window.mouse.LEFT:
            self.release_bodies = False

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.SPACE:
            self.toggle_run()


if __name__ == "__main__":
    app = MyApp()
    pyglet.app.run()
