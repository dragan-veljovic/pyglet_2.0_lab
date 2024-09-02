import random
import pyglet
import os

WIDTH = 800
HEIGHT = 600
INCREMENT = 100

# Set the PYGLET_TEXT_LAYOUT environment variable to 'legacy'
os.environ['PYGLET_TEXT_LAYOUT'] = 'legacy'


class Grid:
    def __init__(self,
                 window_width: int,
                 window_height: int,
                 batch: pyglet.graphics.Batch,
                 spacing: int = 25,
                 line_thickness: int = None,
                 color=None
                 ) -> None:
        self.w = window_width + spacing
        self.h = window_height + spacing
        self.batch = batch
        self.spacing = spacing
        self.thickness = line_thickness if line_thickness else 1
        self.color = color if color else (255, 255, 255, 255)
        self.hlines = self.h // spacing
        self.vlines = self.w // spacing
        self.lines = []

        self.create_grid()

    def create_grid(self):
        for i in range(self.hlines):
            self.lines.append(pyglet.shapes.Line(0, self.spacing*i, self.w, self.spacing*i,
                                                 self.thickness, self.color, self.batch))
        for j in range(self.vlines):
            self.lines.append(pyglet.shapes.Line(self.spacing*j, 0, self.spacing*j, self.h,
                                                 self.thickness, self.color, self.batch))


class BouncingBall(pyglet.shapes.Circle):
    area_width = WIDTH
    area_height = HEIGHT

    def __init__(self, radius, segments, batch):
        super().__init__(self.position[0], self.position[1], radius, segments, batch=batch, color=(200, 220, 50, 255))
        self.vx, self.vy = random.randrange(-500, 500) / 100, random.randrange(-500, 500) / 100
        self.position = random.randrange(0 + radius, WIDTH - radius), random.randrange(0 + radius, HEIGHT - radius)

    def handle_collisions(self):
        """Checks collisions with the sides of given rectangular area.
            Allows for dynamic resize of the given area.
            Returns True in case of collision for statistics.
        """
        x, y, radius, width, height = self.x, self.y, self.radius, self.area_width, self.area_height
        if x - radius < 0:
            self.x = radius
            self.vx *= -1
            return True
        elif x + radius > width:
            self.x = width - radius
            self.vx *= -1
            return True

        if y - radius < 0:
            self.y = radius
            self.vy *= -1
            return True
        elif y + radius > height:
            self.y = height - radius
            self.vy *= -1
            return True

    def update_position(self):
        self.x += self.vx
        self.y += self.vy

    @classmethod
    def set_new_limits(cls, area_width: int, area_height: int):
        cls.area_width, cls.area_height = area_width, area_height


class ScheduledFunction:
    """Not used, introduces stutter. Use pyglet.clock.schedule_interval() instead."""
    def __init__(self,
                 func_to_call: callable = None,
                 interval_secs: int = 1):
        self.interval = interval_secs
        self.func = func_to_call
        self.clock = pyglet.clock.Clock()
        self.start = self.clock.time()
        self.timer = 0
        self.seconds = 0

    def run(self):
        self.timer += self.clock.update_time()
        if self.timer >= self.interval:
            self.timer = 0
            self.seconds += self.interval
            return self.func()

    def elapsed_time(self) -> int:
        return self.clock.time() - self.start


class TestApp(pyglet.window.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "My test app", resizable=True, vsync=True)
        self.scr = pyglet.canvas.Display().get_default_screen()
        self.set_location((self.scr.width - WIDTH) // 2, (self.scr.height - HEIGHT) // 2)
        pyglet.gl.glClearColor(0.1, 0.2, 0.3, 1)
        self.batch = pyglet.graphics.Batch()

        # scene elements
        self.grid = Grid(self.width, self.height, self.batch)
        self.balls = []
        self.no_balls = 0

        # measure collisions
        self.collisions = 0
        self.schedule_function(self.calculate_pressure)

    def calculate_pressure(self, dt):
        print(self.collisions)
        self.collisions = 0

    @staticmethod
    def schedule_function(func: callable, interval_sec: int = 1):
        return pyglet.clock.schedule_interval(func, interval_sec)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
        elif symbol == pyglet.window.key.P:
            pass

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            for _ in range(100):
                self.no_balls += INCREMENT
                self.balls.append(BouncingBall(5, 6, self.batch))

    def update(self):
        for ball in self.balls:
            if ball.handle_collisions():
                self.collisions += 1
            ball.update_position()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.grid = Grid(width, height, self.batch)
        BouncingBall.set_new_limits(width, height)

    def on_draw(self):
        self.update()
        self.clear()
        self.batch.draw()


if __name__ == "__main__":
    app = TestApp()
    pyglet.app.run(1/60)
