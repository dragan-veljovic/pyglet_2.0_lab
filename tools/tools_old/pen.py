import pyglet


class Pen:
    strokes = []

    def __init__(self, batch, group, size=4, color=(255, 255, 255, 255)):
        self.batch = batch
        self.group = group
        self.size = size
        self.color = color
        self.draw_points = []
        self.circles = []
        self.vertex_list = self.batch.add(256, pyglet.gl.GL_LINES, self.group, 'v2f', 'c4B')
        self.dpo = []  # draw points optimized
        self.strokes.append(self)


    def update(self, x, y):
        # using gl lines
        self.draw_points.append([x, y])
        self.circles.append(pyglet.shapes.Circle(x, y, self.size//2, color=self.color[:3], batch=self.batch, group=self.group))
        self.draw_stroke()

    def draw_stroke(self):
        vertices = []
        colors = []
        for i in range(len(self.draw_points) - 1):

            gl_line_points = *self.draw_points[i], *self.draw_points[i + 1]
            vertices.extend(gl_line_points)
            colors.extend(self.color * 2)

        self.vertex_list.resize(len(vertices)//2)
        self.vertex_list.vertices[:] = vertices
        self.vertex_list.colors[:] = colors

        # using circles
        # self.draw_points.append(pyglet.shapes.Circle(x, y, 3, color=self.color[:3], batch=self.batch, group=self.group))

    def optimize(self):
        self.dpo = self.draw_points
        points_to_remove = []
        for i in range(1, len(self.dpo) - 1):
            x_diff = abs(self.dpo[i - 1][0] - self.dpo[i][0])
            y_diff = abs(self.dpo[i - 1][1] - self.dpo[i][1])
            # if x_diff <= 2:
            #     self.dpo[i][0] = self.dpo[i - 1][0]
            #
            # if y_diff <= 2:
            #     self.dpo[i][1] = self.dpo[i - 1][1]

            if x_diff <= 1 and y_diff <= 1:
                points_to_remove.append(self.dpo[i])

        for point in points_to_remove:
            self.dpo.remove(point)

        self.draw_points = self.dpo
        self.draw_stroke()