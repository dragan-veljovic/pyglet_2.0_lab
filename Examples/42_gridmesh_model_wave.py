import pyglet

from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *
from pyglet.model import Model
from pyglet.graphics.vertexdomain import IndexedVertexList
from pyglet.model import *

setteings = {
    "default_mode": True,
    "width": 1280,
    "height": 720,
    "config": get_config()
}

vertex_source = """
#version 330 core
in vec3 position;
in vec4 color;
in vec2 tex_coord;

out vec3 frag_pos;
out vec4 frag_color;
out vec2 frag_coord;

uniform float time;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    float dist1 = length(vec3(250, 500, 0) - position);
    float dist2 = length(vec3(750, 500, 0) - position);
    float dist = dist1 + dist2;
    vec3 new_position = vec3(position.xy, position.z + 5*(sin(0.05*dist1*time-10*time)+sin(0.05*dist2*time-10*time)));
    gl_Position = window.projection * window.view * vec4(new_position, 1.0);
    frag_color = color;
    frag_coord = tex_coord;
    frag_pos = new_position;
}

"""


fragment_source = """
#version 330 core
in vec3 frag_pos;
in vec4 frag_color;
in vec2 frag_coord;

out vec4 final_color;

uniform float time;
uniform sampler2D diffuse_texture;

void main() {
    vec4 diff = texture(diffuse_texture, frag_coord);
    float dist = length(frag_pos - vec3(350, 350, 0));
    // cool shading effect
    final_color = vec4(vec3(0.5)*frag_pos.z*0.1, 1.0) + 0.5*diff;
}
"""


class GridMesh(Model):
    """Tessellated rectangular grid, split into triangles."""
    def __init__(self, width: float, height: float, columns=4, rows=3,
                 batch: Batch | None = None,
                 group: Group | None = None,
                 program: ShaderProgram | None = None,
                 position=Vec3(0, 0, 0),
                 ):
        self._width = width
        self._height = height
        self._pos = position
        self._rows = rows
        self._cols = columns
        self._batch = batch
        self._group = group
        self._program = program

        self._vertex_list = self._create_vertex_list()

        super().__init__([self._vertex_list], [group], batch)

    def _create_vertex_list(self) -> IndexedVertexList:
        cols, rows = self._cols, self._rows
        width, height = self._width, self._height
        pos = self._pos
        delta_x = width / (cols - 1)
        delta_y = height / (rows - 1)

        position = []
        indices = []
        tex_coord = []

        for j in range(rows):
            for i in range(cols):
                x = pos.x + i * delta_x
                y = pos.y + j * delta_y
                z = pos.z
                position.extend((x, y, z))

                u = i * delta_x / width
                v = j * delta_y / height
                tex_coord.extend((u, v))

                if i < cols - 1 and j < rows - 1:
                    # Quad points A(0,0), B(1,0), C(0,1), D(1,1) form ABC and BDC triangles
                    A = i + j * cols   # i + j*c
                    B = A + 1               # i + 1 + j*c
                    C = A + cols       # i + (j + 1)*c
                    D = C + 1               # i + 1 + (j + 1)*c
                    indices.extend((A, B, C, B, D, C))

        return self._program.vertex_list_indexed(
            len(position)//3,
            GL_TRIANGLES,
            indices,
            self._batch,
            self._group,
            position=('f', position),
            tex_coord=('f', tex_coord)
        )


class MyApp(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(MyApp, self).__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()
        set_background_color()
        self.camera = Camera3D(self)
        self.clock = pyglet.clock.Clock()
        self.start = self.clock.time()
        self.time = 0.0
        self.run = True

        self.wireframe = False

        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        self.tex_group = TextureGroup(
            pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            self.program
        )

        self.test = GridMesh(
            1000, 1000, 400, 400, batch=self.batch, program=self.program, group=self.tex_group
        )

    def on_draw(self):
        self.clear()
        if self.run:
            self.time = self.clock.time() - self.start
            self.program['time'] = self.time

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        super(MyApp, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.V:
            self.wireframe = not self.wireframe
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run

if __name__ == '__main__':
    app = start_app(MyApp, setteings)