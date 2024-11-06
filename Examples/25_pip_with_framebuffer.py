"""
Picture in picture implementation using Framebuffer.
TODO: Fix camera problems
TODO: Clamp pip as GUI instead of existing 3D space
"""

import pyglet.math
from tools.definitions import *
from tools.camera import *
from tools.graphics import *

settings = {
    'width': 1280,
    'height': 720,
    'fps': 60
}

vertex_source = """
#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    vertex_colors = colors;
}
"""

fragment_source = """
#version 330 core
in vec4 vertex_colors;
out vec4 final_colors;

void main() {
    final_colors = vertex_colors;
}
"""

pip_vertex_source = """#version 330 core
    in vec3 position;
    in vec2 tex_coords;
    out vec2 texture_coords;
    in vec4 colors;

    out vec4 frag_colors;

    uniform WindowBlock 
    {                       // This UBO is defined on Window creation, and available
        mat4 projection;    // in all Shaders. You can modify these matrixes with the
        mat4 view;          // Window.view and Window.projection properties.
    } window;  


    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1.0);
        texture_coords = tex_coords;
        frag_colors = colors;
    }
"""

pip_fragment_source = """#version 330 core
    in vec2 texture_coords;
    in vec4 frag_colors;

    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        final_colors = frag_colors;
        final_colors = texture(our_texture, texture_coords);
    }
"""


class PointLight:
    def __init__(self, position: Vec3(0, 0, 0), color: (1.0, 1.0, 1.0), ambient=0.2, diffuse=1.0, specular=0.5):
        self.position = position
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

    def get_light_view(self):
        return pyglet.math.Mat4.look_at(self.position, Vec3(0, 0, 0), Vec3(0, 1, 0))


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)
        self.camera_2D = Camera2D(self, mouse_controls=False)
        self.camera.look_at(Vec3(-200, 500, 500), Vec3(0, 0, 0))
        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        self.light = PointLight(Vec3(0, 0, 1000), color=(255, 255, 255, 255))

        # scene elements
        self.floor = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, rotation=(np.pi / 2, 0, 0),
            color=(150, 150, 150, 255)
        )
        self.back_wall = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, color=(150, 150, 150, 255)
        )
        self.cube = Cuboid(self.program, self.batch, color=(128, 0, 0, 255))

        # pip will be rendered in a separate framebuffer
        self.framebuffer = pyglet.image.buffer.Framebuffer()
        self.texture = pyglet.image.Texture.create(self.width, self.height, min_filter=GL_NEAREST,
                                                   mag_filter=GL_NEAREST)
        self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        self.renderbuffer = pyglet.image.buffer.Renderbuffer(self.width, self.height, GL_DEPTH_COMPONENT)
        self.framebuffer.attach_renderbuffer(self.renderbuffer, attachment=GL_DEPTH_ATTACHMENT)

        self.pip_batch = pyglet.graphics.Batch()
        self.pip_program = ShaderProgram(
            Shader(pip_vertex_source, 'vertex'),
            Shader(pip_fragment_source, 'fragment')
        )
        self.texture_group = TextureGroup(self.texture, self.pip_program)
        self.quad = TexturedPlane((0, 0, 0), self.pip_batch, self.texture_group,
                                  self.pip_program, length=500, height=300)

        glEnable(GL_DEPTH_TEST)

    def on_draw(self) -> None:
        self.view = pyglet.math.Mat4.look_at(self.light.position, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.light.position.x += 2

        self.framebuffer.bind()
        glClearColor(0.2, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear buffer
        self.batch.draw()
        set_background_color()
        self.framebuffer.unbind()

        self.clear()
        self.camera.look_at(self.camera.position, self.camera.target)
        self.batch.draw()

        self.pip_batch.draw()


if __name__ == '__main__':
    start_app(App, settings)
