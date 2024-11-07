"""
PIP class for easy instantiation of picture-in-picture views.
TODO: Fix camera mouse problems, why on_resize() must be called to restore view in on_draw?
TODO: add getter setters in PIP class to control PIP view
TODO: calling two render commands in on_draw for each buffer is clumsy
"""

import pyglet.math
from tools.definitions import *
from tools.camera import *
from tools.graphics import *

settings = {
    'default_mode': False,
    'width': 1280,
    'height': 720,
    'fps': 60,
    'resizable': False,
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


class PIP:
    def __init__(
            self,
            window: Window,
            batch: Batch,  # scene to draw
            pip_position=Vec3(0, 0, 0),
            pip_width=200,
            pip_height=200,
            program: ShaderProgram = None,
            group: Group = None,
            cam_position=Vec3(0, 0, 500),
            cam_target=Vec3(0, 0, 0),
            cam_projection: Mat4 = None,
            color_att=True,
            depth_att=False,
            message: str = None
    ):
        """
        A picture-in-picture functionality using Framebuffers.
        Pass a batch and camera view parameters to render to 2D texture.
        Texture will be placed on the screen through passed position and size parameters.
        """
        self.window = weakref.proxy(window)
        self.batch = batch
        self.pip_batch = pyglet.graphics.Batch()
        self.program = program
        self.pip_program = self.get_shader_program()
        self.pip_position = pip_position
        self.width = pip_width
        self.height = pip_height
        self.message = message

        # framebuffer creation
        self.framebuffer = pyglet.image.buffer.Framebuffer()
        if color_att:
            self.texture = pyglet.image.Texture.create(self.window.width, self.window.height, min_filter=GL_LINEAR,
                                                       mag_filter=GL_LINEAR)
            self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        else:
            self.texture = None
        if depth_att:
            self.renderbuffer = pyglet.image.buffer.Renderbuffer(self.width, self.height, GL_DEPTH_COMPONENT)
            self.framebuffer.attach_renderbuffer(self.renderbuffer, attachment=GL_DEPTH_ATTACHMENT)
        else:
            self.renderbuffer = None

        self.group = group
        if not group and self.texture:
            self.group = TextureGroup(self.texture, self.pip_program)
        self.cam_position = cam_position
        self.cam_target = cam_target
        self.cam_projection = cam_projection or pyglet.math.Mat4.perspective_projection(
            self.window.width/self.window.height, 1, 5000, fov=50,
        )
        self.view = self.get_view()

        # a quad plane to attach texture to
        self.vertex_list = self.get_vertex_list()

    def render_buffer(self):
        # record current window view and projection
        current_view = self.window.view
        current_proj = self.window.projection

        # set assigned view and projection
        self.window.view = self.view
        self.window.projection = self.cam_projection

        # render scene from given view
        self.framebuffer.bind()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear buffer
        glClearColor(0.2, 0.0, 0.0, 1.0)
        self.batch.draw()
        self.framebuffer.unbind()
        set_background_color()

        # reset view
        self.window.view = current_view
        self.window.projection = current_proj

    def render_pip(self):
        # saving view
        current_view = self.window.view
        current_proj = self.window.projection

        # setting up 2D orthogonal projection
        self.window.projection = pyglet.math.Mat4.orthogonal_projection(0, self.window.width, 0, self.window.height, -1, 1)
        self.window.view = pyglet.math.Mat4()
        glViewport(int(self.pip_position.x), int(self.pip_position.y), int(self.width), int(self.height))

        self.rectangle = pyglet.shapes.Box(0, 0, self.window.width, self.window.height, 5)
        if self.message:
            self.label = pyglet.text.Label(self.message, 10, 10, font_size=40)
            self.label.draw()

        self.rectangle.draw()
        self.pip_batch.draw()

        # resetting view
        self.window.view = current_view
        self.window.proj = current_proj

    def get_vertex_list(self):
        # return vertex list of a textured quad
        w, h = self.window.width, self.window.height  # entire screen rendered
        position = (
            0, 0, 0, w, 0, 0, w, h, 0,
            0, 0, 0, w, h, 0, 0, h, 0,
        )
        tex_coords = (
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        )

        return self.pip_program.vertex_list(
            6, GL_TRIANGLES, self.pip_batch, self.group,
            position=('f', position),
            tex_coords=('f', tex_coords)
        )

    def get_view(self):
        return pyglet.math.Mat4.look_at(self.cam_position, self.cam_target, Vec3(0, 1, 0))

    def get_shader_program(self):
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

        return ShaderProgram(
            Shader(pip_vertex_source, 'vertex'),
            Shader(pip_fragment_source, 'fragment')
        )


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)
        self.camera_2D = Camera2D(self, mouse_controls=False, centered=False)
        self.camera.look_at(Vec3(-200, 500, 500), Vec3(0, 0, 0))
        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        # scene elements
        self.floor = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, rotation=(np.pi / 2, 0, 0),
            color=(150, 150, 150, 255)
        )
        self.back_wall = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, color=(150, 150, 150, 255)
        )
        self.cube = Cuboid(self.program, self.batch, color=(128, 0, 0, 255))

        glEnable(GL_DEPTH_TEST)

        # instantiating framebuffers
        self.framebuffer1 = PIP(self, self.batch,
                                Vec3(self.width - 500, self.height - self.height/3, 0),
                                pip_width=500, pip_height=self.height//3,
                                message="Framebuffer 1 view"
                                )

        self.framebuffer2 = PIP(self, self.batch,
                                Vec3(self.width - 500, self.height - 2*self.height/3),
                                pip_width=500, pip_height=self.height//3,
                                cam_position=Vec3(1000, 500, 1000),
                                message="Framebuffer 2 view")

        self.framebuffer3 = PIP(self, self.batch,
                                Vec3(self.width - 500, self.height - 3*self.height/3),
                                pip_width=500, pip_height=self.height//3,
                                cam_position=Vec3(0, 1000, 1),
                                message="Top View")

    def on_draw(self) -> None:
        # rendering to framebuffer texture
        self.framebuffer1.render_buffer()
        self.framebuffer2.render_buffer()
        self.framebuffer3.render_buffer()

        self.framebuffer1.cam_position.x += 1
        self.framebuffer1.view = self.framebuffer1.get_view()

        # redering in default buffer
        self.clear()
        self.camera.look_at(self.camera.position, self.camera.target)  # reseting view to default
        self.batch.draw()

        # rendering contents of framebuffer texture
        self.framebuffer1.render_pip()
        self.framebuffer2.render_pip()
        self.framebuffer3.render_pip()

        # must be added to restore view?
        self.camera.on_resize(self.height, self.width)

if __name__ == '__main__':
    start_app(App, settings)
