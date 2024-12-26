"""
PIP class for easy instantiation of picture-in-picture views.
"""
import copy

import pyglet.math
from tools.definitions import *
from tools.camera import *
from tools.graphics import *

settings = {
    'default_mode': True,
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
            batch: Batch,
            pip_position=Vec3(0, 0, 0),
            pip_width=300,
            pip_height=200,
            program: ShaderProgram = None,
            group: Group = None,
            cam_position=Vec3(0, 0, 500),
            cam_target=Vec3(0, 0, 0),
            cam_projection: Mat4 = None,
            color_att=True,
            depth_att=False,
            message: str = None,
            border: bool = True
    ):
        """
        A picture-in-picture functionality using Framebuffers.
        Pass a batch and camera view parameters to render to 2D texture.
        Call Framebuffer.render() in your on_draw() after drawing your main scene batch.
        Texture will be placed on the screen through passed position and size parameters.
        """
        self.window = weakref.proxy(window)
        # batch and program of scene to render
        self.batch = batch
        self.program = program
        # batch and program for pip texture drawing
        self.pip_batch = pyglet.graphics.Batch()
        self.pip_program = self._get_shader_program()

        self.pip_position = pip_position
        self.width = pip_width
        self.height = pip_height
        self.message = message
        self.border = border

        # framebuffer creation
        self.framebuffer = pyglet.image.buffer.Framebuffer()
        if color_att:
            self.texture = pyglet.image.Texture.create(
                self.window.width, self.window.height, min_filter=GL_LINEAR, mag_filter=GL_LINEAR
            )
            self.framebuffer.attach_texture(self.texture, attachment=GL_COLOR_ATTACHMENT0)
        else:
            self.texture = None
        if depth_att:
            self.renderbuffer = pyglet.image.buffer.Renderbuffer(self.window.width, self.window.height, GL_DEPTH_COMPONENT)
            self.framebuffer.attach_renderbuffer(self.renderbuffer, attachment=GL_DEPTH_ATTACHMENT)
        else:
            self.renderbuffer = None

        # enable texturing and bind texture
        self.group = group
        if not group and self.texture:
            self.group = TextureGroup(self.texture, self.pip_program)

        # pip camera view
        self._cam_position = cam_position
        self._cam_target = cam_target
        self._cam_projection = cam_projection or pyglet.math.Mat4.perspective_projection(
            self.window.width/self.window.height, 1, 5000, fov=50,
        )
        self._view = self._get_view()
        self._current_view = self.window.view
        self._current_proj = self.window.projection

        # a quad plane to attach texture to
        self.vertex_list = self._get_vertex_list()

        # pip border and label
        self.rectangle = pyglet.shapes.Box(0, 0, self.window.width, self.window.height, 5)
        self.label = pyglet.text.Label(self.message, 10, 10, font_size=40) if self.message else None

    def render(self):
        self._render_buffer()
        self._render_pip()

    def _save_window_view(self):
        self._current_view = copy.deepcopy(self.window.view)
        self._current_proj = copy.deepcopy(self.window.projection)

    def _reset_window_view(self):
        self.window.view = self._current_view
        self.window.projection = self._current_proj

    def _render_buffer(self):
        """Render passed batch from given view angle into texture."""
        # record current window view and projection
        self._save_window_view()

        # set assigned view and projection
        self.window.view = self._view
        self.window.projection = self._cam_projection

        # render scene from given view
        self.framebuffer.bind()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear buffer
        self.batch.draw()
        self.framebuffer.unbind()

        # reset view
        self._reset_window_view()

    def _render_pip(self):
        """Render contents of the """
        # saving view
        self._save_window_view()

        # setting up 2D orthogonal projection
        self.window.projection = pyglet.math.Mat4.orthogonal_projection(0, self.window.width, 0, self.window.height, -1, 1)
        self.window.view = pyglet.math.Mat4()
        glViewport(int(self.pip_position.x), int(self.pip_position.y), int(self.width), int(self.height))

        if self.message:
            self.label.draw()
        if self.border:
            self.rectangle.draw()
        self.pip_batch.draw()

        # resetting view
        self._reset_window_view()

        # reset Viewport
        pyglet.gl.glViewport(0, 0, *self.window.get_framebuffer_size())

    def _get_vertex_list(self):
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

    def _get_view(self):
        return pyglet.math.Mat4.look_at(self._cam_position, self._cam_target, Vec3(0, 1, 0))

    def _get_shader_program(self):
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

    @property
    def cam_position(self):
        return self._cam_position

    @cam_position.setter
    def cam_position(self, position: Vec3):
        self._cam_position = position
        self._view = self._get_view()

    @property
    def cam_target(self):
        return self._cam_target

    @cam_target.setter
    def cam_target(self, target: Vec3):
        self._cam_target = target
        self._view = self._get_view()


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()

        self.camera = Camera3D(self)
        self.camera.look_at(Vec3(-200, 500, 500), Vec3(0, 0, 0))
        # fix problem with Camera3D.look_at updating angles
        self.camera._yaw = -math.acos(self.camera._front.x)
        self.camera._pitch = math.asin(self.camera._front.y)

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

        # PIP dimensions
        width = int(self.width/3)
        height = int(self.height/3)

        # instantiating framebuffers
        self.framebuffer1 = PIP(self, self.batch,
                                Vec3(self.width - width, self.height - self.height/3, 0),
                                cam_position=Vec3(0, 200, 700),
                                pip_width=width, pip_height=height,
                                message="Updating view", border=True
                                )

        self.framebuffer2 = PIP(self, self.batch,
                                Vec3(self.width - width, self.height - 2*self.height/3),
                                pip_width=width, pip_height=height,
                                cam_position=Vec3(1000, 500, 1000),
                                message="Framebuffer 2 view")

        self.framebuffer3 = PIP(self, self.batch,
                                Vec3(self.width - width, self.height - 3*self.height/3),
                                pip_width=width, pip_height=height,
                                cam_position=Vec3(0, 1000, 1),
                                message="Top View")

    def on_draw(self) -> None:
        # update cam position
        self.framebuffer1.cam_position = self.framebuffer1.cam_position + Vec3(1, 0, 0)

        # rendering in default buffer
        self.clear()
        self.batch.draw()

        # rendering framebuffer
        self.framebuffer1.render()
        self.framebuffer2.render()
        self.framebuffer3.render()


if __name__ == '__main__':
    start_app(App, settings)
