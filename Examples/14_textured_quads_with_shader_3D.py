"""
Following is an example of how to texture a quad made of two GL_TRIANGLES.
When pyglet Texture is created with either
self.texture = pyglet.resource.texture('res/textures/img.png')
self.texture = pyglet.image.load('res/textures/img_1.png').get_texture()
it pyglet automatically binds it to all quads in the batch.

To change the texture of our quad we need a Group subclass.

"""
import random

import pyglet.shapes

import tools.camera
from tools.definitions import *
import tools.tools_old.color
from pyglet.gl import *

FPS = 60
WIDTH = 1280
HEIGHT = 720

vertex_source = ("""
    #version 330 core
    in vec3 position;
    in vec2 tex_coords;
    out vec2 texture_coords;

    uniform WindowBlock{
        mat4 projection;
        mat4 view;
    } window;

    uniform float time;

    void main(){
        vec3 new_position = position;
        new_position.yz += new_position.yz * sin(time) * 0.2;
        new_position.z += new_position.z * sin(time) * 20;
        gl_Position = window.projection * window.view * vec4(new_position, 1);
        texture_coords = tex_coords;
    }
""")

fragment_source = ("""
    #version 330 core
    in vec2 texture_coords;
    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main(){
        final_colors = texture(our_texture, texture_coords.xy);
    }
""")

shader_program = pyglet.graphics.shader.ShaderProgram(
    pyglet.graphics.shader.Shader(vertex_source, 'vertex'),
    pyglet.graphics.shader.Shader(fragment_source, 'fragment')
)


class RenderGroup(pyglet.graphics.Group):
    def __init__(
            self,
            texture: pyglet.image.Texture,
            program: pyglet.graphics.shader.ShaderProgram,
            order=0,
            parent=None
    ):
        """
        A Group that enables and binds a Texture and ShaderProgram.
        RenderGroups are equal if their Texture and ShaderProgram are equal.
        :param texture: Texture to bind.
        :param program: ShaderProgram to use.
        :param order: Change the order to render above or below other Groups.
        :param parent: Parent group.
        """
        super().__init__(order, parent)
        self.texture = texture
        self.program = program

    def set_state(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()

    def unset_state(self):
        glDisable(GL_BLEND)
        self.program.stop()

    def __hash__(self):
        return hash((self.texture.target, self.texture.id, self.order, self.parent, self.program))

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        set_background_color()
        center_window(self)

        self.batch = pyglet.graphics.Batch()
        self.program = shader_program
        self.camera = tools.camera.Camera3D(self, z_near=5, z_far=10_000, speed=20, fov=90)
        self.time = 0.0
        self.run = True

        self.texture = pyglet.resource.texture('res/textures/img.png')
        self.texture2 = pyglet.image.load('res/textures/img_1.png').get_texture()
        self.texture3 = pyglet.resource.texture('res/textures/img_2.png')

        self.group3 = RenderGroup(self.texture, self.program)
        self.group4 = RenderGroup(self.texture2, self.program)
        self.group5 = RenderGroup(self.texture2, self.program)

        self.quad = self.get_vertical_quad(self.batch, group=self.group3, width=400, height=300, z=200)
        self.quad2 = self.get_vertical_quad(self.batch, group=self.group4, z=300)

        self.quads = [self.get_vertical_quad(self.batch, self.group5, x=i * 200, y=j * 200, z=random.randint(-5, 5)) for
                      i in range(-5, 5) for j in range(-3, 3)]

        glEnable(GL_DEPTH_TEST)

    def get_vertical_quad(self, batch=None, group=None, x=0, y=0, z=0, width=200, height=200):
        vertices = (
            x, y, z, x + width, y, z, x + width, y + height, z,
            x, y, z, x + width, y + height, z, x, y + height, z
        )
        tex_coords = (
            0, 0, 1, 0, 1, 1,
            0, 0, 1, 1, 0, 1
        )

        return self.program.vertex_list(
            6, pyglet.gl.GL_TRIANGLES, batch=batch, group=group,
            position=('f', vertices),
            tex_coords=('f', tex_coords)
        )

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.SPACE:
            self.run = not self.run

    def on_draw(self) -> None:
        if self.run:
            self.time += 1 / FPS
            self.program['time'] = self.time

        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.batch.draw()


if __name__ == '__main__':
    App(width=WIDTH, height=HEIGHT, resizable=True, config=get_config())
    pyglet.app.run(1 / FPS)


