import math

import tools.graphics
from tools.definitions import *
from tools.camera import Camera3D
from tools.lighting import DirectionalLight, SpotLight
from pyglet.gl import *
from pyglet.math import Vec4, Mat4
from tools.model import *


class Win(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)

        pyglet.resource.path.append('res/')
        pyglet.resource.reindex()

        self.program = ShaderProgram(
            pyglet.resource.shader('res/shaders/dev.vert', 'vertex'),
            pyglet.resource.shader('res/shaders/dev.frag', 'fragment')
        )

        self.program['shadow_mapping'] = False

        self.create_scene()

        glEnable(GL_DEPTH_TEST)
        self.wireframe = False

    def create_scene(self):
        group = DiffuseNormalTextureGroup(
            pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            self.program
        )

        # uniform block usage with lights
        # create reference to uniform block declared in the shader
        self.light_uniform_block = self.program.uniform_blocks['LightBlock']

        # create custom light
        self.dir_light = DirectionalLight(
            ambient=0.25, diffuse=0.75,
        )
        # get ubo from light attributes
        self.dir_light_ubo = self.dir_light.bind_to_block(self.light_uniform_block)

        # create another custom light
        self.spt_light = SpotLight(
            position=Vec3(500, 1000, -2500), ambient=0.25, diffuse=0.75,
        )
        # change binding index (multiple UBOs can be bound to same UB)
        self.light_uniform_block.set_binding(2)
        self.spt_light_ubo = self.spt_light.bind_to_block(self.light_uniform_block)

        # Now manually switch between two light configurations by changing binding!
        self.light_uniform_block.set_binding(1)

        # representation of the spotlight position
        self.cube = tools.graphics.WireframeCube(
            self.program, self.batch, group=group, position=self.spt_light.position, color=self.dir_light.color
        )

        self.plane = tools.graphics.NormalMappedTexturedPlane(
            (-1000, -100, -1000), self.batch, group, self.program, rotation=(-math.pi/2, 0, 0),
            length=2000, height=2000
        )

        self.program.use()

    def on_draw(self):
        self.program['view_position'] = self.camera.position
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        match symbol:
            # shader render settings
            case pyglet.window.key.M:
                self.program['shadow_mapping'] = not self.program['shadow_mapping']
            case pyglet.window.key.L:
                self.program['lighting'] = not self.program['lighting']
            case pyglet.window.key.T:
                self.program['texturing'] = not self.program['texturing']
            case pyglet.window.key.O:
                self.program['lighting_diffuse'] = not self.program['lighting_diffuse']
            case pyglet.window.key.P:
                self.program['lighting_specular'] = not self.program['lighting_specular']
            case pyglet.window.key.N:
                self.program['soft_shadows'] = not self.program['soft_shadows']
            case pyglet.window.key.H:
                self.program['normal_mapping'] = not self.program['normal_mapping']
            case pyglet.window.key.V:
                self.wireframe = not self.wireframe
            case pyglet.window.key.UP:
                self.program.uniform_blocks['LightBlock'].set_binding(1)
            case pyglet.window.key.DOWN:
                self.program.uniform_blocks['LightBlock'].set_binding(2)


if __name__ == '__main__':
    win = Win(height=1080, width=1920)
    pyglet.app.run(1/100)



