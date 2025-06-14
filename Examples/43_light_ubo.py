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
        self.clock = pyglet.clock.Clock()
        self.start = self.clock.time()
        self.time = 0.0
        self.run = True

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
        self.group = DiffuseNormalTextureGroup(
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
            position=Vec3(0, 300, 0), ambient=0.25, diffuse=0.75
        )
        # change binding index (multiple UBOs can be bound to same UB)
        self.light_uniform_block.set_binding(2)
        self.spt_light_ubo = self.spt_light.bind_to_block(self.light_uniform_block)

        # representation of the spotlight position

        self.cube = tools.graphics.WireframeCube(
            self.program, self.batch, group=self.group, position=self.spt_light.position, color=self.dir_light.color
        )

        self.plane = tools.graphics.NormalMappedTexturedPlane(
            (-1000, -100, 1000), self.batch, self.group, self.program, rotation=(-math.pi/2, 0, 0),
            length=2000, height=2000
        )

        self.model_tex = DiffuseNormalTextureGroup(
            pyglet.image.load('res/model/chair/old_chair_Albedo.png').get_texture(),
            pyglet.image.load('res/model/chair/old_chair_Normal.png').get_texture(),
            self.program
        )

        self.model = get_vertex_list(
            transform_model_data(
            load_obj_model('res/model/chair/old_chair.obj'), scale=3, position=(0, -100, 0)
        ),
            self.program,
            self.batch,
            self.model_tex
        )



        self.program.use()

    def on_draw(self):
        if self.run:
            self.time = self.clock.time() - self.start

            self.spt_light.position = Vec3(100*math.sin(self.time), 300 + 200*math.cos(self.time), 0)
            self.spt_light.target = Vec3(100 * math.cos(self.time), -100, 0)

            # or for flashlight
            # self.spt_light.position = self.camera.position
            # self.spt_light.target = self.camera.target


            # this could be universal way to update objects like Camera, models etc.
            self.spt_light.update_ubo(['position', 'target'])

            self.cube.vertex_list.delete()
            self.cube = tools.graphics.WireframeCube(
                self.program, self.batch, group=self.group, position=self.spt_light.position, color=self.dir_light.color
            )

        self.program['view_position'] = self.camera.position

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.SPACE:
                self.run = not self.run
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



