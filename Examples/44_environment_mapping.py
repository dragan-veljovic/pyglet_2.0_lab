import pyglet.graphics

import tools.model
from tools.definitions import *
from tools.model import *
from tools.camera import Camera3D
from tools.lighting import DirectionalLight
from tools.skybox import Skybox
from pyglet.model import Model
from tools.instancing import InstanceRendering
import numpy as np
from tools.graphics import TextureGroup
from pyglet.graphics.shader import create_string_buffer


settings = {
    'default_mode': True,
    'config': get_config(samples=8)
}


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pyglet.resource.path.append('res')
        pyglet.resource.reindex()
        self.batch = pyglet.graphics.Batch()
        self.program = ShaderProgram(
            pyglet.resource.shader('res/shaders/dev.vert', 'vertex'),
            pyglet.resource.shader('res/shaders/dev.frag', 'fragment')
        )
        self.camera = Camera3D(self, z_far=100_000, speed=5)
        self.time = 0.0
        self.clock = pyglet.clock.Clock()
        self.start_time = self.clock.time()
        self.draw_skybox = True
        self.wireframe = False
        self.run = True

        self.light = DirectionalLight()
        self.light.bind_to_block(self.program.uniform_blocks['LightBlock'])
        self.skybox = Skybox('res/textures/skybox')

        shader_group = pyglet.graphics.ShaderGroup(self.program)
        blend_group = BlendGroup(parent=shader_group)

        group = DiffuseNormalTextureGroup(
            diffuse=pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            normal=pyglet.image.load(
               'res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            program=self.program, parent=blend_group
        )

        #self.cube = Cuboid(self.program, size=(500, 500, 500), batch=self.batch, group=group, color=(0.4, 0.4, 0.55, 1))

        # loading OBJ model procedure
        model_data = load_obj_model('res/model/vessel.obj')
        transformed_data = transform_model_data(model_data, scale=1, tex_scale=3, color=(0.75, 0.2, 0.0, 1.0))
        vertex_list = get_vertex_list(transformed_data, self.program, self.batch, group)
        self.model = Model([vertex_list], [group], batch=self.batch)

        # setting up skybox texture in the shader (should be a group)
        self.program.use()
        # get uniform location
        # skybox_location = glGetUniformLocation(self.program.id, create_string_buffer("skybox".encode('utf-8')))
        # # assign texture unit slot
        # glUniform1i(skybox_location, 5)
        self.program['skybox'] = 5
        # activate texture slot
        glActiveTexture(GL_TEXTURE5)
        # bind texture
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.skybox.cube_map)

        self.plane = Plane(self.program, self.batch, group, length=2000, width=1500, centered=True, color=(0.75, 0, 0, 1))
        # self.plane = tools.graphics.NormalMappedTexturedPlane(
        #     (0, 0, 0), self.batch, group, self.program, length=1000, height=1500, rotation=(math.pi/2, 0, 0)
        # )

        #self.program['rendering_dynamic_object'] = True
        self.program['shadow_mapping'] = False
        self.program['environment_mapping'] = True


        glEnable(GL_DEPTH_TEST)

    def on_draw(self):
        if self.run:
            self.time = self.clock.time() - self.start_time
            self.plane.matrix = get_model_matrix(Vec3(0, 0, 0), self.time*0.1, rotation_dir=Vec3(1, 1, 0).normalize())
            self.program['model_precalc'] = self.plane.matrix

        self.program['view_position'] = self.camera.position
        self.clear()

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.draw_skybox:
            self.skybox.draw()

        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.SPACE:
                self.run = not self.run
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
                self.program['environment_mapping'] = not self.program['environment_mapping']
            case pyglet.window.key.H:
                self.program['normal_mapping'] = not self.program['normal_mapping']
            case pyglet.window.key.V:
                self.wireframe = not self.wireframe
            case pyglet.window.key.B:
                self.draw_skybox = not self.draw_skybox


if __name__ == '__main__':
    app = start_app(App, settings)
