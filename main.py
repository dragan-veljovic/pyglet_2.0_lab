"""
Updated OBJ model loading, allowing for quad or triangle faces.
Includes normal map calculation, if normals are not present.
Includes tangent and bitanget calculation for normal mapping.
TODO: implement 3 methods as a class, possibly fuse with pyglet model
"""
import copy
import math
import pyglet.math

import tools.lighting
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *
from tools.model import *
from tools.skybox import Skybox

settings = {
    'default_mode': True,
    'width': 1280,
    'height': 720,
    'fps': 60,
}


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        self.batch = pyglet.graphics.Batch()
        self.shadow_batch = pyglet.graphics.Batch()

        # camera settings
        self.camera = Camera3D(self, z_far=8000)
        pyglet.resource.path.append('res')
        pyglet.resource.reindex()

        self.program = ShaderProgram(
            pyglet.resource.shader("shaders/dev.vert"),
            pyglet.resource.shader("shaders/dev.frag"),
        )
        self.program['z_far'] = self.camera.z_far

        glEnable(GL_DEPTH_TEST)

        self.wall_group = DiffuseNormalTextureGroup(
            diffuse=pyglet.image.load('res/textures/brick/brickwall.jpg').get_texture(),
            normal=pyglet.image.load('res/textures/brick/normal_mapping_normal_map.png').get_texture(),
            program=self.program
        )

        self.skybox = Skybox('res/textures/skybox2/', extension='png')

        # main scene elements
        self.back_wall = NormalMappedTexturedPlane(
            (-750, -500, -500), self.batch, self.wall_group, self.program, 1500, 1000, color=(150, 150, 150, 255)
        )
        self.floor = NormalMappedTexturedPlane(
            (-750, -500, 500), self.batch, self.wall_group, self.program, 1500, 1000, rotation=(-np.pi / 2, 0, 0),
            color=(100, 100, 100, 255)
        )

        self.left_wall = NormalMappedTexturedPlane(
            (-750, -500, 500), self.batch, self.wall_group, self.program, 1500, 1000, rotation=(0, -np.pi / 3, 0),
            color=(150, 150, 150, 255)
        )

        self.barrel_group = DiffuseNormalTextureGroup(
            pyglet.image.load("res/model/Barrel/barrel_BaseColor.png").get_texture(),
            pyglet.image.load("res/model/Barrel/barrel_Normal.png").get_texture(),
            program=self.program
        )


        self.terrain = create_mesh_from_obj(
            "res/model/terrain/mountain/terrain_01.obj", self.batch, shader_program=self.program,
            group=DiffuseNormalTextureGroup(
                pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg").get_texture(),
                pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg").get_texture(),
                self.program
            ),
            scale=0.6, position=(0, -2000, 0), rotation=(0, 0, 0), tex_scale=10
        )

        # trying lights

        self.light = tools.lighting.SpotLight(Vec3(500, 600, 1500), fov=90, z_far=10000)

        #self.light = tools.lighting.DirectionalLight(position=Vec3(15000, 25000, 15000), z_far=50000, z_near=0)


        self.original_position = self.light.position
        self.program['directional_light'] = self.light.directional_light
        # self.program['ambient_strength'] = 0.05

        self.shadow_map = tools.lighting.ShadowMap(
            self.light, self.program, 1280, 1280
        )

        rows, columns, layers = 3, 3, 2
        spacing = 400
        for i in range(rows):
            for j in range(columns):
                for k in range(layers):
                # scene domain
                    self.plant = create_mesh_from_obj(
                        "res/model/Barrel/Barrel_OBJ.obj", self.batch, shader_program=self.program,
                        scale=300, group=self.barrel_group, position=(i*spacing, k * spacing, j*spacing), rotation=(0, -90, 0))

                    # shadow domain
                    self.shadow_model = self.shadow_map.shadow_program.vertex_list_indexed(
                        count=len(self.plant.position)//3,
                        mode=GL_TRIANGLES,
                        indices=self.plant.indices,
                        batch=self.shadow_map.shadow_batch,
                        position=('f', self.plant.position[:]),
                    )

        self.render_shadow_batch = False
        self.draw_skybox = True
        self.wireframe = False
        self.timer = 0.0
        self.move_light = True

        # light

    def update_main_shader(self):
        # Pass light matrices for shadow calculation
        self.program['light_proj'] = self.light.proj_matrix
        self.program['light_view'] = self.light.view_matrix
        # pass light vectors for phong lighting calculation
        self.program['light_position'] = self.light.position
        self.program['view_position'] = self.camera.position

    def update_light_position(self):
        if self.move_light:
            self.timer += 1 / settings['fps']
            amplitude = 3000
            self.light.position = self.original_position + Vec3(amplitude + math.sin(self.timer), amplitude * math.cos(self.timer), 0)
            self.light.view_matrix = self.light.get_light_view_matrix()

    def on_draw(self):
        # Render shadow map (first pass)
        self.shadow_map.render()

        # input results into main shader
        self.update_main_shader()

        # update light position
        self.update_light_position()

        # Render main scene
        self.clear()

        if self.draw_skybox:
            self.skybox.draw()

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # redner main shader or the depth map
        if self.render_shadow_batch:
            self.shadow_map.shadow_batch.draw()
        else:
            self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.C:
                self.depth_data = self.shadow_map.fetch_depth_data()
                print("buffer depth data: ", self.depth_data)
            case pyglet.window.key.X:
                self.render_shadow_batch = not self.render_shadow_batch
            case pyglet.window.key.SPACE:
                self.move_light = not self.move_light

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
            case pyglet.window.key.B:
                self.draw_skybox = not self.draw_skybox


if __name__ == '__main__':
    start_app(App, settings)
