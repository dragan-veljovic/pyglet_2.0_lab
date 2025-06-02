"""
This explores options of rendering dynamic objects that transform on every frame.
This is controlled through DynamicModel and DynamicRenderGroup offering two options:
1) Transform on GPU:
    - pass bunch of uniforms then calculate and form matrix in the vertex shader
2) Transform on CPU:
    - calculate matrix on the CPU then pass it to vertex shader

Vertex shader flag transform_on_gpu controls which option is processed.
Static objects can be directly rendered after dynamic object as cleanup is
performed in unset_state(), but it adds updates of uniforms.
Press SPACE to toggle between GPU and CPU transformations.

Performance limitations:
GPU - 720 barrel models at ~100fps
    Vertex shader code is executed per vertex, models vertices have same
    transformation matrix, but it is still calculated for each vertex, slow for large models

CPU - 450 barrel models at 100fps
    Matrix calculated only once per model, but slow in python.

"""
import random
import pyglet.math
import tools.lighting
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *
from tools.model import *
from tools.skybox import Skybox
from pyglet.graphics import ShaderGroup

settings = {
    'default_mode': True,
    'width': 3440,
    'height': 1440,
    'fps': 100,
}


class DynamicModel:
    def __init__(
            self,
            batch: Batch,
            program: ShaderProgram,
            texture_group: Group,
            model_data: dict,
            position=Vec3(0, 0, 0),
            rotation: float = 0.0,
            rotation_dir=Vec3(0, 1, 0),
            scale=Vec3(1, 1, 1),
            origin=Vec3(0, 0, 0),
            transform_on_gpu=False
    ):
        self.batch = batch
        self.program = program
        self.texture_group = texture_group
        self.model_data = model_data
        self.position = position
        self.rotation = rotation
        self.rotation_dir = rotation_dir
        self.scale = scale
        self.origin = origin

        self.render_group = DynamicRenderGroup(
            self, self.program, parent=self.texture_group, transform_on_gpu=transform_on_gpu
        )

        self.vertex_list = get_vertex_list(self.model_data, self.program, self.batch, self.render_group)


class DynamicRenderGroup(Group):
    def __init__(
            self,
            model: DynamicModel,
            program: ShaderProgram,
            order=0,
            parent: Group = None,
            transform_on_gpu=False
    ):
        """
        Dynamically transform model on every frame.
        Model matrix can be calculated on the CPU or GPU (if transform_on_gpu=True).
        TODO: UBO!
        """
        super(DynamicRenderGroup, self).__init__(order, parent)
        self.model = model
        self.program = program
        self.transform_on_gpu = transform_on_gpu

    def set_state(self) -> None:
        self.program['rendering_dynamic_object'] = True
        if self.transform_on_gpu:
            self.program['transform_on_gpu'] = True
            self.program['model_position'] = self.model.position
            self.program['model_rotation'] = Vec3(0, self.model.rotation, 0)
            self.program['model_scale'] = self.model.scale
        else:
            model_matrix = get_model_matrix(
                self.model.position, self.model.rotation, self.model.rotation_dir, self.model.scale, self.model.origin
            )
            self.program['model_precalc'] = model_matrix

    def unset_state(self) -> None:
        self.program['transform_on_gpu'] = False
        self.program['rendering_dynamic_object'] = False

    def __eq__(self, other: Group):
        """ Normally every dynamic object will have unique transformation,
        But eq could be useful for grouping objects that move together,
        ex. passengers inside a bus.
        """
        return False

    def __hash__(self):
        return hash((self.order, self.parent, self.program, self.model))


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

        self.skybox = Skybox('res/textures/skybox/', extension='jpg')

        # trying lights
        # self.light = tools.lighting.SpotLight(Vec3(500, 600, 1500), fov=90, z_far=10000)
        self.light = tools.lighting.DirectionalLight(
            position=Vec3(15000, 15000, 15000), z_far=50000, z_near=0
        )

        self.original_position = self.light.position
        self.program['directional_light'] = self.light.directional_light

        self.shadow_map = tools.lighting.ShadowMap(
            self.light, self.program, self.width, self.height
        )

        self.shader_group = ShaderGroup(self.program)
        self.model_group = DiffuseNormalTextureGroup(
            pyglet.image.load("res/model/Barrel/barrel_BaseColor.png").get_texture(),
            pyglet.image.load("res/model/Barrel/barrel_Normal.png").get_texture(),
            program=self.program, parent=self.shader_group
        )

        # self.terrain_group = DiffuseNormalTextureGroup(
        #     pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg").get_texture(),
        #     pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg").get_texture(),
        #     self.program
        # )

        # self.terrain = get_vertex_list(
        #     transform_model_data(
        #         load_obj_model("res/model/terrain/mountain/terrain_01.obj"),
        #         scale=0.6, position=(0, -2000, 0), rotation=(0, 0, 0), tex_scale=10
        #     ), self.program, self.batch, self.terrain_group
        # )

        self.model_data = transform_model_data(
            load_obj_model('res/model/Barrel/Barrel_OBJ.obj'), scale=400
        )
        self.models = []

        rows, columns, layers = 10, 10, 8
        spacing = 400
        for i in range(rows):
            for j in range(columns):
                for k in range(layers):
                    position = Vec3(spacing * i, spacing * j, spacing * k)
                    self.models.append(
                        DynamicModel(
                            self.batch, self.program,
                            self.model_group, self.model_data,
                            position=position,
                            rotation=random.randint(-100, 100) / 100,
                            transform_on_gpu=True,
                        )
                    )

        self.shadow_map.shadow_program.vertex_list_indexed(
            self.model_data['count'],
            GL_TRIANGLES,
            self.model_data['indices'],
            self.shadow_map.shadow_batch,
            position=('f', self.model_data['position']),
            colors=('Bn', self.model_data['colors'])
        )

        self.transform_on_gpu = True
        self.render_shadow_batch = False
        self.draw_skybox = True
        self.wireframe = False
        self.timer = 0.0
        self.run = True

    def update_main_shader(self):
        # Pass light matrices for shadow calculation
        self.program['light_proj'] = self.light.proj_matrix
        self.program['light_view'] = self.light.view_matrix
        # pass light vectors for phong lighting calculation
        self.program['light_position'] = self.light.position
        self.program['view_position'] = self.camera.position

    def update_light_position(self):
        amplitude = 3000
        self.light.position = self.original_position + Vec3(amplitude + math.sin(self.timer),
                                                            amplitude * math.cos(self.timer), 0)
        self.light.view_matrix = self.light.get_light_view_matrix()

    def on_draw(self):
        if self.run:
            for model in self.models:
                model.rotation += 0.025

        self.timer += 1 / settings['fps']

        # update light
        self.update_light_position()

        # Render shadow map (first pass)
        self.shadow_map.render()

        # input results into main shader
        self.update_main_shader()

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
            case pyglet.window.key.X:
                self.render_shadow_batch = not self.render_shadow_batch
            case pyglet.window.key.SPACE:
                self.transform_on_gpu = not self.transform_on_gpu
                for model in self.models:
                    model.render_group.transform_on_gpu = self.transform_on_gpu

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
    app = start_app(App, settings)
