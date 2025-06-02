import pyglet

"""
Example showing current 3 possible rendering modes:
STATIC - objects that do not move, optimized by pyglet internal batching. 
    Theoretically, amount limited by VRAM or CPU calls if many different Groups.
DYNAMIC - objects updated on every frame, with full flexibility. 
This can be done on GPU by passing parameters to shader, or on CPU by passing entire model matrix.
    Limited by CPU due to each DynamicRenderGroup being unique, or
    by GPU as get_model_matrix is called per vertex. 
INSTANCE - highly optimized way of drawing many instances of same 3D mesh.
    Limited by the GPU get_model_matrix(). Work in progress.  
 
TODO: 
1) Programmable InstanceRendering (variable attributes with passed dict)
    So as to enable other effects like normal mapping and future effects.
2) mat4 instance_data format - more parameters for GPU transform and 
    possible passing of mat4 calculated on the CPU
3) if 2 completed, numpy method of generating mat4 and shader update to 
    allow switching from GPU to CPU transform during instance similar to dynamic
"""

import tools.camera
from tools.definitions import *
from tools.instancing import InstanceRendering
import numpy as np
import math
from pyglet.gl import *
from tools.model import load_obj_model, DynamicModel, DynamicRenderGroup
from tools.skybox import Skybox
from pyglet.math import Vec3
from tools.lighting import DirectionalLight
from tools.model import *


class TextureGroup(pyglet.graphics.Group):
    def __init__(self, program, texture, order=0, parent=None):
        super(TextureGroup, self).__init__(order, parent)
        self.program = program
        self.texture = texture
        self.program['diffuse_texture'] = 0

    def set_state(self) -> None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)

    def unset_state(self) -> None:
        glBindTexture(self.texture.target, 0)

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.parent == other.parent)

    def __hash__(self) -> int:
        return hash((self.texture.target, self.texture.id, self.order, self.parent))

    def __enter__(self):
        self.set_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unset_state()


class MyApp(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.batch = pyglet.graphics.Batch()
        self.shadow_batch = pyglet.graphics.Batch()

        pyglet.resource.path.append('res')
        pyglet.resource.reindex()
        self.program = ShaderProgram(
            pyglet.resource.shader("shaders/dev.vert"),
            pyglet.resource.shader("shaders/dev.frag"),
        )

        self.skybox = Skybox('res/textures/skybox2/', 'png')
        self.light = tools.lighting.DirectionalLight(
            position=Vec3(15000, 15000, 15000), z_far=50000, z_near=0
        )
        self.shadow_map = tools.lighting.ShadowMap(
            self.light, self.program, self.width, self.height
        )

        self.original_position = self.light.position
        self.program['directional_light'] = self.light.directional_light

        set_background_color()
        self.camera = tools.camera.Camera3D(self, z_far=30_000)
        self.batch = pyglet.graphics.Batch()
        self.shadow_batch = pyglet.graphics.Batch()
        self.clock = pyglet.clock.Clock()
        self.start = self.clock.time()
        self.time = 0.0

        glEnable(GL_DEPTH_TEST)

        # ------------- STATIC RENDERING ---------------
        # No spe
        self.terrain_group = DiffuseNormalTextureGroup(
            pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg").get_texture(),
            pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg").get_texture(),
            self.program
        )

        self.terrain = get_vertex_list(
            transform_model_data(
                load_obj_model("res/model/terrain/mountain/terrain_01.obj"),
                scale=0.6, position=(0, -2000, 0), rotation=(0, 0, 0), tex_scale=10
            ), self.program, self.batch, self.terrain_group
        )

        # ------------- INSTANCE RENDERING ---------------
        self.model_data = load_obj_model('res/model/asteroid/planet.obj')
        self.texture_group = TextureGroup(
            self.program, pyglet.image.load('res/model/asteroid/mars.png').get_texture()
        )
        N = 200  # number of instances is square of this
        self.num_instances = N ** 2

        # self.instance_data = np.zeros((self.num_instances, 4), dtype=np.float32)
        self.instance_data = np.array((0, 0, 0, 10) * self.num_instances, dtype=np.float32).reshape(self.num_instances,
                                                                                                     4)
        self.instances = InstanceRendering(
            position=np.array(self.model_data['position'], dtype=np.float32),
            indices=np.array(self.model_data['indices'], dtype=np.uint32),
            instance_data=self.instance_data,
            num_instances=self.num_instances,
            color=np.array(self.model_data['colors'], dtype=np.float32),
            tex_coord=np.array(self.model_data['tex_coords'], dtype=np.float32),
            normal=np.array(self.model_data['normals'], dtype=np.float32),
            program=self.program,
            group=self.texture_group,
            update_func=self.update_instance_data_numpy
        )

        # ------------- DYNAMIC RENDERING ---------------
        self.barrel_model_data = load_obj_model('res/model/Barrel/Barrel_OBJ.obj')
        self.dynamic_model = DynamicModel(
            self.batch, self.program,
            DiffuseNormalTextureGroup(
                pyglet.image.load('res/model/Barrel/barrel_BaseColor.png').get_texture(),
                pyglet.image.load('res/model/Barrel/barrel_Normal.png').get_texture(),
                self.program
            ),
            model_data=transform_model_data(self.barrel_model_data),
            scale=Vec3(400, 400, 400)
        )


        self.render_shadow_batch = False
        self.draw_skybox = True
        self.wireframe = False
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
        self.light.position = self.original_position + Vec3(amplitude + math.sin(self.time),
                                                            amplitude * math.cos(self.time), 0)
        self.light.view_matrix = self.light.get_light_view_matrix()

    def update_instance_data(self):
        """Use numpy version to avoid looping."""
        N = int(math.sqrt(self.num_instances))

        index = 0
        for y in range(N):
            for x in range(N):
                # Position (in a grid pattern)
                self.instance_data[index, 0] = x * 75  # x position
                self.instance_data[index, 1] = y * 75  # y position

                distance = math.sqrt((x) ** 2 + (y) ** 2) / 7.0

                # Rotation (animated over time, different for each instance)
                self.instance_data[index, 2] = self.time * 0.5 + distance * 2

                # Scale
                self.instance_data[index, 3] = 10 + 5 * math.sin(self.time)

                index += 1

    def update_instance_data_numpy(self):
        """
        Around 20x faster than pure python.
        Both can be optimized by precalculating x, y and distances.
        """
        N = int(math.sqrt(self.num_instances))

        # Create grid of x, y indices
        y_indices, x_indices = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

        # Flatten the grid
        x_flat = x_indices.ravel()
        y_flat = y_indices.ravel()

        # Compute positions
        self.instance_data[:, 0] = x_flat * 75  # x position
        self.instance_data[:, 1] = y_flat * 75  # y position

        # Compute distance from origin
        distances = np.sqrt(x_flat ** 2 + y_flat ** 2) / 7.0

        # Compute rotation
        self.instance_data[:, 2] = self.time * 0.5 + distances * 2

        # Compute scale (same for all instances)
        self.instance_data[:, 3] = 10 + 2*np.sin(self.time)

    def update_dynamic_object(self):
        self.dynamic_model.position = Vec3(500 + 100*math.sin(self.time), 0, 500)
        self.dynamic_model.scale += math.sin(self.time)
        self.dynamic_model.rotation = self.time*0.1

    def on_draw(self):
        self.time = self.clock.time() - self.start

        self.update_dynamic_object()

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
            self.program.use()
            self.batch.draw()
            self.program.stop()

        self.instances.update()
        self.program['instance_rendering'] = True  # put into Group
        self.instances.draw()
        self.program['instance_rendering'] = False  # put into Group

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.X:
                self.render_shadow_batch = not self.render_shadow_batch

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
    app = start_app(MyApp, {'default_mode': True, 'vsync': False})
    app.instances.delete()
