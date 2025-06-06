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
  - avoid hard-coded instance_data location and attribute names
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
from pyglet.graphics import Group
from pathlib import Path
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyTextureGroup(Group):
    def __init__(self, program, texture, order=0, parent=None):
        super().__init__(order, parent)
        self.program = program
        self.texture = texture
        self.program['diffuse_texture'] = 0

    def set_state(self) -> None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)

    def unset_state(self) -> None:
        # glBindTexture(self.texture.target, 0)
        pass

    def __hash__(self) -> int:
        return hash((self.texture.target, self.texture.id, self.order, self.parent))

    def __eq__(self, other: Group) -> bool:
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.parent == other.parent)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(id={self.texture.id})'

    def __enter__(self):
        self.set_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unset_state()


class InstanceGroup(Group):
    def __init__(self, program, order=0, parent=None):
        super().__init__(order, parent)
        self.program = program

    def set_state(self) -> None:
        self.program['instance_rendering'] = True

    def unset_state(self) -> None:
        self.program['instance_rendering'] = False

    def __hash__(self) -> int:
        return hash((self.program, self.order, self.parent))

    def __eq__(self, other: "InstanceGroup") -> bool:
        return (self.__class__ is other.__class__ and
                self.program == other.program and
                self.order == other.order and
                self.parent == other.parent)


def get_cached_obj_data(
        path: str,
        save_dir="res/model/cached/",
        scale=1.0, position=(0, 0, 0), rotation=(0, 0, 0), tex_scale=1.0,
        force_reload=False,
        old_version_cleanup=True
) -> dict:
    """
    Fast load of a 3D model's transformed data from a cache file if available,
    otherwise process, cache, and return the transformed model data.
    Cached filename includes original model and a hash value based on passed parameters.
    Older versions of the same model are removed by default.
    """

    def save_model_data(save_path: Path, data: dict):
        with save_path.open('wb') as f:
            pickle.dump(data, f)

    def load_model_data(load_path: Path):
        with load_path.open('rb') as f:
            return pickle.load(f)

    def cleanup(model_base_name: str, keep_filename: str, save_dir="res/model/cached/"):
        """Remove all pickled files for a given model, except the one in use."""
        cache_path = Path(save_dir)
        for file in cache_path.glob(f"{model_base_name}_*.pkl"):
            if file.name != keep_filename:
                file.unlink()
                logger.info(f" Removed old cache file: {file}")

    # get filename
    filename = Path(path).name
    if filename.lower().endswith('.obj'):
        name = filename.rsplit('.', 1)[0]
    else:
        raise NameError("Expected '.obj' file format.")

    # generate has based on transformation parameters
    param_str = f"{scale}_{position}_{rotation}_{tex_scale}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]  # short hash
    hashed_name = f"{name}_{param_hash}.pkl"

    # generating cached file path
    save_dir_path = Path(save_dir)
    cached_file_path = save_dir_path / hashed_name

    # load and return cached file if exists, otherwise create cached file
    if force_reload or not cached_file_path.exists():
        save_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f" Generating and caching model to: {cached_file_path}")
        data = transform_model_data(
            load_obj_model(path),
            scale=scale, position=position, rotation=rotation, tex_scale=tex_scale
        )
        save_model_data(cached_file_path, data)

        # Clean cached old versions of this model
        if old_version_cleanup:
            cleanup(name, hashed_name, save_dir)
        return data

    logger.info(f" Loading cached model: {cached_file_path}")
    return load_model_data(cached_file_path)


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

        self.shader_group = pyglet.graphics.ShaderGroup(
            self.program
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
        self.camera = tools.camera.Camera3D(self, z_far=11_000)
        self.program['z_far'] = self.camera.z_far
        self.program['fade_length'] = 2000
        self.batch = pyglet.graphics.Batch()
        self.shadow_batch = pyglet.graphics.Batch()
        self.clock = pyglet.clock.Clock()
        self.start = self.clock.time()
        self.time = 0.0

        glEnable(GL_DEPTH_TEST)

        # Texture groups
        self.barrel_group = DiffuseNormalTextureGroup(
            pyglet.image.load('res/model/Barrel/barrel_BaseColor.png').get_texture(),
            pyglet.image.load('res/model/Barrel/barrel_Normal.png').get_texture(),
            self.program,
            parent=self.shader_group
        )

        self.terrain_group = DiffuseNormalTextureGroup(
            pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg").get_texture(),
            pyglet.image.load("res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg").get_texture(),
            self.program,
            parent=self.shader_group
        )

        self.instance_tex_group = DiffuseNormalTextureGroup(
            pyglet.image.load('res/model/asteroid/mars.png').get_texture(),
            pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            self.program,
            parent=self.shader_group
        )

        # ------------- STATIC RENDERING ---------------
        # file_path = Path('res/model/cached/terrain.pkl')
        # if file_path.exists():
        #     self.terrain_data = load_model_data('res/model/cached/terrain.pkl')
        # else:
        #     self.terrain_data = transform_model_data(
        #             load_obj_model("res/model/terrain/mountain/terrain_01.obj"),
        #             scale=0.6, position=(0, -2000, 0), rotation=(0, 0, 0), tex_scale=10
        #     )
        #     save_model_data(self.terrain_data, 'terrain.pkl')

        self.terrain_data = get_cached_obj_data(
            'res/model/terrain/mountain/terrain_01.obj',
            scale=0.6, position=(0, -2000, 0), rotation=(0, 0, 0), tex_scale=10,
            force_reload=False
        )

        self.terrain = get_vertex_list(
            self.terrain_data, self.program, self.batch, self.terrain_group
        )

        # ------------- DYNAMIC RENDERING ---------------
        self.barrel_model_data = get_cached_obj_data('res/model/Barrel/Barrel_OBJ.obj')
        self.dynamic_model = DynamicModel(
            self.batch, self.program,
            texture_group=self.barrel_group,
            model_data=transform_model_data(self.barrel_model_data),
            scale=Vec3(400, 400, 400)
        )

        # ------------- INSTANCE RENDERING ---------------
        self.model_data = transform_model_data(
            load_obj_model('res/model/asteroid/planet.obj')
        )

        N = 100  # number of instances is square of this
        self.num_instances = N ** 2
        self.instance_data = np.array(
            (
                10, 0, 0, 0,
                0, 10, 0, 0,
                0, 0, 10, 0,
                0, 0, 0, 1,
            ) * self.num_instances, dtype=np.float32
        ).reshape(self.num_instances, 16)

        self.instance_group = InstanceGroup(
            self.program,
            parent=self.instance_tex_group
        )

        self.instances = InstanceRendering(
            model_data=self.model_data,
            instance_data=self.instance_data,
            num_instances=self.num_instances,
            program=self.program,
            group=self.instance_group,
            update_func=self.update_func_for_mat4
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

    def update_func_for_mat4(self):
        time = self.clock.time() - self.start
        N = int(np.sqrt(self.num_instances))
        y_indices, x_indices = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

        # Flatten the grid
        x_flat = x_indices.ravel()
        y_flat = y_indices.ravel()

        distance = np.hypot(x_flat, y_flat) / 7

        # use this data to calculate on the GPU -> get_instance_model_matrix_gpu()
        # translation
        self.instance_data[:, 0] = x_flat * 75 + 75 * np.sin(distance * time * 0.1)  # x position
        self.instance_data[:, 1] = y_flat * 75 + 75 * np.cos(distance * time * 0.1)  # y position
        # rotation
        self.instance_data[:, 4:6] += 0.1
        # scale
        self.instance_data[:, 8:10] = 10 + 3 * np.sin(time)

        # OR precalculated matrix on the CPU -> get_instance_model_matrix()
        # self.instance_data[:, 12] = x_flat * 75 + 50 * np.sin(distance * time*0.1)  # x position
        # self.instance_data[:, 13] = y_flat * 75 + 50 * np.cos(distance * time*0.1)  # y position

    def update_dynamic_object(self):
        self.dynamic_model.position = Vec3(500 + 100 * math.sin(self.time), 0, 500)
        self.dynamic_model.scale += math.sin(self.time)
        self.dynamic_model.rotation = self.time * 0.1

    def on_draw(self):
        self.time = self.clock.time() - self.start

        if self.run:
            self.update_dynamic_object()
            self.instances.update()

        # update light
        self.update_light_position()

        # Render shadow map (first pass)
        self.shadow_map.render()

        # input results into main shader
        self.update_main_shader()

        # Render main scene
        self.clear()

        # redner main shader or the depth map for testing
        if self.render_shadow_batch:
            self.shadow_map.shadow_batch.draw()
        else:
            if self.wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            if self.draw_skybox:
                self.skybox.draw()

            self.batch.draw()

            self.program['specular_strength'] = 5.0
            self.instances.draw()
            self.program['specular_strength'] = 1.0

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.X:
                self.render_shadow_batch = not self.render_shadow_batch
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
                self.program['soft_shadows'] = not self.program['soft_shadows']
            case pyglet.window.key.H:
                self.program['normal_mapping'] = not self.program['normal_mapping']
            case pyglet.window.key.V:
                self.wireframe = not self.wireframe
            case pyglet.window.key.B:
                self.draw_skybox = not self.draw_skybox


if __name__ == '__main__':
    app = start_app(MyApp, {'default_mode': True, 'vsync': True})
    app.instances.delete()
