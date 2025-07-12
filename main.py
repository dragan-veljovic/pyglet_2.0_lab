import pyglet.graphics

from tools.definitions import *
from tools.model import *
from tools.camera import Camera3D
from tools.lighting import DirectionalLight
from tools.skybox import Skybox

settings = {
    'default_mode': True,
    'config': get_config()
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

        self.camera = Camera3D(self, z_far=12000, speed=20)
        self.ubo = self.program.uniform_blocks['WindowBlock'].create_ubo()

        with self.ubo as ubo:
            ubo.projection = self.projection
            ubo.view = self.view
            ubo.z_far = self.camera.z_far
            ubo.fade_length = 2000
            ubo.projection = self.projection
            ubo.view = self.view

        self.time = 0.0
        self.clock = pyglet.clock.Clock()
        self.start_time = self.clock.time()

        self.light = DirectionalLight(position=Vec3(1, 1, 1))
        self.light.bind_to_block(self.program.uniform_blocks['LightBlock'])
        self.skybox = Skybox('res/textures/skybox')
        self.skybox.set_environment_map(self.program)

        shader_group = pyglet.graphics.ShaderGroup(self.program)
        blend_group = BlendGroup(parent=shader_group)

        terrain_texture = DiffuseNormalTextureGroup(
            diffuse=pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            normal=pyglet.image.load(
               'res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            program=self.program, parent=blend_group
        )

        material_texture = DiffuseNormalTextureGroup(
            pyglet.image.load('res/textures/textures/slate_floor_03_diff_4k.jpg').get_texture(),
            normal=pyglet.image.load('res/textures/textures/slate_floor_03_nor_gl_4k.jpg').get_texture(),
            program=self.program, parent=blend_group
        )

        self.material_ubo = self.program.uniform_blocks['MaterialBlock'].create_ubo()
        self.material_group = MaterialGroup(self.material_ubo, parent=terrain_texture, reflection_strength=1.0, shininess=128, bump_strength=0.2, f0_reflectance=0.04)
        self.sphere1 = Sphere(self.program, self.batch, self.material_group, Vec3(0, 0, 0), radius=150, lat_segments=32)

        terrain_group = MaterialGroup(self.material_ubo, parent=terrain_texture, shininess=128, bump_strength=0.2, specular=0, reflection_strength=1.0)
        self.terrain = load_mesh('res/model/terrain/models/mars/terrain/mars.OBJ', self.program, self.batch, terrain_group, False, position=(10000, -5000, -10000), scale=10, tex_scale=5)

        material_group2 = MaterialGroup(self.material_ubo, parent=terrain_texture, reflection_strength=1, shininess=128, bump_strength=1, f0_reflectance=(0.839, 0.565, 0.255, 1.0))
        self.meshes = [
            load_mesh(
                'res/model/vessel.obj', self.program, self.batch, material_group2,
                position=(i*300 + 300, 0, 0), scale=0.5, tex_scale=3, add_tangents=True, color=(0.0, 0.0, 0.0, 1.0)
            ) for i in range(1)
        ]

        self.mesh_x = 0.0
        self.mesh_y = 0.0
        self.mesh_z = 0.0

        self.program['shadow_mapping'] = False
        self.program['normal_mapping'] = False
        self.draw_skybox = True
        self.wireframe = False
        self.run = True

        glEnable(GL_DEPTH_TEST)

    def on_draw(self):
        if self.run:
            self.time = self.clock.time() - self.start_time
            #self.cube.matrix = get_model_matrix(Vec3(0, 0, 0), self.time*0.1, rotation_dir=Vec3(0, 1, 0).normalize(), origin=self.cube.position)
            self.sphere1.matrix = get_model_matrix(Vec3(0, 0, 0), self.time*0.1, rotation_dir=Vec3(0, 1, 0).normalize(), origin=self.sphere1.position)

            #self.mesh.matrix = pyglet.math.Mat4.from_rotation(-math.sin(self.time), Vec3(0, 1, 0)) @ Mat4.from_scale(Vec3(1, 1 + 0.5*math.sin(self.time), 1))

            #self.program['refractive_index'] = 1.52 + 0.5 * math.cos(self.time)

        with self.ubo as ubo:
            ubo.view_position = self.camera.position
            ubo.time = self.time

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
            case pyglet.window.key.K:
                for mesh in self.meshes:
                    mesh.freeze()
            case pyglet.window.key.J:
                for mesh in self.meshes:
                    mesh.unfreeze()
            case pyglet.window.key.F:
                self.program['fresnel'] = not self.program['fresnel']

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons == pyglet.window.mouse.RIGHT:
            self.material_group.bump_strength += dy*0.02
        if buttons == pyglet.window.mouse.LEFT:
            if modifiers & pyglet.window.key.LSHIFT:
                self.mesh_y += dy
            else:
                self.mesh_x += dx
                self.mesh_z -= dy
            for mesh in self.meshes:
                mesh.matrix = Mat4.from_translation(Vec3(self.mesh_x, self.mesh_y, self.mesh_z))


if __name__ == '__main__':
    app = start_app(App, settings)
