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
        self.camera = Camera3D(self, z_far=100_000, speed=5)
        self.program['z_far'] = self.camera.z_far
        self.time = 0.0
        self.clock = pyglet.clock.Clock()
        self.start_time = self.clock.time()
        self.draw_skybox = True
        self.wireframe = False
        self.run = True

        self.light = DirectionalLight()
        self.light.bind_to_block(self.program.uniform_blocks['LightBlock'])
        self.skybox = Skybox('res/textures/skybox')
        self.skybox.set_environment_map(self.program)

        shader_group = pyglet.graphics.ShaderGroup(self.program)
        blend_group = BlendGroup(parent=shader_group)

        texture_group = DiffuseNormalTextureGroup(
            diffuse=pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            normal=pyglet.image.load(
               'res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            program=self.program, parent=blend_group
        )

        self.plane = GridMesh(self.program, self.batch, 1000, 1000, columns=100, rows=100, group=texture_group)

        self.cube = Cuboid(self.program, self.batch, texture_group, position=Vec3(0, 1000, 0), size=(300, 300, 300))

        # loading OBJ model procedure
        model_data = load_obj_model('res/model/vessel.obj')
        self.meshes = [Mesh(
            transform_model_data(model_data.copy(), tex_scale=4, scale=0.5, position=(i * 400, 0, 0)),
            self.program, self.batch, texture_group, dynamic=True)
            for i in range(3)]

        self.mesh_x = 0.0
        self.mesh_y = 0.0

        # self.program['rendering_dynamic_object'] = True
        self.program['shadow_mapping'] = False
        self.program['environment_mapping'] = False
        self.program['normal_mapping'] = False

        glEnable(GL_DEPTH_TEST)

    def on_draw(self):
        if self.run:
            self.time = self.clock.time() - self.start_time
            self.plane.matrix = get_model_matrix(Vec3(0, 0, 0), self.time*0.1, rotation_dir=Vec3(0, 1, 0).normalize())
            self.cube.matrix = self.plane.matrix

            #self.mesh.matrix = pyglet.math.Mat4.from_rotation(-math.sin(self.time), Vec3(0, 1, 0)) @ Mat4.from_scale(Vec3(1, 1 + 0.5*math.sin(self.time), 1))

            self.program['refractive_index'] = 1.52 + 0.5 * math.cos(self.time)

        self.program['view_position'] = self.camera.position
        #self.program['time'] = self.time*0.1
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

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        if buttons == pyglet.window.mouse.LEFT:
            self.mesh_x += dx
            self.mesh_y += dy
            for mesh in self.meshes:
                mesh.matrix = Mat4.from_translation(Vec3(self.mesh_x, 0, -self.mesh_y))

if __name__ == '__main__':
    app = start_app(App, settings)
