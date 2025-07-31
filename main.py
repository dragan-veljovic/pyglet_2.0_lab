import pyglet.graphics

import tools.lighting
from tools.definitions import *
from tools.model import *
from tools.camera import Camera3D
from tools.lighting import DirectionalLight
from tools.skybox import Skybox
from tools.interface import *

import pyglet

settings = {
    'default_mode': False,
    'width': 1920,
    'height': 1080,
    'config': get_config(samples=2),
    'vsync': False,
    'fullscreen': False,
    'resizable': True
}


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        center_window(self)

        pyglet.resource.path.append('res')
        pyglet.resource.reindex()
        self.batch = pyglet.graphics.Batch()

        self.program = ShaderProgram(
            pyglet.resource.shader('res/shaders/dev.vert', 'vertex'),
            pyglet.resource.shader('res/shaders/dev.frag', 'fragment')
        )

        self.camera = Camera3D(self, z_far=12000, speed=10)
        self.mouse_picker = MousePicker(self, self.camera, batch=self.batch)
        self.ubo = self.program.uniform_blocks['WindowBlock'].create_ubo()

        with self.ubo as ubo:
            ubo.projection = self.projection
            ubo.view = self.view
            ubo.z_far = self.camera.z_far
            ubo.fade_length = 2000

        self.time = 0.0
        self.clock = pyglet.clock.Clock()
        self.start_time = self.clock.time()

        self.light = DirectionalLight(position=Vec3(1, 1, 1))
        self.light.bind_to_block(self.program.uniform_blocks['LightBlock'])
        self.skybox = Skybox('res/textures/skybox')
        self.skybox.set_environment_map(self.program)

        shader_group = pyglet.graphics.ShaderGroup(self.program)
        blend_group = BlendGroup(parent=shader_group)

        # terrain_texture = DiffuseNormalTextureGroup(
        #     diffuse=pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
        #     normal=pyglet.image.load(
        #        'res/textures/textures/painted_plaster_wall_nor_gl_1k.jpg').get_texture(),
        #     program=self.program, parent=blend_group
        # )

        material_texture = DiffuseNormalTextureGroup(
            pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            normal=pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            program=self.program, parent=blend_group
        )

        terrain_texture = material_texture

        self.material_ubo = self.program.uniform_blocks['MaterialBlock'].create_ubo()
        self.material_group = MaterialGroup(self.material_ubo, parent=terrain_texture, reflection_strength=0.5, shininess=4, bump_strength=1, f0_reflectance=0.04, specular=0.2, diffuse=0.10)

        terrain_group = MaterialGroup(self.material_ubo, parent=terrain_texture, shininess=128, bump_strength=0.2, specular=0, reflection_strength=0.0)
        #self.terrain = load_mesh('res/model/terrain/models/mars/terrain/mars.OBJ', self.program, self.batch, terrain_group, True, position=(10000, -5000, -10000), scale=10, tex_scale=5)

        self.material_group2 = MaterialGroup(self.material_ubo, parent=terrain_texture, reflection_strength=1, shininess=128, bump_strength=0.25, f0_reflectance=(0.839, 0.565, 0.255, 1.0))
        self.meshes = [
            load_mesh(
                'res/model/vessel.obj', self.program, self.batch, self.material_group2,
                position=(i*300 + 300, 0, 0), scale=0.5, tex_scale=3, add_tangents=True
            ) for i in range(3)
        ]

        self.meshes.append(Sphere(self.program, self.batch, self.material_group, Vec3(0, 0, 0), radius=150, lat_segments=32))
        self.sphere1 = self.meshes[-1]
        self.sphere1.rotation = 0.0

        # must be implemented in Mesh
        self.mesh_x = 0.0
        self.mesh_y = 0.0
        self.mesh_z = 0.0
        self.rotation = 0.0

        self.program['shadow_mapping'] = False
        self.draw_skybox = True
        self.wireframe = False
        self.run = True

        glEnable(GL_DEPTH_TEST)
        self.camera.toggle_fps_controls()
        self.bounding_box = None
        self.selected = None

        # print(self.batch.group_children)

    def on_activate(self):
        self.camera.toggle_fps_controls()

    def on_deactivate(self):
        self.camera.toggle_fps_controls()

    def select(self):
        min_dist = float('inf')
        closest = None

        for mesh in self.meshes:
            ray = self.mouse_picker.get_mouse_ray()
            bbox = mesh.get_bounding_box()
            hit, dist = ray_intersects_aabb(ray._origin, ray._direction, bbox)
            if hit and dist < min_dist:
                closest = mesh
                min_dist = dist

        if closest:
            if self.bounding_box:
                self.bounding_box.delete()

            self.bounding_box = closest.get_bounding_box()
            self.bounding_box.draw(batch=self.batch)
            self.selected = closest

    def on_draw(self):
        if self.run:
            self.time = self.clock.time() - self.start_time
            self.sphere1.rotation += 0.01
            ### Create meshes with dynamic positioning, don't use transformed data!
            self.meshes[0].rotation_dir = ...


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
            self.rotation += dy * 0.1
        if buttons == pyglet.window.mouse.LEFT:
            if modifiers & pyglet.window.key.LSHIFT:
                self.mesh_y += dy
            else:
                self.mesh_x += dx
                self.mesh_z -= dy

            mesh = self.selected
            if mesh:
                if mesh.position:
                    mesh.position += Vec3(dx, 0.0, -dy)
                else:
                    mesh.position = Vec3(dx, 0.0, -dy)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        if button == pyglet.window.mouse.LEFT:
            self.select()

        if button == pyglet.window.mouse.MIDDLE:
            box = self.meshes[0].get_bounding_box()
            box.draw(batch=self.batch)

if __name__ == '__main__':
    app = start_app(App, fps=120, enable_console=True, **settings)



