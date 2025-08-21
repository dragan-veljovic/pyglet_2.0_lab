import imgui
from pyglet.event import EVENT_HANDLE_STATE

from tools.definitions import *
from tools.model import *
from tools.lighting import DirectionalLight
from tools.skybox import Skybox
from tools.interface import *
import pyglet
from imgui.integrations.pyglet import create_renderer

SETTINGS = {
    'default_mode': True,
    'width': 1920,
    'height': 1080,
    'config': get_config(samples=2),
    'vsync': True,
    'fullscreen': False,
    'resizable': True,
    'fps': 120,
    'enable_console': True
}


from pyglet.event import EVENT_HANDLED


class ImGuiEventFilter:
    def __init__(self, impl):
        self.impl = impl  # The PygletRenderer instance

    # Mouse events
    def on_mouse_press(self, x, y, button, modifiers):
        self.impl.on_mouse_press(x, y, button, modifiers)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    def on_mouse_release(self, x, y, button, modifiers):
        self.impl.on_mouse_release(x, y, button, modifiers)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.impl.on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.impl.on_mouse_scroll(x, y, scroll_x, scroll_y)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    # Keyboard events
    def on_key_press(self, symbol, modifiers):
        self.impl.on_key_press(symbol, modifiers)
        if imgui.get_io().want_capture_keyboard:
            return EVENT_HANDLED

    def on_key_release(self, symbol, modifiers):
        self.impl.on_key_release(symbol, modifiers)
        if imgui.get_io().want_capture_keyboard:
            return EVENT_HANDLED

    def on_text(self, text):
        self.impl.on_text(text)
        if imgui.get_io().want_capture_keyboard:
            return EVENT_HANDLED


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
        self.selectables = set()
        self.selection = SelectionManager(self, self.camera, self.selectables, self.batch)

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
        self.skybox = Skybox('res/textures/skybox1/skybox2.png')
        self.skybox.set_environment_map(self.program)

        shader_group = pyglet.graphics.ShaderGroup(self.program)
        blend_group = BlendGroup(parent=shader_group)

        self.material_ubo = self.program.uniform_blocks['MaterialBlock'].create_ubo()

        imgui.create_context()
        self.impl = create_renderer(self)
        io = imgui.get_io()
        io.font_global_scale = 1.5
        self.imgui_filter = ImGuiEventFilter(self.impl)
        # if pushed last, this filter disable any mouse action on the GUI down the handler stack with EVENT_HANDLED
        self.push_handlers(self.imgui_filter)

        terrain_texture = DiffuseNormalTextureGroup(
             diffuse=pyglet.image.load('res/model/cliff/base_color.png').get_texture(),
            normal=pyglet.image.load(
                'res/model/cliff/normalmap.png').get_texture(),
            program=self.program, parent=blend_group
        )

        material_texture = DiffuseNormalTextureGroup(
            pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_diff_2k.jpg').get_texture(),
            normal=pyglet.image.load('res/textures/rock_boulder_dry_2k/textures/rock_boulder_dry_nor_gl_2k.jpg').get_texture(),
            program=self.program, parent=blend_group
        )

        terrain_group = MaterialGroup(self.material_ubo, parent=terrain_texture, shininess=128, bump_strength=0.2, specular=0, reflection_strength=0.0)
        self.material_group = MaterialGroup(self.material_ubo, parent=material_texture, reflection_strength=0.5, shininess=4, bump_strength=1, f0_reflectance=0.04, specular=0.2, diffuse=0.10)
        self.material_group2 = MaterialGroup(self.material_ubo, parent=material_texture, reflection_strength=1, shininess=128, bump_strength=0.25, f0_reflectance=0.1)

        # scene
        self.terrain = load_mesh('res/model/cliff/cliff_low.obj', self.program, self.batch, terrain_group, True, position=(0, -1000, 0), scale=10)
        self.selectables.add(self.terrain)

        data = transform_model_data(load_obj_model('res/model/vessel.obj'))

        N = 1
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    mesh = Mesh(data, self.program, self.batch, self.material_group2)
                    mesh.position = Vec3(200 + 200 * i, 200 + k*200, 200 + 200 * j)
                    mesh.scale = Vec3(0.25, 0.25, 0.25)
                    self.selectables.add(mesh)

        self.sphere1 = Sphere(self.program, self.batch, self.material_group, Vec3(0, 0, 0), radius=150, lat_segments=32)
        self.selectables.add(self.sphere1)

        self.sphere2 = Sphere(self.program, self.batch, self.material_group2, Vec3(0, 0, 0), radius=125, lat_segments=32)
        self.sphere2.position = Vec3(-300, 0, 0)
        self.selectables.add(self.sphere2)

        self.program['shadow_mapping'] = False
        self.draw_skybox = True
        self.wireframe = False
        self.run = True

        glEnable(GL_DEPTH_TEST)

    def on_activate(self):
        self.camera.toggle_fps_controls()

    def on_deactivate(self):
        self.camera.toggle_fps_controls()

    def on_draw(self):
        if self.run:
            self.time = self.clock.time() - self.start_time
            if self.sphere1.dynamic:
                self.sphere1.rotation += 0.01

            if self.sphere2.dynamic:
                self.sphere2.scale = Vec3(1 + 0.5 * math.sin(self.time), 1 + 0.5 * math.cos(self.time), 1)

        #self.visualise_actual_bounding_boxes()

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

        # Start new ImGui frame
        imgui.new_frame()
        imgui.show_demo_window()
        imgui.begin(
            "Fixed Window",
            flags=imgui.WINDOW_NO_MOVE
        )
        imgui.text("You can't drag me anywhere!")
        imgui.end()
        # Render ImGui
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def gui_interaction(self) -> bool:
        return imgui.get_io().want_capture_mouse

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super().on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.R:
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
            case pyglet.window.key.F:
                self.program['fresnel'] = not self.program['fresnel']

            case pyglet.window.key.K:
                if self.selection:
                    for mesh in self.selection:
                        mesh.freeze()
            case pyglet.window.key.J:
                if self.selection:
                    for mesh in self.selection:
                        mesh.unfreeze()

            case pyglet.window.key.G:
                ray = self.mouse_picker.get_mouse_ray()
                ray.create_vertex_list()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        if imgui.get_io().want_capture_mouse:
            self.impl.on_mouse_press(x, y, button, modifiers)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        """Example how to handle selection, in this case some typical transforms. """
        if not self.gui_interaction():
            if self.selection:
                for selected in self.selection:
                    if selected.dynamic:
                        if buttons == pyglet.window.mouse.LEFT:
                            if modifiers & pyglet.window.key.LSHIFT:
                                selected.position += Vec3(0.0, dy, 0.0)
                            else:
                                speed = 1  # tweak as needed
                                right = Vec3(self.camera._right.x, 0, self.camera._right.z).normalize()
                                forward = Vec3(self.camera._front.x, 0, self.camera._front.z).normalize()

                                movement = (right * dx + forward * dy) * speed
                                selected.position += movement

                        if buttons == pyglet.window.mouse.RIGHT:
                            selected.rotation += (dy+dx)*0.005
                            selected.rotation_dir = Vec3(1, 1, 0)

                        if buttons == pyglet.window.mouse.MIDDLE:
                            selected.scale += Vec3(dx*0.001, dy*0.001, 0)


if __name__ == '__main__':
    app = start_app(App, **SETTINGS)



