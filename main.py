from pyglet.graphics import Batch
from pyglet.math import Vec3, Vec4

import tools.skybox
from tools.definitions import *
from tools.camera import Camera3D
from tools.model import Sphere, BlendGroup, Group, GridMesh
from pyglet.gl import *
from tools.interface import SelectionManager, Selectable

SETTINGS = {
    'default_mode': False,
    'width': 1920,
    'height': 1080,
    'config': get_config(samples=2),
    'vsync': True,
    'fullscreen': False,
    'resizable': True,
    'fps': 120,
    'enable_console': False
}


class NoTransformationGroup(Group):
    """Don't apply wave transformation on these objects"""
    def __init__(self, wave_id: int, program: ShaderProgram, parent: Group | None = None):
        super().__init__(parent=parent)
        self.id = wave_id
        self.program = program

    def set_state(self) -> None:
        self.program['wavy_transformation'] = False
        self.program['wave_id'] = self.id

    def unset_state(self) -> None:
        self.program['wavy_transformation'] = True

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: "NoTransformationGroup"):
        return (self.__class__ is other.__class__ and
                self.parent == other.parent and
                self.id == other.id)


class WaveSource:
    def __init__(
            self,
            shader_uniform_name: str,
            wave_id: int,
            program: ShaderProgram,
            batch: Batch,
            group: Group = None,
            position=Vec3(0, 0, 0),
            frequency=0.5,
            amplitude=25,
            wavelength=300,
            **kwargs  # additional kwargs to be passed to a Sphere
    ):
        self._uniform_name = shader_uniform_name
        self._id = wave_id
        self._position = position
        self._program = program
        self._batch = batch
        self._group = NoTransformationGroup(self._id, self._program, group)
        self._frequency = frequency
        self._amplitude = amplitude
        self._wavelength = wavelength
        self.active = False

        self._model = Sphere(self._program, self._batch, self._group, self._position, **kwargs)

        # link WaveSource instance, so that Mesh knows to which wave source it belongs to
        self._model.wave_source = self

        self._update_shader_uniform()

    def _update_shader_uniform(self):
        self._program[self._uniform_name] = self._position

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: Vec3):
        self._position = value
        self._model.position = self._position
        self._update_shader_uniform()

    @property
    def model(self):
        return self._model


class MyApp(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        center_window(self)
        pyglet.resource.path.append('res')
        pyglet.resource.reindex()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self)

        self.selectables : [..., Selectable] = set()
        self.selection = SelectionManager(self, self.camera, self.selectables, self.batch)

        self.skybox = tools.skybox.Skybox('res/textures/skyboxes/evening.png')
        self.draw_skybox = True

        self.program = ShaderProgram(
            pyglet.resource.shader('shaders/PBR.vert', 'vertex'),
            pyglet.resource.shader('shaders/PBR.frag', 'fragment'),
        )

        self.clock = pyglet.clock.Clock()
        self.start_time = self.clock.time()
        self.run_time = 0.0
        self.run = True

        self.ubo = self.program.uniform_blocks['WindowBlock'].create_ubo()
        with self.ubo as ubo:
            ubo.projection = self.projection
            ubo.view = self.view
            ubo.view_position = self.camera.position
            ubo.time = self.start_time

        shader_group = pyglet.graphics.ShaderGroup(self.program)
        blend_group = BlendGroup(parent=shader_group)

        # scene
        self.no_waves = 4
        self.program['no_waves'] = self.no_waves
        self.wave_limit = 4

        self.wave_source_uniform_names = ('wave0', 'wave1', 'wave2', 'wave3')
        self.wave_source_positions = (
            Vec3(-500, 100, -750),
            Vec3(500, 100, -750),
            Vec3(1000, 100, 1000),
            Vec3(-1000, 100, 1000),
        )

        self.wave_sources = [
            WaveSource(
                wave_id=i, shader_uniform_name=self.wave_source_uniform_names[i],
                program=self.program, batch=self.batch, group=blend_group,
                color=(1, 0.1, 0, 1.0), lat_segments=16, lng_segments=16, dynamic=True
            ) for i in range(self.no_waves)
        ]

        for wave_source, pos in zip(self.wave_sources, self.wave_source_positions):
            wave_source.position = pos
            self.selectables.add(wave_source.model)

        self.toggle_source(self.wave_sources[0])
        self.toggle_source(self.wave_sources[1])

        # Mesh
        self.grid = GridMesh(
            self.program, self.batch, position=Vec3(-2000, 0, -2000), group=blend_group,
            length=4000, width=4000, columns=250, rows=250,
        )

        vlist = self.wave_sources[0].model._vertex_list
        glEnable(GL_DEPTH_TEST)

    def toggle_source(self, wave_source: WaveSource):
        wave_source.active = not wave_source.active
        # create active truth list
        activity = [0.0, 0.0, 0.0, 0.0]
        for i in range(len(self.wave_sources)):
            wave_source = self.wave_sources[i]
            if wave_source.active:
                activity[i] = 1.0
        # create vec4 from truth list and update shader
        vector = Vec4(activity[0], activity[1], activity[2], activity[3])
        self.program['source_activity_map'] = vector

    def update_time(self):
        self.run_time += 1/SETTINGS['fps']

    def on_draw(self):
        if self.run:
            self.update_time()
        # update camera position
        with self.ubo as ubo:
            ubo.view_position = self.camera.position
            ubo.time = self.run_time

        self.clear()

        if self.draw_skybox:
            self.skybox.draw()
        self.batch.draw()

    def on_activate(self):
        self.camera.toggle_fps_controls()

    def on_deactivate(self):
        self.camera.toggle_fps_controls()

    def on_key_press(self, symbol: int, modifiers: int):
        super().on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.V:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        elif symbol == pyglet.window.key.B:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        elif symbol == pyglet.window.key.R:
            self.run = not self.run
        elif symbol == pyglet.window.key.UP:
            if self.no_waves < self.wave_limit:
                self.no_waves += 1
                self.program['no_waves'] = self.no_waves
        elif symbol == pyglet.window.key.DOWN:
            if self.no_waves > 0:
                self.no_waves -= 1
                self.program['no_waves'] = self.no_waves
        elif symbol == pyglet.window.key.P:
            self.program['wave_phases'] = Vec4(0.0, 3.14, 0.0, 0.0)

        elif symbol == pyglet.window.key.O:
            self.program['wave_phases'] = Vec4(0.0, 0.0, 0.0, 0.0)

        elif symbol == pyglet.window.key.N:
            self.draw_skybox = not self.draw_skybox

        elif symbol == pyglet.window.key._1:
            self.toggle_source(self.wave_sources[0])
        elif symbol == pyglet.window.key._2:
            self.toggle_source(self.wave_sources[1])
        elif symbol == pyglet.window.key._3:
            self.toggle_source(self.wave_sources[2])
        elif symbol == pyglet.window.key._4:
            self.toggle_source(self.wave_sources[3])

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int):
        """Example how to handle selection, in this case some typical transforms. """
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
                            selected.wave_source.position += movement


if __name__ == '__main__':
    start_app(MyApp, **SETTINGS)




