from pyglet.event import EVENT_HANDLE_STATE
from pyglet.graphics import Batch
from pyglet.graphics.vertexdomain import VertexList
from pyglet.math import Vec3, Vec4

import tools.skybox
from tools.definitions import *
from tools.camera import Camera3D
from tools.model import Sphere, BlendGroup, Group, GridMesh
from pyglet.gl import *
from tools.interface import SelectionManager, Selectable, MousePicker

SETTINGS = {
    'default_mode': False,
    'width': 1920,
    'height': 1080,
    'config': get_config(samples=4),
    'vsync': True,
    'fullscreen': False,
    'resizable': True,
    'fps': 100,
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


class WavelineGroup(Group):
    def __init__(self, wave_id: int, program: ShaderProgram, parent: Group | None = None, order=0):
        super().__init__(parent=parent, order=order)
        self.id = wave_id
        self.program = program

    def set_state(self) -> None:
        self.program.use()
        self.program['wave_id'] = self.id

    def __hash__(self):
        return hash((self.id, self.program, self.parent))

    def __eq__(self, other: "WavelineGroup"):
        return (self.__class__ is other.__class__ and
                self.id == other.id and
                self.program == other.program
        )


class WaveSource:
    """Source of the wave to drive shader calculations.
    Sphere model is added as a visual representation.
    TODO: Sphere vibration not synced with the actual wave.
    """
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
            model_color=(255, 255, 255, 255),
            **kwargs  # additional kwargs to be passed to a Sphere
    ):
        self._uniform_name = shader_uniform_name
        self.id = wave_id
        self._position = position
        self._program = program
        self._batch = batch
        self._group = NoTransformationGroup(self.id, self._program, group)
        self._frequency = frequency
        self._amplitude = amplitude
        self._wavelength = wavelength
        self.active = False
        self._model_color = model_color

        self._model = Sphere(self._program, self._batch, self._group, self._position, color=self._model_color, **kwargs)

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


class WaveLine:
    """A line strip to showing shape of the wave,
    from wave source to selected position."""

    _waveline_vert_src = """
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec4 color;
    
    out vec3 frag_position;
    out vec4 frag_color;
    
    const float speed = 150.0;
    const float pi = 3.1416;
    uniform float time = 0.0;
    
    uniform WindowBlock{
    mat4 projection;
    mat4 view;
} window;

    // wave declarations
    uniform vec3 wave0;
    uniform vec3 wave1;
    uniform vec3 wave2;
    uniform vec3 wave3;
    uniform int wave_id;
    
    // wave data
    vec3 wave_positions[] = vec3[](wave0, wave1, wave2, wave3);
    uniform vec4 wave_amplitudes = vec4(20, 25,30, 10);
    uniform vec4 wave_frequencies = vec4(0.5, 0.5, 0.75, 1.0);
    uniform vec4 wave_phases = vec4(0, pi, 0, 0);
    uniform vec4 source_activity_map = vec4(0.0);
    
    float wave_displacement(vec3 pos) {
        float y = 0.0;
        int i = wave_id;
        
        vec3 wave_position = wave_positions[i];
        float A = wave_amplitudes[i];
        float f = wave_frequencies[i];
        float lambda = speed/f;
        float p = wave_phases[i];
        
        float k = 2.0 * pi / lambda;
        float w = 2.0 * pi * f;
    
        float r = length(pos - wave_position);
        y = A * sin(k * r - w * time + p);
    
        return y;
    }
    
    void main() {
        float displacement = wave_displacement(position);
        vec3 new_position = position + vec3(0, displacement, 0);
        vec4 transformed =  window.projection * window.view * vec4(new_position, 1.0);
        gl_Position = transformed;
        frag_position = position;
        frag_color = color;
    }
    """

    _waveline_frag_src = """
    #version 330 core
    in vec3 frag_position;
    in vec4 frag_color;
    
    out vec4 final_color;
    
    uniform float time;
    
    void main() {
        final_color = frag_color;
    }
    """

    line_program = ShaderProgram(
        Shader(_waveline_vert_src, 'vertex'),
        Shader(_waveline_frag_src, 'fragment')
    )

    def __init__(
            self,
            wave_source: WaveSource,
            batch: Batch,
            destination: Vec3,
            steps: int = 200,
    ):
        self.wave_source = wave_source
        self.batch = batch
        self.group = WavelineGroup(self.wave_source.id, self.line_program, order=0)
        self.dest = destination
        self.steps = steps
        self.color = self.wave_source._model_color

        self.vertex_list = self._generate_vertex_list()

    # def _generate_vertex_list(self) -> VertexList:
    #     # line vector
    #     l = self.dest - self.wave_source.position
    #     # increments
    #     dx = l.x / self.steps
    #     dz = l.z / self.steps
    #
    #     x, y, z = self.wave_source.position.x, 0, self.wave_source.position.z
    #     points = [x, y, z]
    #
    #     for i in range(self.steps):
    #         x += dx
    #         z += dz
    #         points.extend((x, y, z, x, y, z))
    #
    #     count = len(points)//3
    #
    #     return self.program.vertex_list(
    #         count=count,
    #         mode=GL_LINES,
    #         batch=self.batch,
    #         group=self.group,
    #         position=('f', points),
    #         color=('Bn', self.color * count)
    #     )


    def _generate_vertex_list(self, width=5.0, height=5.0) -> VertexList:
        l = self.dest - self.wave_source.position
        dx = l.x / self.steps
        dz = l.z / self.steps

        positions = []
        colors = []

        x, y, z = self.wave_source.position.x, 0.0, self.wave_source.position.z

        for _ in range(self.steps):
            p0 = Vec3(x, y, z)
            p1 = Vec3(x + dx, y, z + dz)

            forward = (p1 - p0).normalize()
            world_up = Vec3(0, 1, 0)

            right = forward.cross(world_up)
            if right.length() < 1e-6:
                right = Vec3(1, 0, 0)
            right = right.normalize()

            up = right.cross(forward).normalize()

            hw = width * 0.75
            hh = height * 0.75

            c0 = [
                p0 + right * hw + up * hh,
                p0 - right * hw + up * hh,
                p0 - right * hw - up * hh,
                p0 + right * hw - up * hh,
            ]

            c1 = [
                p1 + right * hw + up * hh,
                p1 - right * hw + up * hh,
                p1 - right * hw - up * hh,
                p1 + right * hw - up * hh,
            ]

            verts = c0 + c1

            faces = [
                (0, 1, 5), (5, 4, 0),
                (1, 2, 6), (6, 5, 1),
                (2, 3, 7), (7, 6, 2),
                (3, 0, 4), (4, 7, 3),
            ]

            for a, b, c in faces:
                for v in (verts[a], verts[b], verts[c]):
                    positions.extend((v.x, v.y, v.z))
                    colors.extend(self.color)

            x += dx
            z += dz

        count = len(positions) // 3

        return self.line_program.vertex_list(
            count=count,
            mode=GL_TRIANGLES,
            batch=self.batch,
            group=self.group,
            position=('f', positions),
            color=('f', colors)
        )


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
        self.mouse_picker = MousePicker(self, self.camera)

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

        self.main_shader = pyglet.graphics.ShaderGroup(self.program)
        self.line_shader_group = pyglet.graphics.ShaderGroup(WaveLine.line_program)
        self.blend_group = BlendGroup(parent=self.main_shader, order=1)

        self.default_shader = get_default_shader_program()
        self.default_group = pyglet.graphics.ShaderGroup(self.default_shader)

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
                program=self.program, batch=self.batch, group=self.blend_group,
                model_color=(1, 0.1, 0, 1.0), lat_segments=16, lng_segments=16, dynamic=True
            ) for i in range(self.no_waves)
        ]

        for wave_source, pos in zip(self.wave_sources, self.wave_source_positions):
            wave_source.position = pos
            WaveLine.line_program[wave_source._uniform_name] = pos
            self.selectables.add(wave_source.model)

        self.toggle_source(self.wave_sources[0])
        self.toggle_source(self.wave_sources[1])

        self.wave_lines = []

        # Mesh
        self.grid = GridMesh(
            self.program, self.batch, position=Vec3(-2000, 0, -2000), group=self.blend_group,
            length=4000, width=4000, columns=250, rows=250,
        )

        glEnable(GL_DEPTH_TEST)

    def toggle_source(self, wave_source: WaveSource):
        """TODO: move to WaveSource?"""
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

        WaveLine.line_program['time'] = self.run_time

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

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        if button == pyglet.window.mouse.RIGHT:
            ray = self.mouse_picker.get_mouse_ray()
            hit_point = ray.intersect_plane()
            hit_x, hit_z = hit_point.x, hit_point.z
            for wave_source in self.wave_sources:
                if wave_source.active:
                    self.wave_lines.append(WaveLine(
                        wave_source, self.batch, Vec3(hit_x, 0, hit_z)
                    ))


if __name__ == '__main__':
    start_app(MyApp, **SETTINGS)




