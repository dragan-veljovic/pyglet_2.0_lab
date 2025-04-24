"""
A rendered skybox made of 6 textured quads, and integrated with main camera, excluding it's translation.
TODO: Incorporate get_skybox, Skybox texture group, and shaders into a class for efficient instancing.
"""
import pyglet.math
import numpy as np
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    'default_mode': True,
    'width': 1280,
    'height': 720,
    'fps': 60
}

vertex_source = """
#version 330 core
in vec3 position;
in vec4 colors;

out vec3 frag_position;
out vec4 frag_colors;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

uniform float time; // Time uniform for animation
uniform vec3 center;  // wave source center position
uniform bool apply_fading = false;

void main() {
    // Create a copy of the position that we can modify to calculate resultant displacement
    vec3 animated_position = position;

    // Wave 1
    vec3 center = vec3(750, 0, 0);
    float distance = length(vec2(position.x - center.x, position.z - center.z));
    // Create expanding circular wave
    float waveSpeed = 2;
    float wavelength = 100.0;
    float waveAmplitude = 100.0;
    float frequency = 1.0;
    // Expanding ring wave formula
    float wave = sin(distance / wavelength - time * frequency * waveSpeed) * waveAmplitude;

    // square wave
    // float wave = sign(sin(position.x / wavelength - time * frequency * waveSpeed)) * waveAmplitude;


    // Wave 2
    vec3 center2 = vec3(-750, 0, 0);
    float distance2 = length(vec2(position.x - center2.x, position.z - center2.z));
    // Create expanding circular wave
    float waveSpeed2 = 2;
    float wavelength2 = 100.0;
    float waveAmplitude2 = 100.0;
    float frequency2 = 1.0;
    // Expanding ring wave formula
    float wave2 = sin(distance2 / wavelength2 - time * frequency2 * waveSpeed2) * waveAmplitude2;

    // Apply fade-out based on distance from center for a more natural look
    if (apply_fading) {
        float fadeDistance = 4000.0; // Adjust this to change where the wave starts to fade
        float fade = max(0.0, 1.0 - 0.5 * distance / fadeDistance);
        float fade2 = max(0.0, 1.0 - distance2 / fadeDistance);
        wave *= fade;
        wave2 *= fade2;
    }

    float superposition = wave + wave2;

    // Apply the wave to y-coordinate
    animated_position.y += superposition;

    // Color modulation based on wave height for visual effect
    float color_factor = superposition / waveAmplitude * 2; 
    frag_colors = vec4(colors.r * color_factor, colors.gb, 1.0);

    gl_Position = window.projection * window.view * vec4(animated_position, 1.0);
    frag_position = animated_position;
}
"""

fragment_source = """
#version 330 core

in vec3 frag_position;
in vec4 frag_colors;

out vec4 final_colors;

uniform vec3 camera_pos = vec3(0.0);
uniform bool fade_out = true;

float get_fade_factor(){
    float fade_start = 4500.0;
    float fade_end = 5000.0;
    float distance = length(frag_position - camera_pos);
    return clamp(1.0 - (distance - fade_start) / (fade_end - fade_start), 0.0, 1.0);
}

void main() {
    // fade out effect
    float fade = fade_out ? get_fade_factor() : 1.0;

    final_colors = vec4(frag_colors.rgb, frag_colors.a * fade);
}
"""

default_vert = """
    #version 330 core
    in vec3 position;
    in vec2 tex_coords;
    in vec4 colors;

    out vec4 frag_colors;
    out vec2 frag_coords;

    uniform float time;

    uniform WindowBlock
        {
            mat4 projection;
            mat4 view;
        } window;

    uniform mat4 sky_view;

    void main() {
        vec3 new_position = position;
        gl_Position = window.projection * sky_view * vec4(new_position, 1.0);
        frag_colors = colors;
        frag_coords = tex_coords;
    }
"""

default_frag = """
    #version 330 core
    in vec4 frag_colors;
    in vec2 frag_coords;
    out vec4 final_colors;

    uniform float time;
    uniform sampler2D my_texture;

    void main() {
        vec4 diff_texture = texture(my_texture, frag_coords);
        final_colors = vec4(diff_texture.rgb, frag_colors.a);
    }
"""


def get_quantized_grid_gl_line_vertices(span: 2000, pitch=0, yaw=0, roll=0) -> np.array:
    grid_width = span
    grid_height = span
    spacing = 50

    nh = int(grid_height / spacing)
    nv = int(grid_width / spacing)

    points = []
    vertices = []

    for i in range(-nh, nh):
        for j in range(-nv, nv):
            points.append([i * spacing, j * spacing, 0])
        rotated_points = rotate_points(points, pitch, yaw, roll)
        vertices_this_pass = get_gl_lines_vertices_numpy(rotated_points)
        vertices.extend(vertices_this_pass)
        points = []
    #
    for i in range(-nh, nh):
        for j in range(-nv, nv):
            points.append([j * spacing, i * spacing, 0])
        rotated_points = rotate_points(points, pitch, yaw, roll)
        vertices_this_pass = get_gl_lines_vertices_numpy(rotated_points)
        vertices.extend(vertices_this_pass)
        points = []

    return vertices


class SkyboxTextureGroup(TextureGroup):
    def __init__(
            self,
            texture: Texture,
            program: ShaderProgram,
            camera: Camera3D,
            order=0, parent=None
    ):
        """
        Skybox shares camera orientation with the rest of the scene, but not its translation.
        Here we use scene's camera orientation to create and pass a new "sky_view" uniform to the shader,
        that will be used instead of a window view.
        """
        super(SkyboxTextureGroup, self).__init__(texture, program, order, parent)
        self.camera = camera
        self.position = Vec3(0, 0, 0)

    def set_state(self):
        super(SkyboxTextureGroup, self).set_state()
        self.program['sky_view'] = pyglet.math.Mat4.look_at(self.position, self.camera._front, self.camera._up)


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        # center_window(self)
        set_background_color(255, 0, 0, 0)
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self, z_far=75_000)
        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        # Time tracking for animation
        self.time = 0.0
        self.run = True

        self.skybox = self.get_skybox(self.batch, path='res/skybox1/', ext='png')
        self.cube = Cuboid(self.program, self.batch, color=(128, 0, 0, 255), size=(100, 100, 100), position=(750, 0, 0))
        self.cube2 = Cuboid(self.program, self.batch, color=(128, 0, 0, 255), size=(100, 100, 100),
                            position=(-750, 0, 0))

        grid_vertices = get_quantized_grid_gl_line_vertices(span=5000, pitch=np.pi / 2)
        self.grid = self.program.vertex_list(len(grid_vertices) // 3, GL_LINES, self.batch,
                                             position=('f', grid_vertices),
                                             colors=('Bn', (92, 92, 92, 255) * (len(grid_vertices) // 3)))

        self.sphere = pyglet.model.Sphere(100, batch=self.batch)
        # Enable transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending

    def get_skybox(self, batch, path, ext='png', size=10_000):
        faces = ('right', 'left', 'top', 'bottom', 'front', 'back')

        # custom shader program for skybox
        program = ShaderProgram(
            Shader(default_vert, 'vertex'),
            Shader(default_frag, 'fragment'),
        )

        # loading and preparing textures
        textures = []
        groups = []
        for face in faces:
            # load texture with default parameters
            texture = pyglet.image.load(path + f'{face}.{ext}').get_texture()
            # Bind the texture and set filtering/wrapping options
            glBindTexture(texture.target, texture.id)
            # Ensure edges don’t interpolate weirdly
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            # Fix any possible filtering issues (optional)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            # Unbind for safety
            glBindTexture(texture.target, 0)
            # append prepared texture
            textures.append(texture)

            # creating texture group
            groups.append(SkyboxTextureGroup(texture, program, self.camera))

        # Quad positions and rotations
        half = size / 2
        positions = (
            (half, -half, -half),  # right
            (-half, -half, half),  # left
            (-half, half, -half),  # top
            (-half, -half, half),  # bottom
            (-half, -half, -half),  # front
            (half, -half, half),  # back
        )
        rotations = (
            (0, np.pi / 2, 0),  # right
            (0, -np.pi / 2, 0),  # left
            (np.pi / 2, 0, 0),  # top
            (-np.pi / 2, 0, 0),  # bottom
            (0, 0, 0),  # front
            (0, -np.pi, 0),  # back

        )
        # creating textured planes
        planes = [
            TexturedPlane(
                position=positions[i], batch=batch, group=groups[i], program=program,
                length=size, height=size, rotation=rotations[i])
            for i in range(6)
        ]

        return planes
    
    # garbage here, remove it
    def create_cube_map(self):
        self.cube_map = GLuint()
        glGenTextures(1, self.cube_map)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cube_map)
        cube_targets = (
            GL_TEXTURE_CUBE_MAP_POSITIVE_X,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
        )

        self.tex = pyglet.image.load('res/Jupiter.jpg')

        for target in cube_targets:
            glTexImage2D(
                target,
                0, GL_RGBA8, self.tex.width, self.tex.height, 0, GL_RGBA8, GL_FLOAT, self.tex
            )

        # wrapping and filtering
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def update(self, dt):
        # Update the time value
        self.time += dt
        # Pass the time value to the shader program
        with self.program:
            self.program['time'] = self.time
            self.program['camera_pos'] = self.camera.position

    def on_draw(self) -> None:
        if self.run:
            self.update(1 / settings['fps'])

        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.SPACE:
                self.run = not self.run


if __name__ == '__main__':
    start_app(App, settings)