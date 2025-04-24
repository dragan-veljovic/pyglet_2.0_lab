"""
A rendered skybox made of 6 textured quads, and integrated with main camera, excluding it's translation.
TODO: Incorporate get_skybox, Skybox texture group, and shaders into a class for efficient instancing.
"""
import ctypes

import pyglet.math
import numpy as np
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    'default_mode': False,
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

# Skybox Vertex Shader
skybox_vertex_shader = """
#version 330 core
in vec3 position;
out vec3 texCoords;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

uniform mat4 scale_matrix;

void main() {
    texCoords = position;

    // Remove translation from the view matrix to keep skybox centered on camera
    mat4 viewNoTranslation = mat4(mat3(window.view));

    // The skybox will always be at the far plane
    vec4 pos = window.projection * viewNoTranslation * scale_matrix * vec4(position, 1.0);

    // Ensure skybox is always at the far plane (z/w = 1.0)
    gl_Position = pos.xyww;
}
"""

# Skybox Fragment Shader
skybox_fragment_shader = """
#version 330 core
in vec3 texCoords;
out vec4 fragColor;

uniform samplerCube skybox;

void main() {
    fragColor = texture(skybox, texCoords);
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


def create_cube_map(path: str, ext: str = 'jpg') -> GLuint:
    cube_map = GLuint()
    glGenTextures(1, cube_map)
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map)

    face_targets = (
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    )

    # Fixed mapping - swapped 'top' and 'bottom'
    face_images = ('right', 'left', 'top', 'bottom', 'front', 'back')

    for target, image in zip(face_targets, face_images):
        # loading image
        image = pyglet.image.load(path + f'{image}.{ext}').get_texture()
        image_data = image.get_image_data()
        # get the raw pixel bytes (RGBA format)
        raw_data = image_data.get_bytes('RGBA', image.width * 4)  # 4 bytes per pixel
        glTexImage2D(target, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw_data)

    # wrapping and filtering
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return cube_map  # Return the cube map texture ID


class Skybox:
    def __init__(self, shader_program, cube_map_texture):
        # Store the shader program and cube map texture
        self.shader = shader_program
        self.cube_map = cube_map_texture

        # Create the skybox vertices (a simple cube centered at origin)
        # Each vertex is just the direction vector from center to the vertex
        vertices = [
            # positions
            -1.0, 1.0, -1.0,
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, 1.0, -1.0,
            -1.0, 1.0, -1.0,

            -1.0, -1.0, 1.0,
            -1.0, -1.0, -1.0,
            -1.0, 1.0, -1.0,
            -1.0, 1.0, -1.0,
            -1.0, 1.0, 1.0,
            -1.0, -1.0, 1.0,

            1.0, -1.0, -1.0,
            1.0, -1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, -1.0,
            1.0, -1.0, -1.0,

            -1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, -1.0, 1.0,
            -1.0, -1.0, 1.0,

            -1.0, 1.0, -1.0,
            1.0, 1.0, -1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            -1.0, 1.0, 1.0,
            -1.0, 1.0, -1.0,

            -1.0, -1.0, -1.0,
            -1.0, -1.0, 1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            -1.0, -1.0, 1.0,
            1.0, -1.0, 1.0
        ]

        # Create the vertex array and buffer
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)

        self.vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Convert vertices to ctypes array and upload to GPU
        vertices_array = (GLfloat * len(vertices))(*vertices)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(vertices_array), vertices_array, GL_STATIC_DRAW)

        # Configure vertex attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(GLfloat), 0)

        # Unbind
        glBindVertexArray(0)

        # fixing inverted skybox rendering
        scale_matrix = pyglet.math.Mat4.from_scale(Vec3(-1, -1, -1))
        self.shader['scale_matrix'] = scale_matrix

    def draw(self):
        # Save the current front face setting
        frontface = GLint()
        glGetIntegerv(GL_FRONT_FACE, ctypes.byref(frontface))

        if frontface.value == GL_CCW:
            glFrontFace(GL_CW)
        else:
            glFrontFace(GL_CCW)

        # Save OpenGL state
        depth_func = GLint()
        glGetIntegerv(GL_DEPTH_FUNC, ctypes.byref(depth_func))

        # Change depth function so depth test passes when values are equal to depth buffer's content
        glDepthFunc(GL_LEQUAL)

        # Activate shader
        self.shader.use()

        # Bind skybox texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cube_map)
        # Set the skybox uniform to texture unit 0
        self.shader["skybox"] = 0

        # Draw the skybox cube
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        # Restore OpenGL state
        glDepthFunc(depth_func.value)


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        # center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.camera = Camera3D(self, z_far=75_000)

        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        # Create skybox shader program
        self.skybox_program = ShaderProgram(
            Shader(skybox_vertex_shader, 'vertex'),
            Shader(skybox_fragment_shader, 'fragment')
        )

        self.cube_map_texture = create_cube_map('res/skybox2/', "jpg")

        self.skybox = Skybox(self.skybox_program, self.cube_map_texture)

        # Time tracking for animation
        self.time = 0.0
        self.run = True
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
        self.skybox.draw()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.SPACE:
                self.run = not self.run


if __name__ == '__main__':
    start_app(App, settings)