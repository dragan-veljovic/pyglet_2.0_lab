import pyglet
from pyglet.gl import *
import ctypes
from pyglet.math import Vec3
from pyglet.graphics.shader import Shader, ShaderProgram


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


def create_cube_map(path: str, six_images_loading=False, six_images_ext: str = 'jpg') -> GLuint:
    """
    Create a cubemap from a single image, or a set of 6 face images.
    Read :py:class:`Skybox` docstring for explanation of the parameters.
    """
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

    if not six_images_loading:
        # code for extracting faces from single image
        texture = pyglet.image.load(path).get_texture()
        width, height = texture.width, texture.height
        region_origins = (
            (width // 2, height // 3),  # right
            (0, height // 3),  # left
            (width // 4, int(height * 2 / 3)),  # top
            (width // 4, 0),  # bottom
            (width // 4, height // 3),  # front
            (int(width * 3 / 4), height // 3)  # back
        )

        for origin, target in zip(region_origins, face_targets):
            region = texture.get_region(*origin, width // 4, height // 3)
            image_data = region.get_image_data()
            raw_data = image_data.get_bytes('RGBA', image_data.width * 4)
            glTexImage2D(target, 0, GL_RGBA, image_data.width, image_data.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw_data)

    if six_images_loading:
        # code for 6 separate images
        face_images = ('right', 'left', 'top', 'bottom', 'front', 'back')

        for target, image in zip(face_targets, face_images):
            # loading image
            image = pyglet.image.load(path + f'/{image}.{six_images_ext}').get_texture()
            image_data = image.get_image_data()
            # get the raw pixel bytes (RGBA format)
            raw_data = image_data.get_bytes('RGBA', image.width * 4)  # 4 bytes per pixel
            glTexImage2D(target, 0, GL_RGBA, image_data.width, image_data.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw_data)

    # wrapping and filtering
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return cube_map  # Return the cube map texture ID


class Skybox:
    """
    Creates a skybox around the scene.

    Box is textured with a cube map through custom shader.
    Cube texture can be created from single image, by passing entire filename:
        skybox = Skybox('res/skybox/skybox.png').
    If single image is used, it is expected to be of right orientation (like Denmark flag).

    Alternatively, 6 separate images for each side can be used, by passing path to folder where images are:
        skybox = Skybox('res/skybox/', six_images_loading=True, six_images_ext: str = 'png')
    In this case, it is expected that images are labelled 'right', 'left', 'top', 'bottom', 'front', 'back',
    and with appropriate extension passed.

    After skybox is created, use :py:method:`set_environment_map` to add it to your shader for lighting purposes.
    Call :py:method:`draw` in your on_draw() to display skybox.
    """
    def __init__(self, path: str, six_images_loading=False, six_images_ext: str = 'jpg'):

        # Store the shader program and cube map texture
        self.shader = ShaderProgram(
            Shader(skybox_vertex_shader, 'vertex'),
            Shader(skybox_fragment_shader, 'fragment')
        )

        self.cube_map = create_cube_map(path, six_images_loading, six_images_ext)
        # assign cube map texture to texture slot 0
        self.shader["skybox"] = 0

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

        # Draw the skybox cube
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        # Restore OpenGL state
        glDepthFunc(depth_func.value)

    def set_environment_map(self, program: ShaderProgram, uniform: str = "skybox", tex_slot: int = 5):
        program.use()
        program[uniform] = tex_slot
        glActiveTexture(GL_TEXTURE0 + tex_slot)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cube_map)
