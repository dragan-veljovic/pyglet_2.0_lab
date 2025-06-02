import pyglet

import tools.camera
from tools.definitions import *
from tools.instancing import InstanceRendering
import numpy as np
import math
from pyglet.gl import *
from tools.model import load_obj_model
from tools.skybox import Skybox

vertex_source = """
#version 150 core

in vec3 position;
in vec4 color;
in vec4 instance_data;  // pos_x, pos_y, scale, rotation
in vec2 tex_coord;

out vec4 fragColor;
out vec2 fragCoord;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

void main() {
    // Extract transformation data
    vec3 offset = vec3(instance_data.xy, position.z*5);
    float scale = instance_data.z;
    float angle = instance_data.w;

    // Apply rotation
    float cos_rot = cos(angle);
    float sin_rot = sin(angle);
    mat3 rotMatrix = mat3(cos_rot, -sin_rot, 0, sin_rot, cos_rot, 0, 0, 0, 0);

    // Transform the vertex: scale -> rotate -> translate
    vec3 scaledPos = position.xyz * scale;
    vec3 rotatedPos = rotMatrix * scaledPos;
    vec3 finalPos = rotatedPos + offset;

    gl_Position = window.projection * window.view * vec4(finalPos, 1.0);
    fragColor = color;
    fragCoord = tex_coord; 
}
"""

fragment_source = """
#version 150 core
in vec4 fragColor;
in vec2 fragCoord;

out vec4 outColor;

uniform sampler2D diffuse_tex;

void main() {
    vec3 diffuse = texture(diffuse_tex, fragCoord).rgb;
    outColor = vec4(fragColor.rgb*diffuse, 1.0);
}
"""


class TextureGroup(pyglet.graphics.Group):
    def __init__(self, program, texture, order=0, parent=None):
        super(TextureGroup, self).__init__(order, parent)
        self.program = program
        self.texture = texture
        self.program['diffuse_tex'] = 1

    def set_state(self) -> None:
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(self.texture.target, self.texture.id)

    def unset_state(self) -> None:
        glBindTexture(self.texture.target, 0)

    def __eq__(self, other) -> bool:
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.parent == other.parent)

    def __hash__(self) -> int:
        return hash((self.texture.target, self.texture.id, self.order, self.parent))

    def __enter__(self):
        self.set_state()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unset_state()


class MyApp(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        #self.skybox = Skybox('../res/textures/skybox2/', 'png')

        set_background_color()
        self.clock = pyglet.clock.Clock()
        self.time = self.clock.time()
        self.camera = tools.camera.Camera3D(self, z_far=30_000)

        glEnable(GL_DEPTH_TEST)

        N = 300  # number of instances is square of this
        self.num_instances = N ** 2
        self.model_data = load_obj_model(
            'res/model/chair/old_chair.obj'
        )
        self.texture_group = TextureGroup(self.program, pyglet.image.load('res/model/chair/old_chair_Albedo.png').get_texture())
        # instance_data = np.zeros((num_instances, 4), dtype=np.float32)
        self.instance_data = np.array((0, 0, 1, 0) * self.num_instances, dtype=np.float32).reshape(self.num_instances, 4)

        self.instances = InstanceRendering(
            self.program,
            np.array(self.model_data['position'], dtype=np.float32),
            np.array((1, 1, 1) * (len(self.model_data['position'])//3), dtype=np.float32),
            np.array(self.model_data['indices'], dtype=np.uint32),
            self.instance_data,
            self.num_instances,
            np.array(self.model_data['tex_coords'], dtype=np.float32),
            self.texture_group
        )

    def update_instance_data(self):
        """Use numpy version to avoid looping."""
        time_value = self.clock.time() - self.time
        N = int(math.sqrt(self.num_instances))

        index = 0
        for y in range(N):
            for x in range(N):
                # Position (in a grid pattern)
                self.instance_data[index, 0] = x * 75  # x position
                self.instance_data[index, 1] = y * 75  # y position

                # Scale (varies based on position)
                distance = math.sqrt((x) ** 2 + (y) ** 2) / 7.0
                self.instance_data[index, 2] = 1 + 0.5*math.sin(time_value)

                # Rotation (animated over time, different for each instance)
                self.instance_data[index, 3] = time_value*0.5 + distance * 2

                index += 1

    def update_instance_data_numpy(self):
        """Around 20x faster than pure python."""
        time_value = self.clock.time() - self.time
        N = int(math.sqrt(self.num_instances))

        # Create grid of x, y indices
        y_indices, x_indices = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')

        # Flatten the grid
        x_flat = x_indices.ravel()
        y_flat = y_indices.ravel()

        # Compute positions
        self.instance_data[:, 0] = x_flat * 75  # x position
        self.instance_data[:, 1] = y_flat * 75  # y position

        # Compute distance from origin
        distances = np.sqrt(x_flat ** 2 + y_flat ** 2) / 7.0

        # Compute scale (same for all instances)
        self.instance_data[:, 2] = 1 + 5 * np.sin(time_value)

        # Compute rotation
        self.instance_data[:, 3] = time_value*0.5 + distances * 2

    def on_draw(self):
        self.clear()
        self.skybox.draw()
        self.instances.update(self.update_instance_data_numpy)
        self.instances.draw()

if __name__ == '__main__':
    app = start_app(MyApp, {'default_mode': True, 'vsync': False})
    app.instances.delete()
