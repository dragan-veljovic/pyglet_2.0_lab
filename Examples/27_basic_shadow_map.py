"""
Shadow mapping notes:
1) render scene from light's perspective, z-distance is stored in the depth (shadow) map, and shows distance of the closest
fragment to the light (one hit first)
2) render fragments from viewer's perspective. To apply shadows, every observed fragment "P" (point) must first\
 be transformed to light perspective, by multiplying with light's view and proj matrix.
Remember that transformations change xyz coordinates from world space, so now z coordinate of "P" is the depth distance
from the light source to the observed fragment.
Use observed fragment "P" to index shadow map from previous pass, and obtain the closest visible depth from shadow map.
If map returns smaller depth than z-coordinate of P, we can conclude that is in shadow.
"""

import math

import pyglet.math
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    'default_mode': False,
    'width': 1920,
    'height': 1080,
    'fps': 100
}

SHADOW_WIDTH = 1280
SHADOW_HEIGHT = 720

vertex_source = """
#version 330 core

in vec3 position;
in vec2 tex_coords;
in vec4 colors;

out vec3 frag_position;
out vec2 frag_tex_coords;
out vec4 frag_colors;
out vec4 frag_shadow_coords;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

uniform mat4 light_proj;
uniform mat4 light_view;

void main() {

    gl_Position = window.projection * window.view * vec4(position, 1.0);
    frag_tex_coords = tex_coords;
    frag_colors = colors;

    // Pass light-space coordinates to fragment shader
    frag_shadow_coords = light_proj * light_view * vec4(position, 1.0);
}
"""

fragment_source = """#version 330 core

in vec3 frag_position;
in vec2 frag_tex_coords;
in vec4 frag_colors;
in vec4 frag_shadow_coords;

out vec4 final_colors;

uniform sampler2D shadow_map;

float get_shadow_factor(vec4 shadow_coords) {
    // Perform perspective divide
    vec3 proj_coords = shadow_coords.xyz / shadow_coords.w;

    // Transform to [0, 1] texture space
    proj_coords = proj_coords * 0.5 + 0.5;

    // Read the depth from the shadow map
    float closest_depth = texture(shadow_map, proj_coords.xy).r;

    // Current fragment depth in light space
    float current_depth = proj_coords.z;

    // Check if the fragment is in shadow
    return (current_depth > closest_depth + 0.005) ? 0.5 : 1.0; // Bias to reduce artifacts
}

void main() {
    float shadow_factor = get_shadow_factor(frag_shadow_coords);

    // Use shadow factor to darken the fragment
    vec3 color = frag_colors.rgb * shadow_factor;

    // Output final color
    final_colors = vec4(color, frag_colors.a);
}

"""

shadow_source_vert = """
#version 330 core
in vec3 position;
in vec4 colors;

uniform mat4 light_proj;
uniform mat4 light_view;

out vec4 frag_colors;

void main(){
    gl_Position = light_proj * light_view * vec4(position, 1.0);
    frag_colors = colors;
}
"""

shadow_source_frag = """
#version 330 core
in vec4 frag_colors; 

out vec4 frag_depth;

void main() {
    /* 
    Depth information gl_FragCoord.z is automatically saved in the shadow map. 
    But if we want to visualise this depth information, we can manually output float or vec4. 
    */

    frag_depth = vec4(1-gl_FragCoord.z);
}
"""


class PointLight:
    def __init__(self, position: Vec3(0, 0, 0), color: (1.0, 1.0, 1.0), ambient=0.2, diffuse=1.0, specular=0.5):
        self.position = position
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

    def get_light_view(self):
        return pyglet.math.Mat4.look_at(self.position, Vec3(0, 0, 0), Vec3(0, 1, 0))

    def get_light_proj(self):
        return pyglet.math.Mat4.perspective_projection(SHADOW_WIDTH / SHADOW_HEIGHT, 200, 3000, 60)


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.shadow_batch = pyglet.graphics.Batch()

        # camera settings
        self.camera = Camera3D(self)
        self.camera.look_at(Vec3(-200, 500, 500), Vec3(0, 0, 0))
        # fix problem with Camera3D.look_at updating angles
        self.camera._yaw = -math.acos(self.camera._front.x)
        self.camera._pitch = math.asin(self.camera._front.y)

        self.program = ShaderProgram(
            Shader(vertex_source, 'vertex'),
            Shader(fragment_source, 'fragment')
        )

        self.shadow_program = ShaderProgram(
            Shader(shadow_source_vert, 'vertex'),
            Shader(shadow_source_frag, 'fragment')
        )

        # light
        self.light = PointLight(Vec3(-500, 600, 700), color=(255, 255, 255, 255))
        # visual representation of the light with sprite
        # self.light_sprite = pyglet.sprite.Sprite(
        #     pyglet.image.load('res/textures/whitelight.png'),
        #     self.light.position.x, self.light.position.y, self.light.position.z
        # )

        glEnable(GL_DEPTH_TEST)

        # main scene elements
        self.floor = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, rotation=(np.pi / 2, 0, 0),
            color=(100, 100, 100, 255)
        )
        self.back_wall = TexturedPlane(
            (-500, -500, -500), self.batch, None, self.program, 1000, 1000, color=(150, 150, 150, 255)
        )
        self.cube = Cuboid(self.program, self.batch, color=(164, 0, 0, 255))

        # shadow map scene elements (has to repeat?)
        self.floor = TexturedPlane(
            (-500, -500, -500), self.shadow_batch, None, self.shadow_program, 1000, 1000, rotation=(np.pi / 2, 0, 0),
            color=(150, 150, 150, 255)
        )
        self.back_wall = TexturedPlane(
            (-500, -500, -500), self.shadow_batch, None, self.shadow_program, 1000, 1000, color=(150, 150, 150, 255)
        )
        self.cube = Cuboid(self.shadow_program, self.shadow_batch, color=(128, 0, 0, 255))

        self.depth_data = None
        self.render_shadow_batch = False
        self.timer = 0.0
        self.move_light = True

        self.create_shadow_fbo()

    def create_shadow_fbo(self):
        """Create framebuffer and attach the depth texture."""
        # generating framebuffer
        self.depthMapFBO = GLuint()
        glGenFramebuffers(1, self.depthMapFBO)
        # generating depth texture map
        self.shadow_width, self.shadow_height = SHADOW_WIDTH, SHADOW_HEIGHT
        self.depthMap = GLuint()
        glGenTextures(1, self.depthMap)
        glBindTexture(GL_TEXTURE_2D, self.depthMap)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height,
            0, GL_DEPTH_COMPONENT, GL_FLOAT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # attaching texture to the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.depthMapFBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depthMap, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        # reverting to default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer is incomplete! Status:", status)

    def render_shadow_map(self):
        """Render scene from light's view and perspective"""
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depthMapFBO)
        glClear(GL_DEPTH_BUFFER_BIT)
        self.shadow_program['light_proj'] = self.light.get_light_proj()
        self.shadow_program['light_view'] = self.light.get_light_view()
        self.shadow_batch.draw()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)

    def fetch_depth_data(self):
        """Use or inspect data from the depth texture attachment."""
        # Create an array to hold the depth values
        depth_data = (GLfloat * (self.shadow_width * self.shadow_height))()
        # Bind the depth texture
        glBindTexture(GL_TEXTURE_2D, self.depthMap)
        # Retrieve depth texture data
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth_data)
        # Convert the depth data into a Python list or NumPy array for processing
        depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape(self.shadow_height, self.shadow_width)
        return depth_array

    def update_main_shader(self):
        """Update main shader light uniforms and shadow map information from the first pass."""
        # Bind shadow map texture for use in the main scene
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.depthMap)
        self.program['shadow_map'] = 0  # Texture unit 0
        # Pass light matrices to main shader
        self.program['light_proj'] = self.light.get_light_proj()
        self.program['light_view'] = self.light.get_light_view()

    def update_light(self):
        if self.move_light:
            self.timer += 1 / settings['fps']
            self.light.position.x = 500 * math.sin(self.timer)
            self.light.position.y = 500 * math.cos(self.timer)

    def on_draw(self):
        # Render shadow map (first pass)
        self.render_shadow_map()

        # Update main shaders
        self.update_main_shader()

        # update light
        self.update_light()

        # Render main or the shadow batch
        self.clear()
        if self.render_shadow_batch:
            self.shadow_batch.draw()
        else:
            self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        # check contents of the depth data
        if symbol == pyglet.window.key.X:
            self.render_shadow_batch = not self.render_shadow_batch
        if symbol == pyglet.window.key.SPACE:
            self.move_light = not self.move_light
        if symbol == pyglet.window.key.B:
            depth_data = self.fetch_depth_data().flatten()
            print("shadow map resolution", depth_data.shape)
            print(self.fetch_depth_data())


if __name__ == '__main__':
    start_app(App, settings)