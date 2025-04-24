"""
Example of enabling normal mapping on a textured plane, for orientations different than vertical.
A tangent and bitangent vectors are calculated and passed to a shader, where TBN matrix is used to adjust normals.
TODO: implement TB calculation in model loading
"""

import math
import pyglet.math
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    'default_mode': True,
    'width': 1280,
    'height': 720,
    'fps': 60,
}

SHADOW_WIDTH = 1280
SHADOW_HEIGHT = 720

vertex_source = """
#version 330 core
in vec3 position;
in vec2 tex_coords;
in vec3 normals;
in vec4 colors;
in vec3 tangents;
in vec3 bitangents;

out vec3 frag_position;
out vec2 frag_tex_coords;
out vec3 frag_normals;
out vec4 frag_colors;
out vec4 frag_shadow_coords;
out mat3 TBN;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

uniform mat4 light_proj;
uniform mat4 light_view;
uniform float time;

void main() {
    // calculating TBN
    vec3 T = normalize(tangents);
    vec3 B = normalize(bitangents);
    vec3 N = normalize(normals);

    gl_Position = window.projection * window.view * vec4(position, 1.0);
    frag_tex_coords = tex_coords;
    frag_position = position; 
    frag_colors = colors;
    frag_normals = normals;

    TBN = mat3(T, B, N);


    // Pass light-space coordinates to fragment shader
    frag_shadow_coords = light_proj * light_view * vec4(position, 1.0);
}
"""

fragment_source = """
#version 330 core
in vec3 frag_position;
in vec2 frag_tex_coords;
in vec3 frag_normals;
in vec4 frag_colors;
in vec4 frag_shadow_coords;
in mat3 TBN;

out vec4 final_colors;

uniform sampler2D shadow_map;
uniform sampler2D diffuse_texture;
uniform sampler2D normal_map;

uniform vec3 light_position;
uniform vec3 view_position;

uniform bool shadowing = true;
uniform bool soft_shadows = true;
uniform bool lighting = true;
uniform bool lighting_diffuse = true; 
uniform bool lighting_specular = true;
uniform bool texturing = true;
uniform bool normal_mapping = true;

float shadow_bias = 0.005;

// Number of samples for PCF
uniform int pcf_samples = 9;  // Adjust this for quality (more samples = smoother shadows)

float get_shadow_factor(vec4 shadow_coords) {
    // Perform perspective divide
    vec3 proj_coords = shadow_coords.xyz / shadow_coords.w;

    // Transform to [0, 1] texture space
    proj_coords = proj_coords * 0.5 + 0.5;

    if (soft_shadows) {
        // Apply Percentage Closer Filtering (PCF)
        float shadow = 0.0;
        float sample_offset = 0.001;  // Adjust this for smoothness

        // PCF sampling around the current fragment
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 offset = vec2(x, y) * sample_offset;
                float depth = texture(shadow_map, proj_coords.xy + offset).r;
                shadow += (proj_coords.z > depth + 0.005) ? 0.5 : 1.0;  // Bias to reduce artifacts
            }
        }

        // Average the result
        shadow /= float(pcf_samples);
        return shadow;
    } else {
         // Read the depth from the shadow map
        float closest_depth = texture(shadow_map, proj_coords.xy).r;

        // Current fragment depth in light space
        float current_depth = proj_coords.z;

        // Check if the fragment is in shadow
        return (current_depth > closest_depth + 0.005) ? 0.5 : 1.0; // Bias to reduce artifacts
    }
}

void main() {
    float shadow_factor;
    // Get pixel shadow information using PCF
    if (shadowing) {
        shadow_factor = get_shadow_factor(frag_shadow_coords);
    } else {
        shadow_factor = 1.0;
    }

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    if (lighting){
        vec3 normal;
        if (normal_mapping) {
            // obtain normal from normal map in range [0, 1]
            normal = texture(normal_map, frag_tex_coords).rgb;
            normal = normal * 2.0 - 1.0;
            normal = normalize(TBN * normal);
        } else {
            normal = normalize(frag_normals);
        }

        // normal = normalize(frag_normals);

        vec3 light_color = vec3(1.0);
        vec3 light_direction = normalize(light_position - frag_position);

        // Ambient lighting
        ambient = 0.3 * light_color;

        // Diffuse lighting
        float diff = lighting_diffuse ? max(dot(light_direction, normal), 0.0) : 0.0;
        diffuse = diff * light_color;

        // Specular lighting
        float spec;
        if (lighting_specular){
            vec3 view_dir = normalize(view_position - frag_position);  // direction to viewer
            vec3 reflect_dir = reflect(-light_direction, normal);  // reflection around normal
            spec = pow(max(dot(view_dir, reflect_dir), 0.0), 64);  // Specular factor
        } else {
            spec = 0.0;
        }
        specular = spec * light_color;
    } else {
        ambient = vec3(1.0);
        diffuse = vec3(0.0);
        specular = vec3(0.0);
    }

    vec3 texture_diff;
    if (texturing){
        // Apply textures
        texture_diff = texture(diffuse_texture, frag_tex_coords).rgb;
    } else {
        texture_diff = frag_colors.rgb;
    }

    // Combine lighting and shadow factors
    vec3 lighting = vec3(ambient + (diffuse + specular) * shadow_factor);
    final_colors = vec4(texture_diff * lighting, frag_colors.a);
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
    // set explicitly or not, this is what effectively happens behind the scene
    frag_depth = vec4(1 - gl_FragCoord.z);
}
"""


class DiffuseNormalTextureGroup(Group):
    def __init__(
            self,
            diffuse: Texture,
            normal: Texture,
            program: ShaderProgram,
            transparency: bool = False,
            order=0, parent=None
    ):
        """
        A Group that enables and binds a diffuse and normal map to a ShaderProgram.
        TextureGroups are equal if their Texture and ShaderProgram
        are equal.
        :param texture: Texture to bind.
        :param program: Shader program to use.
        :param order: Change the order to render above or below other Groups.
        :param parent: Parent group.
        """
        super().__init__(order, parent)
        self.diffuse = diffuse
        self.normal = normal
        self.program = program
        self.transparency = transparency

    def set_state(self):
        # activate and bind diffuse texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.diffuse.target, self.diffuse.id)

        # activate and bind normal texture
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(self.normal.target, self.normal.id)
        try:
            self.program['normal_map'] = 2  # Texture unit 2
        except:
            pass
        self.program['normal_mapping'] = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.transparency:
            glDepthMask(GL_FALSE)
        self.program.use()

    def unset_state(self):
        glDisable(GL_BLEND)
        if self.transparency:
            glDepthMask(GL_TRUE)
        self.program.stop()
        self.program['normal_mapping'] = False

    def __hash__(self):
        return hash((self.diffuse.target, self.diffuse.id, self.normal.target, self.normal.id, self.order, self.parent,
                     self.program))

    def __eq__(self, other: Group):
        return (self.__class__ is other.__class__ and
                self.normal.target == other.normal.target and
                self.normal.id == other.normal.id and
                self.diffuse.target == other.diffuse.target and
                self.diffuse.id == other.diffuse.id and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)


class PointLight:
    def __init__(self, position=Vec3(0, 0, 0), color=(1.0, 1.0, 1.0), ambient=0.2, diffuse=1.0, specular=0.5):
        self.position = position
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

    def get_light_view(self):
        return pyglet.math.Mat4.look_at(self.position, Vec3(0, 0, 0), Vec3(0, 1, 0))

    def get_light_proj(self):
        return pyglet.math.Mat4.perspective_projection(SHADOW_WIDTH / SHADOW_HEIGHT, 200, 3000, 90)


class NormalMappedTexturedPlane:
    def __init__(
            self,
            position: tuple[float, float, float],
            batch, group, program: ShaderProgram,
            length=300, height=200, rotation=(0, 0, 0),
            color=None,
    ):
        """
        A 2D textured plane in a 3D space.
        Position parameter is lower-left corner of the rectangle.
        Vertical rectangle is assumed at initiation, which is then rotated around lower-left corner
        with rotation parameter, representing a tuple of Oiler angles (pitch, yaw, roll) in radians.
        """
        self.position = position  # lower left corner of the rect
        self.x, self.y, self.z = position
        self.batch = batch
        self.group = group
        self.program = program
        self.length = length
        self.height = height
        self.rotation = rotation  # tuple of pitch, yaw, and roll angles in radians, respectively
        self.pitch, self.yaw, self.roll = rotation

        self.color = color or (255, 255, 255, 255)

        vertices = np.array((
            (self.x, self.y, self.z),
            (self.x + length, self.y, self.z),
            (self.x + length, self.y + height, self.z),
            (self.x, self.y + height, self.z)
        ))

        vertex_normals = np.array((
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1),
            (0, 0, 1)
        ))

        # rotation factor
        vertices = rotate_points(vertices, self.pitch, self.yaw, self.roll, anchor=self.position)
        vertex_normals = rotate_points(vertex_normals, self.pitch, self.yaw, self.roll)

        # normalizing rotated normals
        magnitude = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / magnitude

        texture_coordinates = np.array((
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ), dtype=np.float32)

        # indexing
        indices = (0, 1, 2, 0, 2, 3)
        gl_triangles_vertices = []
        tex_coords = []
        normals = []
        for idx in indices:
            gl_triangles_vertices.extend(vertices[idx])
            tex_coords.extend(texture_coordinates[idx])
            normals.extend(vertex_normals[idx])

        # calculating TBN
        tangent, bitangent = get_tangent_and_bitangent_vectors(
            vertices[0], vertices[1], vertices[2],
            texture_coordinates[0], texture_coordinates[1], texture_coordinates[2]
        )

        # creating vertex buffers
        count = len(gl_triangles_vertices) // 3
        self.vertex_list = self.program.vertex_list(
            count, GL_TRIANGLES, batch, group,
            position=('f', gl_triangles_vertices),
            normals=('f', normals),
            colors=('Bn', self.color * count),
            tex_coords=('f', tex_coords),
            tangents=('f', tangent * count),
            bitangents=('f', bitangent * count)
        )


def get_tangent_and_bitangent_vectors(
        p1, p2, p3,  # triangle vertices in (x, y, z) format
        t1, t2, t3,  # triangle vertex texture coordinates in (u, v) format
) -> tuple[tuple, tuple]:
    # triangle vertices

    delta_uvs = np.array((
        (t2 - t1),
        (t3 - t1)
    ), dtype=np.float32)

    edges = np.array((
        (p2 - p1),
        (p3 - p1)
    ))

    # shortcut to solving following matrix equation
    # edges = delta_uvs @ TB
    # TB = 1/det(delta_uvs) * adj(delta_uvs) @ edges
    TB = np.linalg.solve(delta_uvs, edges).tolist()

    return TB[0], TB[1]


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        # center_window(self)
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

        glEnable(GL_DEPTH_TEST)

        self.rock_group = DiffuseNormalTextureGroup(
            diffuse=pyglet.image.load(
                'res/textures/castle_brick_02_red_2k/textures/castle_brick_02_red_diff_2k.png').get_texture(),
            normal=pyglet.image.load(
                'res/textures/castle_brick_02_red_2k/textures/castle_brick_02_red_nor_gl_2k.png').get_texture(),
            program=self.program
        )

        self.wall_group = DiffuseNormalTextureGroup(
            diffuse=pyglet.image.load('res/textures/brick/brickwall.jpg').get_texture(),
            normal=pyglet.image.load('res/textures/brick/normal_mapping_normal_map.png').get_texture(),
            program=self.program
        )

        # main scene elements
        self.back_wall = NormalMappedTexturedPlane(
            (-750, -500, -500), self.batch, self.wall_group, self.program, 1500, 1000, color=(150, 150, 150, 255)
        )
        self.floor = NormalMappedTexturedPlane(
            (-750, -500, -500), self.batch, self.wall_group, self.program, 1500, 1000, rotation=(np.pi / 2, 0, 0),
            color=(100, 100, 100, 255)
        )

        self.left_wall = NormalMappedTexturedPlane(
            (-750, -500, 500), self.batch, self.wall_group, self.program, 1500, 1000, rotation=(0, -np.pi / 3, 0),
            color=(150, 150, 150, 255)
        )

        self.cube2 = Cuboid(
            self.program, self.batch, position=(500, 0, -200), size=(100, 750, 300),
            texture=pyglet.image.load('res/textures/img.png').get_texture()
        )

        position, tex_coords, normals = load_model_from_obj('model/porsche/Porsche_911_GT2.obj',
                                                            rotation=[0, np.pi / 2, 0])
        position = np.array(position) * 150
        count = len(position) // 3

        # shadow domain
        self.shadow_model = self.shadow_program.vertex_list(
            count=count,
            mode=GL_TRIANGLES,
            batch=self.shadow_batch,
            position=('f', position),
        )

        self.depth_data = None
        self.render_shadow_batch = False
        self.wireframe = False
        self.timer = 0.0
        self.move_light = True

        self.create_shadow_fbo()
        self.program['normal_mapping'] = True

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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT,
                     GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # attaching texture to the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.depthMapFBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depthMap, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer is incomplete! Status:", status)

        # Bind shadow map texture for use in the main scene
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depthMap)
        self.program['shadow_map'] = 1  # Texture unit 1

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

    def update_main_shader(self):
        # Pass light matrices for shadow calculation
        self.program['light_proj'] = self.light.get_light_proj()
        self.program['light_view'] = self.light.get_light_view()
        # pass light vectors for phong lighting calculation
        self.program['light_position'] = self.light.position
        self.program['view_position'] = self.camera.position

    def update_light_position(self):
        if self.move_light:
            self.timer += 1 / settings['fps']
            self.light.position = Vec3(500 * math.sin(self.timer), 500 * math.cos(self.timer), 500)

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

    def on_draw(self):
        # Render shadow map (first pass)
        self.render_shadow_map()

        # input results into main shader
        self.update_main_shader()

        # update light position
        self.update_light_position()

        # Render main scene
        self.clear()

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # redner main shader or the depth map
        if self.render_shadow_batch:
            self.shadow_batch.draw()
        else:
            self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        match symbol:
            case pyglet.window.key.C:
                self.depth_data = self.fetch_depth_data()
                print("buffer depth data: ", self.depth_data)
            case pyglet.window.key.X:
                self.render_shadow_batch = not self.render_shadow_batch
            case pyglet.window.key.SPACE:
                self.move_light = not self.move_light

            # shader render settings
            case pyglet.window.key.M:
                self.program['shadowing'] = not self.program['shadowing']
            case pyglet.window.key.L:
                self.program['lighting'] = not self.program['lighting']
            case pyglet.window.key.T:
                self.program['texturing'] = not self.program['texturing']
            case pyglet.window.key.O:
                self.program['lighting_diffuse'] = not self.program['lighting_diffuse']
            case pyglet.window.key.P:
                self.program['lighting_specular'] = not self.program['lighting_specular']
            case pyglet.window.key.N:
                self.program['soft_shadows'] = not self.program['soft_shadows']
            case pyglet.window.key.H:
                self.program['normal_mapping'] = not self.program['normal_mapping']
            case pyglet.window.key.B:
                self.wireframe = not self.wireframe


if __name__ == '__main__':
    start_app(App, settings)
