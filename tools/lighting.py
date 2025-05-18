from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.graphics import Batch
from pyglet.math import Vec3, Vec4, Mat4
from pyglet.gl import *


class Light:
    """Base class for all lights."""
    def __init__(
            self,
            ambient=0.25,
            diffuse=1.0,
            specular=1.0,
            color=(1.0, 1.0, 1.0),
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.color = color
        self.up = Vec3(0, 1, 0)
        self.proj_matrix = Mat4()
        self.view_matrix = Mat4()

    def get_light_view_matrix(self):
        raise NotImplementedError("Light's View matrix is required for shadow mapping.")

    def get_light_projection_matrix(self):
        raise NotImplementedError("Light's projection matrix is required for shadow mapping.")


class SpotLight(Light):
    """
    A spotlight is a frustum, formed from given z_near, z_far, aspect and fov parameters,
    oriented in space with position and target vectors.
    Shader lighting calculations consider light and fragment position to determine
    light direction.
    Shadow mapping consider spotlight as a perspective camera.
    TODO: limit lighting and shadowing to frustum volume only
    """
    def __init__(
            self,
            position=Vec3(100, 100, 100),
            target=Vec3(0, 0, 0),
            aspect_ratio=1.0,
            z_near=200,
            z_far=5000,
            fov=80,
            ambient=0.25,
            diffuse=1.0,
            specular=1.0,
            color=(1.0, 1.0, 1.0),
    ):
        super().__init__(ambient, diffuse, specular, color)
        self.position = position
        self.target = target
        self.aspect = aspect_ratio
        self.z_near = z_near
        self.z_far = z_far
        self.fov = fov
        self.view_matrix = self.get_light_view_matrix()
        self.proj_matrix = self.get_light_projection_matrix()
        self.directional_light = False

    def get_light_view_matrix(self) -> Mat4:
        return Mat4.look_at(self.position, self.target, self.up)

    def get_light_projection_matrix(self) -> Mat4:
        return Mat4.perspective_projection(self.aspect, self.z_near, self.z_far, self.fov)


class DirectionalLight(Light):
    """
    Directional light is a beam of parallel rays contained in a cuboid with given parameters.
    Light direction is determined by position and target vectors, and constant for all fragments.
    Shadow mapping consider directional light as an orthographic camera.
    """
    def __init__(
            self,
            position=Vec3(0.5, 1.0, 0.5),
            width=1280,
            height=720,
            z_near=200,
            z_far=5000
    ):
        super().__init__()
        self.position = position
        self.target = Vec3(0, 0, 0)
        self.width = width
        self.height = height
        self.z_near = z_near
        self.z_far = z_far
        self.view_matrix = self.get_light_view_matrix()
        self.proj_matrix = self.get_light_projection_matrix()
        self.directional_light = True

    def get_light_view_matrix(self) -> Mat4:
        return Mat4.look_at(self.position, self.target, self.up)

    def get_light_projection_matrix(self) -> Mat4:
        return Mat4.orthogonal_projection(-3440, 3440, -1440, 1440, self.z_near, self.z_far)


class ShadowMap:
    def __init__(
            self,
            light: Light,
            program: ShaderProgram,
            shadow_width=1000, shadow_height=1000,
    ):
        """
        Shadow (depth) map will bind to Texture Unit 4 for use in your main shader.
        """
        self.light = light
        self.program = program
        self.shadow_width = shadow_width
        self.shadow_height = shadow_height
        self.width, self.height = 3440, 1440  # USE WINDOW

        # shader program for the shadow map creation pass
        self.shadow_batch = Batch()
        self.shadow_program = self._get_default_shadow_program()
        self.depth_map_fbo = GLuint()
        self.depth_map = GLuint()
        self._create_shadow_fbo()

    def render(self):
        """Render scene from light's view and perspective"""
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)

        glClear(GL_DEPTH_BUFFER_BIT)

        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        #self.program['shadow_map'] = 4  # Texture unit 4

        self.shadow_program['light_proj'] = self.light.proj_matrix
        self.shadow_program['light_view'] = self.light.view_matrix
        self.shadow_batch.draw()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)

    @staticmethod
    def _get_default_shadow_program() -> ShaderProgram:
        shadow_vert = """
            #version 330 core
            in vec3 position;
            
            uniform mat4 light_proj;
            uniform mat4 light_view;
            
            out vec4 frag_colors;
            
            void main(){
                gl_Position = light_proj * light_view * vec4(position, 1.0);
            }
        """
        shadow_frag = """
            #version 330 core
            out vec4 frag_depth;
            
            void main() {
                // set explicitly or not, this is what effectively happens behind the scene
                frag_depth = vec4(1 - gl_FragCoord.z);
            }
        """
        return ShaderProgram(
            Shader(shadow_vert, 'vertex'),
            Shader(shadow_frag, 'fragment')
        )

    def _create_shadow_fbo(self):
        """Create framebuffer and attach the depth texture."""
        # generating framebuffer
        glGenFramebuffers(1, self.depth_map_fbo)
        # generating depth texture map
        glGenTextures(1, self.depth_map)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT,
                     GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # preventing depth texture wrapping issue
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        # Optional: define the border color for clamping
        border_color = (GLfloat * 4)(1.0, 1.0, 1.0, 1.0)  # white border = fully lit
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # attaching texture to the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer is incomplete! Status:", status)

        # Bind shadow map texture for use in the main scene
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.program['shadow_map'] = 4  # Texture unit 4

    # def fetch_depth_data(self):
    #     """Use or inspect data from the depth texture attachment."""
    #     # Create an array to hold the depth values
    #     depth_data = (GLfloat * (self.width * self.height))()
    #     # Bind the depth texture
    #     glBindTexture(GL_TEXTURE_2D, self.depth_map)
    #     # Retrieve depth texture data
    #     glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth_data)
    #     # Convert the depth data into a Python list or NumPy array for processing
    #     depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape(self.width, self.height)
    #     return depth_array

