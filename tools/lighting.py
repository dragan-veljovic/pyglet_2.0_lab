import math
import weakref

from pyglet.graphics.shader import Shader, ShaderProgram, UniformBufferObject, UniformBlock
from pyglet.graphics import Batch
from pyglet.math import Vec3, Vec4, Mat4
from pyglet.gl import *
from pyglet.window import Window
from tools.definitions import get_logger

logger = get_logger(__name__)


class Light:
    """
    Base class for all lights.
    Before 'Light' object can be used for lighting calculations, or any changes reflected
    in the rendering, its attributes must be updated in the shader as uniforms.
    This is done most efficiently through pyglet 'UniformBlock' object, and its
    proper declaration in the shaders.

    UniformBlock can be bound with 'set_binding()' to different UniformBufferObjects,
    each holding different light setting.
    """
    def __init__(
            self,
            ambient=0.25,
            diffuse=0.75,
            specular=1.0,
            color=(1.0, 1.0, 1.0),
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.color = color
        self.up = Vec3(0, 1, 0)
        self.projection = Mat4()
        self.view = Mat4()

        self.ubo: UniformBufferObject | None = None

    def get_light_view_matrix(self):
        raise NotImplementedError("Light's View matrix is required for shadow mapping.")

    def get_light_projection_matrix(self):
        raise NotImplementedError("Light's projection matrix is required for shadow mapping.")

    def bind_to_block(self, light_block: UniformBlock) -> UniformBufferObject:
        """
        This helper method creates UniformBufferObject and populates its data with current attributes.

        Optionally returns created UBO for future manual updates.
        Several UBOs can be bound to same UniformBlock, and set_binding() controls which is used.
        """
        self.ubo = light_block.create_ubo()
        uniform_names = [udata[0] for udata in light_block.uniforms.values()]
        self.update_ubo(uniform_names)

        return self.ubo

    def update_ubo(self, attributes: list, ubo: UniformBufferObject | None = None):
        """
        Update selected attributes data to UniformBufferObject (self.ubo by default, if no ubo is specified).
        Alternatively, pass another appropriate UBO to update.
        Use this before drawing to apply any changes of the Light instance and reflect them in UBO.
        """
        target_ubo = ubo or self.ubo
        if not target_ubo:
            return

        with target_ubo as ubo:
            for attr in attributes:
                if hasattr(self, attr):
                    setattr(ubo, attr, getattr(self, attr))


class SpotLight(Light):
    """
    A spotlight is a frustum, formed from given z_near, z_far, aspect and fov parameters,
    oriented in space with position and target vectors.
    Shader lighting calculations consider light and fragment position to determine
    light direction. Shadow mapping consider spotlight as a perspective camera.
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
            cutoff=5,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.position = position
        self.target = target
        self.aspect = aspect_ratio
        self.directional = False
        self.z_near = z_near
        self.z_far = z_far
        self.fov = fov
        self.view = self.get_light_view_matrix()
        self.projection = self.get_light_projection_matrix()

        # cutoff settings
        self.cutoff_start = math.cos(math.radians(self.fov / 2))
        self.cutoff_end = math.cos(math.radians(self.fov / 2 + cutoff))

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
            z_far=5000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.position = position  # trivial for lighting calc, but useful direction and shadow mapping
        self.target = Vec3(0, 0, 0)
        self.width = width
        self.height = height
        self.directional = True
        self.z_near = z_near
        self.z_far = z_far
        self.view = self.get_light_view_matrix()
        self.projection = self.get_light_projection_matrix()
        self.cutoff_start, self.cutoff_end = -2.0, -2.0  # any cos(theta) is greater, no cutoff

    def get_light_view_matrix(self) -> Mat4:
        return Mat4.look_at(self.position, self.target, self.up)

    def get_light_projection_matrix(self) -> Mat4:
        return Mat4.orthogonal_projection(-self.width, self.width, -self.height, self.height, self.z_near, self.z_far)


class ShadowMap:
    def __init__(
            self,
            light: Light,
            window: Window,
            program: ShaderProgram,
            shadow_width=1000, shadow_height=1000,
    ):
        """
        Shadow (depth) map will bind to Texture Unit 4 for use in your main shader.
        """
        self.light = light
        self.program = program
        self.window = weakref.proxy(window)
        self.shadow_width = shadow_width
        self.shadow_height = shadow_height

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

        self.shadow_program['light_proj'] = self.light.projection
        self.shadow_program['light_view'] = self.light.view
        self.shadow_batch.draw()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window.width, self.window.height)

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
