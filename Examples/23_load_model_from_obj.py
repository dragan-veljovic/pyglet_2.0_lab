"""
Implementation of Phong lighting model for a point source.
"""
import random

import pyglet.window
from tools.definitions import *
from tools.camera import Camera3D
from tools.graphics import *

settings = {
    "default_mode": True,
    "width": 1280,
    "height": 720,
    "fps": 60,
    "resizable": True,
    "config": get_config(samples=4),
}

_vertex_source = """#version 330 core
    in vec3 position;
    in vec3 normals;
    in vec4 colors;
    in vec2 tex_coords;

    out vec3 frag_position;
    out vec3 frag_normals;
    out vec4 object_colors;
    out vec2 texture_coords;

    uniform WindowBlock 
    {                       // This UBO is defined on Window creation, and available
        mat4 projection;    // in all Shaders. You can modify these matrixes with the
        mat4 view;          // Window.view and Window.projection properties.
    } window;  

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1);
        frag_position = position;
        frag_normals = normals;
        object_colors = colors;
        texture_coords = tex_coords;
    }
"""

_fragment_source = """#version 330 core
    in vec3 frag_position;
    in vec3 frag_normals;
    in vec4 object_colors;
    in vec2 texture_coords;

    out vec4 final_colors;

    uniform sampler2D our_texture;

    // light uniforms
    uniform vec3 light_position;
    uniform vec3 light_color;
    uniform float ambient_strength;
    uniform vec3 view_position;
    uniform float specular_strength;
    uniform float shininess;

    // texturing or shading
    uniform bool texturing = true;

    void main()
    {   
        // Ambient lighting
        vec3 ambient = ambient_strength * light_color;  // scale light by ambient strength

        // Diffuse lighting
        vec3 norm = normalize(frag_normals);
        vec3 light_dir = normalize(light_position - frag_position);
        float diff = max(dot(light_dir, norm), 0.0);  // diffusion factor is positive cos(theta)
        vec3 diffuse = diff * light_color;  // Scale light by diffusion factor

        // Specular lighting
        vec3 view_dir = normalize(view_position - frag_position);  // direction to viewer
        vec3 reflect_dir = reflect(-light_dir, norm);  // reflection around normal
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);  // Specular factor
        vec3 specular = specular_strength * spec * light_color;  // Scale light by specular factor

        // Inverse square-distance attenuation
        float distance = length(light_position - frag_position);
        float attenuation = min(1000 / (distance), 1.0);
        diffuse *= attenuation;
        specular *= attenuation;

        // Combine results
        vec3 result = (ambient + diffuse + specular) * object_colors.rgb;

        // texturing or monochrome shading
        if (texturing){
            vec4 tex_color = texture(our_texture, texture_coords);
            final_colors = tex_color * vec4(result, 1.0);
        } else {
            final_colors = vec4(result, 1.0);
        }
    }
"""

vert_shader = Shader(_vertex_source, 'vertex')
frag_shader = Shader(_fragment_source, 'fragment')
shader_program = ShaderProgram(vert_shader, frag_shader)


class PointLight:
    def __init__(self, position: Vec3(0, 0, 0), color: (1.0, 1.0, 1.0), ambient=0.2, diffuse=1.0, specular=0.5):
        self.position = position
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular


def load_model_from_obj(file_path, uvw_tex_coords=False, rotation=(0, 0, 0)) -> tuple:
    """
    Loads and process 3D model data from *.obj file.
    Returns "position", "tex_coords" and "normals" for VertexList.
    Assumes triangle faces.
    """
    # input data
    vertex_position = []
    texture_coords = []
    vertex_normals = []
    faces = []

    # output data
    position = []
    normals = []
    tex_coords = []

    with open(file_path, "r") as file:
        for line in file:
            # remove any starting or ending whitespaces
            line = line.strip()

            # fetch vertex data
            if line.startswith("v "):
                parts = line.split()
                assert len(parts) == 4, "Incompatible vertex data found, expected 3 component per vertex."
                vertex = float(parts[1]), float(parts[2]), float(parts[3])
                vertex_position.append(vertex)

            # fetch texture data
            elif line.startswith("vt "):
                parts = line.split()
                if uvw_tex_coords:
                    tex_coord = float(parts[1]), float(parts[2]), float(parts[3]) if len(parts) > 3 else 0.0
                else:
                    tex_coord = float(parts[1]), float(parts[2])
                texture_coords.append(tex_coord)

            # fetch normal data
            elif line.startswith("vn "):
                parts = line.split()
                assert len(parts) == 4, "Incompatible vertex data found, expected 3 component per vertex."
                normal = float(parts[1]), float(parts[2]), float(parts[3])
                vertex_normals.append(normal)

            # fetch face data
            elif line.startswith("f "):
                line = line.split()
                assert len(line) == 4, "Expected triangle face with 3 sets of vertex/texture/normal data."
                face = []
                for part in line[1:]:
                    ptn_str = part.split("/")
                    ptn = int(ptn_str[0]), int(ptn_str[1]), int(ptn_str[2])
                    face.append(ptn)
                faces.append(face)

    if any(rotation):
        pitch, yaw, roll = rotation
        vertex_position = rotate_points(vertex_position, pitch, yaw, roll)
        vertex_normals = rotate_points(vertex_normals, pitch, yaw, roll)

    # assembling vertex data (allowing for obj indexing that starts for 1)
    for face in faces:
        for ptn in face:
            pos_index, tex_index, nor_index = ptn
            pos = vertex_position[pos_index - 1]
            tex = texture_coords[tex_index - 1]
            nor = vertex_normals[nor_index - 1]
            position.extend(pos)
            tex_coords.extend(tex)
            normals.extend(nor)

    return position, tex_coords, normals


class App(pyglet.window.Window):
    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        center_window(self)
        set_background_color()
        self.batch = pyglet.graphics.Batch()
        self.program = shader_program
        self.camera = Camera3D(self)
        self.camera.look_at(Vec3(0, 500, 500), Vec3(1000, 0, 0))
        self.time = 0.0
        self.run = True
        self.clamp_lith_to_camera = False

        pyglet.resource.path = ['res/textures/']
        pyglet.resource.reindex()
        self.texture = pyglet.resource.texture('img.png')
        self.texture2 = pyglet.resource.texture('brick_wall.jpg')
        self.wall_texture = pyglet.resource.texture('rock_2K.jpg')
        self.texture_group = TextureGroup(self.texture, self.program)
        self.wall_group = TextureGroup(self.wall_texture, self.program)
        self.wall_group2 = TextureGroup(self.texture2, self.program)

        glEnable(GL_DEPTH_TEST)

        layers = 2
        N = 6
        s = 10

        self.floor = TexturedPlane(
            (-500, 0, -500), self.batch,
            self.wall_group,
            self.program, length=2500, height=2000, rotation=(np.pi / 2, 0, 0)
        )

        self.back_wall = TexturedPlane(
            (-500, 0, -500), self.batch, self.wall_group2, self.program, 2500, 1800
        )

        self.right_wall = TexturedPlane(
            (2000, 0, -500), self.batch, self.wall_group2, self.program, 2000, 1800, rotation=(0, np.pi / 2, 0)
        )

        self.ceiling = TexturedPlane(
            (-500, 1800, 1500), self.batch,
            self.wall_group, self.program, length=2500, height=2000, rotation=(-np.pi / 2, 0, 0)
        )

        # self.planes = []
        # for n in range(layers):
        #     for i in range(N):
        #         for j in range(N):
        #             self.planes.append(
        #                 TexturedPlane(((300 + s) * i, (200 + s) * j, n * 200 + 30 * (-j)), self.batch,
        #                               self.texture_group, self.program)
        #             )

        cube_positions = (
            (100, 100, 600),
            (350, 100, 600),
            (350, 100, 800),
            (350, 100, 1000),
            (550, 100, 600),
            (550, 100, 800),
            (550, 100, 1000),
            (750, 100, 600),
            (750, 100, 800),
            (750, 100, 1000),
            (750, 300, 800),
        )

        self.cubes = []
        for position in cube_positions:
            self.cubes.append(
                Cuboid(self.program, self.batch, position, texture=self.texture)
            )

        # light properties
        self.light = PointLight(
            Vec3(300, 500, 600), (1, 1, 1.0)
        )
        self.shininess = 64
        self.object_color = [1.0, 1.0, 1.0]

        my_position, my_tex_coords, my_normals = load_model_from_obj("model/church.obj", rotation=(0, np.pi, 0))

        self.program.vertex_list(
            len(my_position) // 3,
            GL_TRIANGLES,
            self.batch,
            group=TextureGroup(
                texture=pyglet.image.load("model/church_diffuse.png").get_texture(),
                program=self.program,
            ),
            position=('f', (np.array(my_position) * 60).tolist()),
            colors=('f', (1, 1, 1, 1) * (len(my_position) // 3)),
            normals=('f', my_normals),
            tex_coords=('f', my_tex_coords)
        )

        self.update_shader_uniforms()

    def update_light_position(self):
        if self.clamp_lith_to_camera:
            self.light.position = self.camera.position
        else:
            if self.run:
                self.light.position.x = 300 * np.sin(self.time * 2)
                self.light.position.y = 200 * np.sin(self.time) + 300
                self.light.position.z = 200 * np.sin(self.time) + 600

    def update_shader_uniforms(self):
        self.program['light_position'] = self.light.position
        self.program['light_color'] = self.light.color
        self.program['ambient_strength'] = self.light.ambient
        self.program['view_position'] = self.camera.position
        self.program['specular_strength'] = self.light.specular
        self.program['shininess'] = self.shininess

    def on_draw(self) -> None:
        if self.run:
            self.time += 1 / settings['fps']

        self.update_light_position()
        self.update_shader_uniforms()

        self.clear()
        self.batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        super(App, self).on_key_press(symbol, modifiers)
        if symbol == pyglet.window.key.ESCAPE:
            self.on_close()
        elif symbol == pyglet.window.key.SPACE:
            self.run = not self.run
        elif symbol == pyglet.window.key.C:
            self.program['texturing'] = not self.program['texturing']
        elif symbol == pyglet.window.key.X:
            self.clamp_lith_to_camera = not self.clamp_lith_to_camera


if __name__ == '__main__':
    start_app(App, settings)
