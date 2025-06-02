import typing

import pyglet
from pyglet.gl import *
import numpy as np
from numpy.typing import NDArray
import ctypes
import math
from pyglet.graphics.shader import ShaderProgram, Shader
from pyglet.graphics import Group
from collections.abc import Callable
from typing import Optional

# class InstanceRenderer:
#     def __init__(
#             self,
#             program: ShaderProgram,
#             position: np.array,
#             colors: np.array,
#             indices: np.array,
#             instance_data: np.array,
#             num_instances: int,
#     ):
#         self.program = program
#         self.position = position
#         self.colors = colors
#         self.indices = indices
#         self.num_instances = num_instances
#         self.instance_data = instance_data
#
#         self.num_indices = len(self.indices)
#         self.vao = None
#         self.buffers = []
#         self.counter = 0.0
#
#         #self._create_vao()
#         self._create_buffers()
#
#     def _create_vao(self):
#         self.program.use()
#
#         # Create and bind VAO
#         self.vao = GLuint()
#         glGenVertexArrays(1, ctypes.byref(self.vao))
#         glBindVertexArray(self.vao)
#
#     def _create_buffers(self):
#         """
#         TODO: Hardcoded, make it programmable
#         :return:
#         """
#         # Get attribute locations
#         position_location = glGetAttribLocation(self.program.id, b'position')
#         color_location = glGetAttribLocation(self.program.id, b'color')
#         data_location = glGetAttribLocation(self.program.id, b'instance_data')
#
#         # Create and bind VAO
#         vao = GLuint()
#         glGenVertexArrays(1, vao)
#         glBindVertexArray(vao)
#
#         # Create and populate VBO for positions
#         vbo = GLuint()
#         glGenBuffers(1, vbo)
#         glBindBuffer(GL_ARRAY_BUFFER, vbo)
#         glBufferData(
#             GL_ARRAY_BUFFER,
#             self.position.nbytes,
#             self.position.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#             GL_STATIC_DRAW
#         )
#
#         # create and populate VBO for colors
#         cbo = GLuint()
#         glGenBuffers(1, cbo)
#         glBindBuffer(GL_ARRAY_BUFFER, cbo)
#         glBufferData(
#             GL_ARRAY_BUFFER,
#             self.colors.nbytes,
#             self.colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#             GL_STATIC_DRAW
#         )
#
#         # Create and populate VBO for instance data
#         self.instance_vbo = GLuint()
#         glGenBuffers(1, ctypes.byref(self.instance_vbo))
#         glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
#         glBufferData(
#             GL_ARRAY_BUFFER,
#             self.instance_data.nbytes,
#             self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#             GL_STATIC_DRAW
#         )
#
#         # Create and populate EBO (Element Buffer Object)
#         ebo = GLuint()
#         glGenBuffers(1, ctypes.byref(ebo))
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
#         glBufferData(
#             GL_ELEMENT_ARRAY_BUFFER,
#             self.indices.nbytes,
#             self.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
#             GL_STATIC_DRAW
#         )
#
#         # Configure vertex attributes
#         # Position attribute
#         glBindBuffer(GL_ARRAY_BUFFER, vbo)
#         glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, 3 * 4, 0)
#         glEnableVertexAttribArray(position_location)
#
#         # Color attribute
#         glBindBuffer(GL_ARRAY_BUFFER, cbo)
#         glVertexAttribPointer(color_location, 3, GL_FLOAT, GL_TRUE, 3 * 4, 0)
#         glEnableVertexAttribArray(color_location)
#
#         # Instance transformation attribute
#         glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
#         glVertexAttribPointer(data_location, 4, GL_FLOAT, GL_FALSE, 4 * 4, 0)
#         glEnableVertexAttribArray(data_location)
#         glVertexAttribDivisor(data_location, 1)  # This makes it per-instance
#
#     def update(self):
#         self.counter += 1
#         time_value = self.counter * 0.01
#
#         index = 0
#         for y in range(6):
#             for x in range(6):
#                 # Position (in a grid pattern)
#                 self.instance_data[index, 0] = (x - 10) * 0.1  # x position
#                 self.instance_data[index, 1] = (y - 10) * 0.1  # y position
#
#                 # Scale (varies based on position)
#                 distance = math.sqrt((x - 5) ** 2 + (y - 5) ** 2) / 7.0
#                 self.instance_data[index, 2] = 0.05 + 0.03 * math.sin(distance * 3 + time_value)
#
#                 # Rotation (animated over time, different for each instance)
#                 self.instance_data[index, 3] = time_value + distance * 2
#
#                 index += 1
#
#         # Update the instance VBO with new data
#         glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
#         glBufferSubData(GL_ARRAY_BUFFER, 0, self.instance_data.nbytes,
#                         self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
#
#
#     @staticmethod
#     def create_vbo(
#             program: ShaderProgram,
#             attribute: str,
#             data: np.array,
#             size=3,
#             buffer_type=GL_ARRAY_BUFFER,
#             data_type=GL_FLOAT,
#             normalize=GL_FALSE,
#             instance_data=False,
#     ) -> GLuint:
#         # get attribute location in the shader
#         attr = attribute.encode('utf-8')
#         location = glGetAttribLocation(program.id, attr)
#
#         # create, bind and populate VBO
#         vbo = GLuint()
#         glGenBuffers(1, vbo)
#         glBindBuffer(buffer_type, vbo)
#         glBufferData(
#             buffer_type,
#             data.nbytes,
#             data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
#             GL_STATIC_DRAW
#         )
#
#         # configure attributes
#         if data_type == GL_ARRAY_BUFFER:
#             glVertexAttribPointer(location, size, data_type, normalize, size * 4, 0)
#             glEnableVertexAttribArray(location)
#
#             # divide data per instance, for instanced rendering
#             if instance_data:
#                 glVertexAttribDivisor(location, 1)
#
#         return vbo
#
#     def delete(self):
#         glDeleteVertexArrays(1, ctypes.byref(self.vao))
#         for buffer in self.buffers:
#             glDeleteBuffers(1, ctypes.byref(buffer))
#
#     def draw(self):
#         self.program.use()
#         glBindVertexArray(self.vao)
#         glDrawElementsInstanced(
#             GL_TRIANGLES,
#             self.num_indices,
#             GL_UNSIGNED_INT,
#             None,
#             self.num_instances
#         )

vertex_source = """
#version 150 core

in vec3 position;
in vec4 colors;
in vec4 instance_data;  // pos_x, pos_y, scale, rotation

out vec4 fragColor;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

void main() {
    // Extract transformation data
    vec3 offset = vec3(instance_data.xy, position.z*5);
    float angle = instance_data.z;
    float scale = instance_data.w;

    // Apply rotation
    float cos_rot = cos(angle);
    float sin_rot = sin(angle);
    mat3 rotMatrix = mat3(cos_rot, -sin_rot, 0, sin_rot, cos_rot, 0, 0, 0, 0);

    // Transform the vertex: scale -> rotate -> translate
    vec3 scaledPos = position.xyz * scale;
    vec3 rotatedPos = rotMatrix * scaledPos;
    vec3 finalPos = rotatedPos + offset;

    gl_Position = window.projection * window.view * vec4(finalPos, 1.0);
    fragColor = colors;
}
"""

fragment_source = """
#version 150 core
in vec4 fragColor;
in vec2 fragCoord;

out vec4 outColor;

void main() {
    outColor = vec4(fragColor);
}
"""


class InstanceRendering:
    """
    Efficient way to draw thousands of instances of a single 3D mesh,
    animated with different instance data.
    Set and unset uniforms in your shader before and after drawing with passed 'group'.
    Implement data extraction and transformation in your shader.
    Execute your custom 'update_func' to update shader instance data with 'update()',
    before calling 'draw()' in your Window.on_draw().
    """
    def __init__(
            self,
            position: NDArray[np.float32],
            indices: NDArray[np.uint32],
            instance_data: NDArray[np.float32],
            num_instances: int,
            color: Optional[NDArray[np.float32]] = None,
            tex_coord: Optional[NDArray[np.float32]] = None,
            normal: Optional[NDArray[np.float32]] = None,
            program: Optional[ShaderProgram] = None,
            group: Optional[Group] = None,
            update_func: Optional[Callable] = None
    ):
        self.instance_data = instance_data
        self.num_instances = num_instances
        self.num_indices = len(indices)
        self.color = color if color is not None else np.array((1.0, 1.0, 1.0, 1.0) * (len(position) // 3), dtype=np.float32)
        self.tex_coord = tex_coord
        self.normal = normal
        if program:
            self.program = program
        else:
            self.program = ShaderProgram(
                Shader(vertex_source, 'vertex'),
                Shader(fragment_source, 'fragment')
            )

        self.group = group if group else Group()
        self.func = update_func

        self.program.use()
        position_location = glGetAttribLocation(self.program.id, b'position')
        color_location = glGetAttribLocation(self.program.id, b'colors')
        data_location = glGetAttribLocation(self.program.id, b'instance_data')
        tex_location = glGetAttribLocation(self.program.id, b'tex_coords')
        nor_location = glGetAttribLocation(self.program.id, b'normals')

        # Create and bind VAO
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)

        # Create and populate VBO for positions
        self.pos_vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.pos_vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            position.nbytes,
            position.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            GL_STATIC_DRAW
        )

        # Configure position attribute
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, 3 * 4, 0)
        glEnableVertexAttribArray(position_location)

        # Create and populate VBO for colors
        self.color_vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.color_vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.color.nbytes,
            self.color.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            GL_STATIC_DRAW
        )

        # Configure color attribute
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glVertexAttribPointer(color_location, 3, GL_FLOAT, GL_FALSE, 3 * 4, 0)
        glEnableVertexAttribArray(color_location)

        # Create and populate VBO for instance data
        self.instance_vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.instance_vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.instance_data.nbytes,
            self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            GL_STATIC_DRAW
        )

        # Instance transformation attribute
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glVertexAttribPointer(data_location, 4, GL_FLOAT, GL_FALSE, 4 * 4, 0)
        glEnableVertexAttribArray(data_location)
        # This tells the GPU to read a new value of instance_data once per instance, not per vertex.
        glVertexAttribDivisor(data_location, 1)

        # Create and populate EBO (Element Buffer Object)
        self.ebo = GLuint()
        glGenBuffers(1, ctypes.byref(self.ebo))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            GL_STATIC_DRAW
        )

        if tex_coord is not None and program is not None:
            # Create and populate VBO for texture coordinates
            self.tex_vbo = GLuint()
            glGenBuffers(1, ctypes.byref(self.tex_vbo))
            glBindBuffer(GL_ARRAY_BUFFER, self.tex_vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                tex_coord.nbytes,
                tex_coord.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                GL_STATIC_DRAW
            )

            # Configure texture coordinates attribute
            glBindBuffer(GL_ARRAY_BUFFER, self.tex_vbo)
            glVertexAttribPointer(tex_location, 2, GL_FLOAT, GL_FALSE, 2 * 4, 0)
            glEnableVertexAttribArray(tex_location)

        if normal is not None and program is not None:
            # Create and populate VBO for normals
            self.nor_vbo = GLuint()
            glGenBuffers(1, ctypes.byref(self.nor_vbo))
            glBindBuffer(GL_ARRAY_BUFFER, self.nor_vbo)
            glBufferData(
                GL_ARRAY_BUFFER,
                self.normal.nbytes,
                self.normal.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                GL_STATIC_DRAW
            )

            # Configure normal attributes
            glBindBuffer(GL_ARRAY_BUFFER, self.nor_vbo)
            glVertexAttribPointer(nor_location, 3, GL_FLOAT, GL_FALSE, 3 * 4, 0)
            glEnableVertexAttribArray(nor_location)

    def draw(self):
        """Call in your Window.on_draw() to draw instances."""
        self.program.use()
        self.group.set_state()
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None, self.num_instances)
        self.group.unset_state()
        self.program.stop()

    def update(self):
        """
        Updates instance_data on the CPU, by calling your custom function, then updates shader attribute.
        Call this before InstanceRendering.draw() in your Window.on_draw() to animate instances.
        """
        if self.func:
            self.func()
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            self.instance_data.nbytes,
            self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    def delete(self):
        """Call when done using this object within the same running context."""
        glDeleteVertexArrays(1, ctypes.byref(self.vao))
        glDeleteBuffers(1, ctypes.byref(self.pos_vbo))
        glDeleteBuffers(1, ctypes.byref(self.color_vbo))
        glDeleteBuffers(1, ctypes.byref(self.ebo))
        glDeleteBuffers(1, ctypes.byref(self.instance_vbo))

