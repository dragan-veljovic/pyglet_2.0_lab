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
import warnings


def custom_warning_format(message, category, filename, lineno, line=None):
    return f"⚠️ {category.__name__}: {message}\n"

warnings.formatwarning = custom_warning_format


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

    Notes:
    Designed to work with the return of load_obj_model(). You can pass dictionary with
    model data, including "indices" as glDrawElementsInstanced is used for instance rendering.
    Keys in the dictionary should match attributes in the shader program.
    """
    def __init__(
            self,
            model_data: dict,
            instance_data: NDArray[np.float32],
            num_instances: int,
            program: Optional[ShaderProgram] = None,
            group: Optional[Group] = None,
            update_func: Optional[Callable] = None
    ):
        self.model_data = model_data
        self.instance_data = instance_data
        self.num_instances = num_instances
        if program:
            self.program = program
        else:
            self.program = ShaderProgram(
                Shader(vertex_source, 'vertex'),
                Shader(fragment_source, 'fragment')
            )

        self.attributes = [attr for attr in self.model_data.keys() if attr != 'indices']

        self.group = group if group else Group()
        self.func = update_func
        self.indices = np.array(self.model_data['indices'], dtype=np.uint32)
        self.num_indices = len(self.indices)

        self.program.use()

        self.locations = {}
        for attr in self.attributes[:]:
            location = glGetAttribLocation(self.program.id, attr.encode('utf-8'))
            if location != -1:
                self.locations[attr] = location
            else:
                warnings.warn(f"Vertex attribute '{attr}' is not found in the shader program and will not be used!")
                self.attributes.remove(attr)

        # Create and bind VAO
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)

        # Create and populate EBO (Element Buffer Object)
        self.ebo = GLuint()
        glGenBuffers(1, ctypes.byref(self.ebo))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.indices.nbytes,
            self.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            GL_STATIC_DRAW
        )

        # Create and populate VBO for instance data
        # User must rename per instance data attribute as "model_data" in the shader
        # self.instance_data_loc = glGetAttribLocation(self.program.id, b'instance_data')
        # if self.instance_data_loc == -1:
        #     raise AttributeError("Expected 'instance_data' attribute is not found in the shader program.")

        self.instance_vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.instance_vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.instance_data.nbytes,
            self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            GL_DYNAMIC_DRAW
        )

        vec4_size = 4 * 4  # 16 bytes
        stride = 4 * 4 * 4  # 64 bytes for 4x4 matrix
        start_loc = glGetAttribLocation(self.program.id, b'instance_data_0')

        for i in range(4):
            loc = start_loc + i
            # Instance data configuration
            glVertexAttribPointer(
                loc, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(i * vec4_size)
            )
            glEnableVertexAttribArray(loc)
            # This tells the GPU to read a new value of instance_data once per instance, not per vertex.
            glVertexAttribDivisor(loc, 1)

        # create, populate and configure VBO for vertex attributes
        self.vbos = []
        for attr in self.attributes:
            # prepare data
            vbo = GLuint()
            location = self.locations[attr]
            data = np.array(self.model_data[attr], dtype=np.float32)
            size = 2 if attr == 'tex_coord' else 3  # consider passing ('type', data) as for pyglet vlists
            # generate buffer
            glGenBuffers(1, ctypes.byref(vbo))
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            # populate buffer data
            glBufferData(
                GL_ARRAY_BUFFER,
                data.nbytes,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                GL_STATIC_DRAW
            )
            # configure buffer data
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, size*4, 0)
            glEnableVertexAttribArray(location)
            self.vbos.append(vbo)

    def draw(self):
        """Call in your Window.on_draw() to draw instances."""
        self.program.use()
        self.group.set_state_recursive()

        glBindVertexArray(self.vao)
        glDrawElementsInstanced(
            GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None, self.num_instances
        )

        self.group.unset_state_recursive()
        self.program.stop()

    def update(self):
        """
        Updates instance_data on the CPU by calling your custom function, then updates shader attribute.
        Call this before InstanceRendering.draw() in your Window.on_draw() to animate instances.
        """
        if self.func:
            self.func()
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        # Orphan the buffer (discard old data to avoid GPU stalls)
        glBufferData(GL_ARRAY_BUFFER, self.instance_data.nbytes, None, GL_DYNAMIC_DRAW)
        # Upload the new instance data
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            self.instance_data.nbytes,
            self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    def delete(self):
        """Call when done using this object within the same running context."""
        glDeleteVertexArrays(1, ctypes.byref(self.vao))
        glDeleteBuffers(1, ctypes.byref(self.ebo))
        glDeleteBuffers(1, ctypes.byref(self.instance_vbo))
        for buffer in self.vbos:
            glDeleteBuffers(1, ctypes.byref(buffer))
