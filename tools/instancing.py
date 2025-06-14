from pyglet.gl import *
import numpy as np
from numpy.typing import NDArray
import ctypes
from pyglet.graphics.shader import ShaderProgram, Shader
from pyglet.graphics import Group
from typing import Optional, Sequence, Callable
from definitions import get_logger

logger = get_logger(__name__)


class InstanceRendering:
    """
    Efficient way to draw many instances of a single 3D mesh, and transform them
    using unique instance data.

    Keys in the vertex_data should match attributes used in the shader.
    Vertex indexing data is expected under 'indices': Sequence[int,...] in the vertex_data.
    Set and unset uniforms in your shader before and after drawing with the passed 'group'.
    Execute your custom 'update_func' to update shader instance data with 'update()',
    before calling 'draw()' in your Window.on_draw().

    Typically, you may want to use 4 x vec4 instance_attributes in your shader,
    then extract data in the vertex shader to form transformation mat4, for example.
    Pass shader attribute names as strings via instance_attributes,
    but pass instance_data as numpy array of 4x4 matrices "shape(num_instances, 16)".
    """
    def __init__(
            self,
            vertex_data: dict[str, list | np.ndarray],
            instance_data: NDArray[np.float32],
            num_instances: int,
            instance_attributes: Sequence[str],
            program: Optional[ShaderProgram],
            group: Optional[Group] = None,
            update_func: Optional[Callable] = None
    ):
        self.vertex_data = vertex_data
        self.instance_data = instance_data
        self.num_instances = num_instances
        self.instance_attributes = instance_attributes
        self.program = program

        self.attributes = [attr for attr in self.vertex_data.keys() if attr != 'indices']

        self.group = group if group else Group()
        self.func = update_func

        try:
            self.indices = np.array(self.vertex_data['indices'], dtype=np.uint32)
        except KeyError:
            raise KeyError("Key/value 'indices': Sequence[int,...] is expected in vertex_data.")

        self.num_indices = len(self.indices)

        self.program.use()

        # Create and bind VAO
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)

        # create buffers from passed data
        self.ebo = self._create_element_buffer_object()
        self.vbos = self._create_vertex_buffer_objects()
        self.instance_vbo = self._create_instance_buffer_object()

        all_passed_attributes = self.attributes + list(self.instance_attributes)
        for shader_attribute in self.program.attributes.keys():
            if shader_attribute not in all_passed_attributes:
                logger.info(f"Shader attribute '{shader_attribute}' is declared in the vertex shader, but got no data.")

    def _create_element_buffer_object(self) -> GLuint:
        ebo = GLuint()
        glGenBuffers(1, ctypes.byref(ebo))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.indices.nbytes,
            self.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            GL_STATIC_DRAW
        )
        return ebo

    def _create_instance_buffer_object(self) -> GLuint:
        # creating buffer
        instance_vbo = GLuint()
        glGenBuffers(1, ctypes.byref(instance_vbo))
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)

        # populating buffer data
        glBufferData(
            GL_ARRAY_BUFFER,
            self.instance_data.nbytes,
            self.instance_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            GL_DYNAMIC_DRAW
        )

        # configuring attributes
        for i, attr in enumerate(self.instance_attributes):
            location = glGetAttribLocation(self.program.id, attr.encode('utf-8'))
            if location == -1:
                logger.info(f" Data set '{attr}' has no corresponding vertex shader attribute (it may be optimized or not declared).")
                continue

            size = self.program.attributes[attr]['count']
            stride = len(self.instance_attributes) * size

            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, stride * 4, ctypes.c_void_p(i * size * 4))
            glEnableVertexAttribArray(location)
            # This tells the GPU to read a new value of instance_data once per instance, not per vertex.
            glVertexAttribDivisor(location, 1)

        return instance_vbo

    def _create_vertex_buffer_objects(self) -> list[GLuint]:
        vbos = []
        for attr in self.attributes:
            location = glGetAttribLocation(self.program.id, attr.encode('utf-8'))
            if location == -1:
                logger.info(f" Data set '{attr}' has no corresponding vertex shader attribute (it may be optimized or not declared).")
                continue

            data = np.array(self.vertex_data[attr], dtype=np.float32)
            size = self.program.attributes[attr]['count']

            # generating buffer
            vbo = GLuint()
            glGenBuffers(1, ctypes.byref(vbo))
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            # populating buffer data
            glBufferData(
                GL_ARRAY_BUFFER,
                data.nbytes,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                GL_STATIC_DRAW
            )
            # configuring attributes
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, size * 4, 0)
            glEnableVertexAttribArray(location)
            vbos.append(vbo)

        return vbos

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
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        for buffer in self.vbos:
            glDeleteBuffers(1, ctypes.byref(buffer))
        glDeleteBuffers(1, ctypes.byref(self.instance_vbo))
        glDeleteBuffers(1, ctypes.byref(self.ebo))
        glDeleteVertexArrays(1, ctypes.byref(self.vao))

