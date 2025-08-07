import hashlib
import math
import pickle
from pathlib import Path
from pyglet.graphics.shader import ShaderProgram, UniformBufferObject
from pyglet.graphics import Group, Batch
from pyglet.graphics.vertexdomain import IndexedVertexList, VertexList
from pyglet.image import Texture
from pyglet.gl import *
from pyglet.math import Vec3, Mat4, Vec4
from typing import Sequence
from concurrent.futures import ThreadPoolExecutor

from tools.interface import Selectable

executor = ThreadPoolExecutor()

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')
logger = logging.getLogger(__name__)


class Mesh(Selectable):
    _id_counter = 0

    def __init__(
            self,
            data: dict,
            program: ShaderProgram,
            batch: Batch,
            group: Group | None = None,
            dynamic=True
    ):
        """
        A single set of vertex data, with its own transformation and optional group for textures/materials.
        Provides freeze() and unfreeze() optimization methods to make mesh static or dynamic in the scene on demand.
        If Mesh is dynamic, set 'matrix' attribute to transform it.

        :param data: A dictionary of prepared vertex data for VertexListIndexed instance creation.
                     Expected necessary key names: 'indices', 'position', 'normal', 'tangent', 'bitangent'.
        :param program: A ShaderProgram to attach vertex list to. Declared attributes must match key names.
        :param batch: A pyglet.graphics.Batch() that this vertex list belongs to.
        :param group: Optional Group slot, for example to pass textures and/or materials
        :param dynamic: Set this flag to False if you expect your object not to move  most of the time.
                        Can be changed at any time through freeze(), unfreeze().

        TODO: "dynamic" as a property instead of two methods?
        TODO: inefficient freeze(), a lot of Vec4s, adjusted transform_model_data() should work with dict directly.
        TODO: another group updating uniform (block?) of parameters, allowing for transformation on the GPU
        TODO: trans param added, but what happens with them when model if freezed?
        """
        Mesh._id_counter += 1
        self.id = Mesh._id_counter

        self._data = data if dynamic else None
        self._program = program
        self._batch = batch
        self._parent_group = group
        self._dynamic_group = DynamicRenderGroup(
            self, self._program, order=self._parent_group.order, parent=self._parent_group
        )
        self._group = self._dynamic_group if dynamic else self._parent_group
        self._dynamic = dynamic
        self._vertex_list = get_vertex_list(data, self._program, self._batch, self._group)

        # transformation parameters
        self._matrix = Mat4()
        self._position = Vec3(0, 0, 0)
        self._rotation = 0.0
        self._rotation_dir = Vec3(0, 1, 0)
        self._scale = Vec3(1.0, 1.0, 1.0)
        self._origin = Vec3(0, 0, 0)
        self._dirty = False

        super().__init__(batch=self._batch)

    @property
    def matrix(self) -> Mat4:
        """Matrix is updated through self._group, before draw,
        and only if any positional parameter has been changed."""
        if self._dirty:
            self._matrix = get_model_matrix(
                self._position, self._rotation, self._rotation_dir, self._scale, self._origin
            )
            self._dirty = False
        return self._matrix

    @property
    def position(self) -> Vec3:
        return self._position

    @position.setter
    def position(self, value: Vec3):
        self._position = value
        self._dirty = True

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, value: float):
        self._rotation = value
        self._dirty = True

    @property
    def rotation_dir(self) -> Vec3:
        return self._rotation_dir

    @rotation_dir.setter
    def rotation_dir(self, value: Vec3):
        self._rotation_dir = value
        self._dirty = True

    @property
    def scale(self) -> Vec3:
        return self._scale

    @scale.setter
    def scale(self, value: Vec3):
        self._scale = value
        self._dirty = True

    @property
    def origin(self) -> Vec3:
        return self._origin

    @origin.setter
    def origin(self, value: Vec3):
        self._origin = value
        self._dirty = True

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    def freeze(self):
        """
        Bake current transformations into the vertex data, making 3D object static in the scene.
        This eliminates need for shader uniform update on every frame, but requires high one-time CPU work.
        Static meshes can be made dynamic or static again on demand with unfreeze() or freeze() methods respectively.
        """
        if not self._dynamic:
            return

        self._data = self.get_vertex_data()

        transform_matrix = self._matrix
        adjusted_transform_matrix = self._matrix.__invert__().transpose()

        transformed_positions = []
        transformed_normals = []
        transformed_tangents = []
        transformed_bitangents = []

        # Transform each vertex position
        for i in range(0, len(self._data['position']), 3):
            vertex = Vec4(*self._data['position'][i:i + 3], 1)
            normal = Vec4(*self._data['normal'][i:i + 3], 0)
            new_position = transform_matrix @ vertex
            new_normal = adjusted_transform_matrix @ normal
            transformed_positions.extend(new_position[:3])
            transformed_normals.extend(new_normal[:3])

            if 'tangent' in self._data:
                tangent = Vec4(*self._data['tangent'][i:i + 3], 0)
                bitangent = Vec4(*self._data['bitangent'][i:i + 3], 0)
                new_tangent = adjusted_transform_matrix @ tangent
                new_bitangent = adjusted_transform_matrix @ bitangent
                transformed_tangents.extend(new_tangent[:3])
                transformed_bitangents.extend(new_bitangent[:3])

        self._data.update({
            'position': transformed_positions,
            'normal': transformed_normals,

        })

        if 'tangent' in self._data:
            self._data.update({
                'tangent': transformed_tangents,
                'bitangent': transformed_bitangents
            })

        self._vertex_list.delete()  # free gpu memory
        self._group = self._parent_group

        self._vertex_list = get_vertex_list(
            self._data,
            self._program,
            self._batch,
            self._group,
        )

        self._dynamic = False
        self._data = None  # free system memory

        # debugging
        self.origin += self._position
        self._position = Vec3(0, 0, 0)
        self._rotation = 0.0
        self._rotation_dir = Vec3(0, 1, 0)
        self._scale = Vec3(1.0, 1.0, 1.0)

    def unfreeze(self):
        """
        Make mesh dynamic again, re-assigning the group that updates mesh matrix to shader before every draw call.
        Simply set new Mesh.matrix in on_draw() to see changes applied.
        """
        if self._dynamic:
            return

        self._data = self.get_vertex_data()
        self._vertex_list.delete()
        self._group = self._dynamic_group

        self._vertex_list = get_vertex_list(self._data, self._program, self._batch, self._group)
        self._dynamic = True

    def get_vertex_data(self) -> dict:
        """Extract vertex data from the vertex list and return as a dictionary."""
        data = {}
        for name in self._vertex_list.domain.attribute_names:
            if 'instance_data' not in name:
                data[name] = getattr(self._vertex_list, name)[:]
        data['indices'] = self._vertex_list.indices[:]
        return data

    def get_aabb_min_max(self) -> tuple[Vec3, Vec3]:
        """Bounding box is calculated once, then transformed with update_bounding_box()."""
        positions = self._vertex_list.position[:]

        x_coords = positions[0::3]
        y_coords = positions[1::3]
        z_coords = positions[2::3]

        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        zmin, zmax = min(z_coords), max(z_coords)

        return Vec3(xmin, ymin, zmin), Vec3(xmax, ymax, zmax)

    def __eq__(self, other: 'Mesh'):
        return isinstance(other, Mesh) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Plane(Mesh):
    """A rectangular plane with support for texturing and advanced lighting.
    TODO: winding problem
    """
    def __init__(
            self,
            program: ShaderProgram,
            batch: Batch,
            group: Group,
            position=Vec3(0, 0, 0),
            length=100,
            width=100,
            centered=False,
            color=(1.0, 1.0, 1.0, 1.0)
    ):
        self.program = program
        self.position = position
        self.length, self.width = length, width
        self.centered = centered

        positions, indices, tex_coords = self._generate_vertex_data()
        normals = calculate_normals(positions, indices)
        tangents, bitangents = calculate_tangents(positions, normals, tex_coords, indices)

        data = {
            'indices': indices,
            'position': positions,
            'tex_coord': tex_coords,
            'color': color * (len(positions) // 3),
            'normal': normals,
            'tangent': tangents,
            'bitangent': bitangents
        }

        super().__init__(data, program, batch, group, dynamic=True)

    def _generate_vertex_data(self) -> tuple[tuple, tuple, tuple]:
        l, w = self.length, self.width
        if self.centered:
            x, y, z = self.position.x - l / 2, self.position.y, self.position.z - w / 2
        else:
            x, y, z = self.position.x, self.position.y, self.position.z
        positions = (
            x, y, z,
            x + l, y, z,
            x + l, y, z + w,
            x, y, z + w
        )
        indices = (0, 2, 1, 0, 3, 2)
        tex_coords = (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)

        return positions, indices, tex_coords


class Sphere(Mesh):
    """A sphere model with support for texturing and advanced lighting."""

    def __init__(
            self,
            program: ShaderProgram,
            batch: Batch,
            group: Group | None = None,
            position=Vec3(0, 0, 0),
            radius=50,
            lat_segments=16,
            lng_segments=32,
            color=(1.0, 1.0, 1.0, 1.0)
    ):
        self.position = position
        self.radius = radius
        self.lat_segments = lat_segments
        self.lng_segments = lng_segments

        positions, indices, tex_coords = self._generate_vertex_data()

        # Calculate normals using your existing function
        normals = calculate_normals(positions, indices)

        # Fix seam normals by averaging with corresponding vertices
        self._fix_seam_normals(normals)

        tangents, bitangents = calculate_tangents(positions, normals, tex_coords, indices)

        data = {
            'indices': indices,
            'position': positions,
            'tex_coord': tex_coords,
            'color': color * (len(positions) // 3),
            'normal': normals,
            'tangent': tangents,
            'bitangent': bitangents
        }

        super().__init__(data, program, batch, group)

    def _generate_vertex_data(self) -> tuple:
        """Generate vertices, indices, and texture coordinates in a single pass."""
        vertices = []
        tex_coords = []
        indices = []

        x, y, z = self.position.x, self.position.y, self.position.z

        # Generate vertices, texture coordinates, and indices in a single loop
        for lat in range(self.lat_segments + 1):
            # Latitude angle from 0 to π (top to bottom)
            lat_angle = math.pi * lat / self.lat_segments
            sin_lat = math.sin(lat_angle)
            cos_lat = math.cos(lat_angle)
            v = lat / self.lat_segments

            for lng in range(self.lng_segments + 1):
                # Longitude angle from 0 to 2π (around)
                lng_angle = 2 * math.pi * lng / self.lng_segments
                sin_lng = math.sin(lng_angle)
                cos_lng = math.cos(lng_angle)

                # Calculate vertex position
                vertex_x = x + self.radius * sin_lat * cos_lng
                vertex_y = y + self.radius * cos_lat
                vertex_z = z + self.radius * sin_lat * sin_lng

                vertices.extend([vertex_x, vertex_y, vertex_z])

                # Calculate texture coordinates
                u = min(lng / self.lng_segments, 1.0)
                tex_coords.extend([u, v])

                # Generate indices (skip last longitude segment and last latitude segment)
                if lat < self.lat_segments and lng < self.lng_segments:
                    # Current vertex and next vertex indices
                    current = lat * (self.lng_segments + 1) + lng
                    next_lng = lat * (self.lng_segments + 1) + lng + 1
                    next_lat = (lat + 1) * (self.lng_segments + 1) + lng
                    next_both = (lat + 1) * (self.lng_segments + 1) + lng + 1

                    # Skip triangles at the poles to avoid degenerate triangles
                    if lat == 0:
                        # Top cap - only create one triangle per segment
                        indices.extend([current, next_both, next_lat])
                    elif lat == self.lat_segments - 1:
                        # Bottom cap - only create one triangle per segment
                        indices.extend([current, next_lng, next_both])
                    else:
                        # Middle sections - create two triangles per quad
                        indices.extend([current, next_lng, next_both])
                        indices.extend([current, next_both, next_lat])

        return tuple(vertices), tuple(indices), tuple(tex_coords)

    def _fix_seam_normals(self, normals):
        """Fix normals at the seam by averaging with corresponding vertices."""
        # For each latitude row, average the normal of the first and last longitude vertices
        for lat in range(self.lat_segments + 1):
            # Index of first vertex in this latitude row (lng=0)
            first_idx = lat * (self.lng_segments + 1)
            # Index of last vertex in this latitude row (lng=lng_segments)
            last_idx = lat * (self.lng_segments + 1) + self.lng_segments

            # Get the normals
            first_normal = normals[first_idx * 3:first_idx * 3 + 3]
            last_normal = normals[last_idx * 3:last_idx * 3 + 3]

            # Average them
            avg_normal = [
                (first_normal[0] + last_normal[0]) / 2,
                (first_normal[1] + last_normal[1]) / 2,
                (first_normal[2] + last_normal[2]) / 2
            ]

            # Normalize
            length = (avg_normal[0] ** 2 + avg_normal[1] ** 2 + avg_normal[2] ** 2) ** 0.5
            if length > 0:
                avg_normal = [avg_normal[0] / length, avg_normal[1] / length, avg_normal[2] / length]

            # Apply to both vertices
            for i in range(3):
                normals[first_idx * 3 + i] = avg_normal[i]
                normals[last_idx * 3 + i] = avg_normal[i]


class GridMesh(Mesh):
    """Tessellated rectangular grid, split into triangles.
    TODO: winding problem
    """
    def __init__(
            self,
            program: ShaderProgram,
            batch: Batch,
            length=300, width=200,
            columns=6, rows=4,
            group: Group | None = None,
            position=Vec3(0, 0, 0),
            color=(1.0, 1.0, 1.0, 1.0)
    ):
        self._length = length
        self._width = width
        self._pos = position
        self._rows = rows
        self._cols = columns
        self._batch = batch
        self._group = group
        self._program = program

        indices, positions, tex_coords = self._get_data()
        normals = calculate_normals(positions, indices)
        tangents, bitangents = calculate_tangents(positions, normals, tex_coords, indices)

        data = {
            'indices': indices,
            'position': positions,
            'tex_coord': tex_coords,
            'color': color * (len(positions) // 3),
            'normal': normals,
            'tangent': tangents,
            'bitangent': bitangents
        }

        super().__init__(data, program, batch, group)

    def _get_data(self) -> tuple[list, list, list]:
        cols, rows = self._cols, self._rows
        length, width = self._length, self._width
        pos = self._pos
        delta_x = length / (cols - 1)
        delta_z = width / (rows - 1)

        position = []
        indices = []
        tex_coord = []

        for j in range(rows):
            for i in range(cols):
                x = pos.x + i * delta_x
                y = pos.y
                z = pos.z + j * delta_z
                position.extend((x, y, z))

                u = i * delta_x / length
                v = j * delta_z / width
                tex_coord.extend((u, v))

                if i < cols - 1 and j < rows - 1:
                    # Quad points A(0,0), B(1,0), C(0,1), D(1,1) form ABC and BDC triangles
                    A = i + j * cols  # i + j*c
                    B = A + 1  # i + 1 + j*c
                    C = A + cols  # i + (j + 1)*c
                    D = C + 1  # i + 1 + (j + 1)*c
                    indices.extend((A, C, B, B, C, D))

        return indices, position, tex_coord


class Cuboid(Mesh):
    """A cube model with support for texturing and advanced lighting."""

    def __init__(
            self,
            program: ShaderProgram,
            batch: Batch,
            group: Group | None = None,
            position=Vec3(0, 0, 0),
            size=(100, 100, 100),
            color=(1.0, 1.0, 1.0, 1.0)
    ):
        self.position = position
        self.size = size

        positions = self._get_vertices()
        indices = self._get_indices()
        tex_coords = self._get_tex_coords()

        normals = calculate_normals(positions, indices)
        tangents, bitangents = calculate_tangents(positions, normals, tex_coords, indices)

        data = {
            'indices': indices,
            'position': positions,
            'tex_coord': tex_coords,
            'color': color * (len(positions) // 3),
            'normal': normals,
            'tangent': tangents,
            'bitangent': bitangents
        }

        super().__init__(data, program, batch, group)

    def _get_vertices(self) -> tuple:
        x, y, z = self.position
        lxh, lyh, lzh = self.size[0] / 2, self.size[1] / 2, self.size[2] / 2

        # 4 vertices per face, 6 faces
        vertices = (
            # Front face (+Z)
            x - lxh, y - lyh, z + lzh,
            x + lxh, y - lyh, z + lzh,
            x + lxh, y + lyh, z + lzh,
            x - lxh, y + lyh, z + lzh,

            # Back face (-Z)
            x + lxh, y - lyh, z - lzh,
            x - lxh, y - lyh, z - lzh,
            x - lxh, y + lyh, z - lzh,
            x + lxh, y + lyh, z - lzh,

            # Left face (-X)
            x - lxh, y - lyh, z - lzh,
            x - lxh, y - lyh, z + lzh,
            x - lxh, y + lyh, z + lzh,
            x - lxh, y + lyh, z - lzh,

            # Right face (+X)
            x + lxh, y - lyh, z + lzh,
            x + lxh, y - lyh, z - lzh,
            x + lxh, y + lyh, z - lzh,
            x + lxh, y + lyh, z + lzh,

            # Top face (+Y)
            x - lxh, y + lyh, z + lzh,
            x + lxh, y + lyh, z + lzh,
            x + lxh, y + lyh, z - lzh,
            x - lxh, y + lyh, z - lzh,

            # Bottom face (-Y)
            x - lxh, y - lyh, z - lzh,
            x + lxh, y - lyh, z - lzh,
            x + lxh, y - lyh, z + lzh,
            x - lxh, y - lyh, z + lzh,
        )
        return vertices

    @staticmethod
    def _get_indices() -> tuple:
        # 2 triangles per face × 6 faces
        indices = (
            # Front face
            0, 1, 2, 0, 2, 3,
            # Back face
            4, 5, 6, 4, 6, 7,
            # Left face
            8, 9, 10, 8, 10, 11,
            # Right face
            12, 13, 14, 12, 14, 15,
            # Top face
            16, 17, 18, 16, 18, 19,
            # Bottom face
            20, 21, 22, 20, 22, 23,
        )
        return indices

    @staticmethod
    def _get_tex_coords() -> tuple:
        # Each face: (0,0)-(1,1)
        face_uvs = (
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        )
        tex_coords = face_uvs * 6  # repeat for all 6 faces
        return tex_coords


class DynamicRenderGroup(Group):
    """Set shader to use CPU precalculated model matrix."""
    def __init__(self, mesh: Mesh, program: ShaderProgram, order=0, parent: Group = None):
        super().__init__(order, parent)
        self.program = program
        self.mesh = mesh

    def set_state(self) -> None:
        self.program['rendering_dynamic_object'] = True
        self.program['model_precalc'] = self.mesh.matrix

    def unset_state(self) -> None:
        self.program['rendering_dynamic_object'] = False

    def __hash__(self):
        return hash((self.program, self.mesh, self.order, self.parent))

    def __eq__(self, other: "DynamicRenderGroup"):
        return (
                self.__class__ == other.__class__ and
                self.mesh is other.mesh and
                self.program is other.program and
                self.parent == other.parent and
                self.order == other.order
        )


class BlendGroup(Group):
    """Blend settings group. Parent is usually ShaderGroup."""
    def __init__(
            self,
            blend_src: int = GL_SRC_ALPHA,
            blend_dest: int = GL_ONE_MINUS_SRC_ALPHA,
            order=0,
            parent: Group | None = None
    ):
        super().__init__(order, parent)
        self.blend_src = blend_src
        self.blend_dest = blend_dest

    def set_state(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def unset_state(self) -> None:
        glDisable(GL_BLEND)

    def __hash__(self):
        return hash((self.blend_src, self.blend_dest, self.order, self.parent))

    def __eq__(self, other: 'BlendGroup'):
        return (self.__class__ is other.__class__ and
                self.blend_src == other.blend_src and
                self.blend_dest == other.blend_dest and
                self.order == other.order and
                self.parent == other.parent
                )


class DiffuseNormalTextureGroup(Group):
    def __init__(
            self,
            diffuse: Texture = None,
            normal: Texture = None,
            program: ShaderProgram = None,
            transparency: bool = False,
            order=0, parent=None
    ):
        """
        A Group that enables and binds a diffuse and normal map to a ShaderProgram.
        TextureGroups are equal if their Texture and ShaderProgram
        are equal.
        :param diffuse: a diffuse Texture to bind.
        :param normal: a normal Texture to bind.
        :param program: Shader program to use.
        :param order: Change the order to render above or below other Groups.
        :param parent: Parent group.
        """
        super().__init__(order, parent)
        self.diffuse = diffuse
        self.normal = normal
        self.program = program
        self.transparency = transparency
        self.reset_normal_mapping_uniform = False

        glActiveTexture(GL_TEXTURE0)
        self.program['diffuse_texture'] = 0
        glActiveTexture(GL_TEXTURE1)
        self.program['normal_map'] = 1  # Texture unit

    def set_state(self):
        # activate and bind diffuse texture
        if self.diffuse:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(self.diffuse.target, self.diffuse.id)

        # activate and bind normal texture
        if self.program['normal_mapping']:
            if self.normal:
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(self.normal.target, self.normal.id)
            else:
                self.program['normal_mapping'] = False
                self.reset_normal_mapping_uniform = True

    def unset_state(self):
        if self.reset_normal_mapping_uniform:
            self.program['normal_mapping'] = True
            self.reset_normal_mapping_uniform = False

        if self.diffuse:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(self.diffuse_target, 0)
        if self.normal:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(self.normal_target, 0)

    def __hash__(self):
        return hash((self.diffuse_target, self.diffuse_id, self.normal_target, self.normal_id, self.order, self.parent,
                     self.program))

    def __eq__(self, other: "DiffuseNormalTextureGroup"):
        return (self.__class__ is other.__class__ and
                self.normal_target == other.normal_target and
                self.normal.id == other.normal_id and
                self.diffuse_target == other.diffuse_target and
                self.diffuse_id == other.diffuse_id and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)

    @property
    def normal_id(self):
        return self.normal if self.normal else None

    @property
    def normal_target(self):
        return self.normal.target if self.normal else None

    @property
    def diffuse_id(self):
        return self.diffuse.id if self.diffuse else None

    @property
    def diffuse_target(self):
        return self.diffuse.target if self.diffuse else None


class MaterialGroup(Group):
    def __init__(
            self,
            ubo: UniformBufferObject,
            ambient=(0.25, 0.25, 0.25, 1.0),
            diffuse=(0.7, 0.7, 0.7, 1.0),
            specular=(0.75, 0.75, 0.75, 1.0),
            emission=(0.0, 0.0, 0.0, 1.0),
            shininess=32,
            reflection_strength=0.0,  # leave 0.0 for reflections off
            refraction_strength=0.0,  # leave 0.0 for refractions off
            refractive_index=1.52,
            f0_reflectance=(0.04, 0.04, 0.04, 1.0),
            fresnel_power=5.0,
            bump_strength=1.0,
            order=0,
            parent=None,
    ):
        """
        If all RGB values are identical, they can be also assigned with a single float for convenience.
        For example diffuse=0.65 will be equivalent to diffuse=(0.65, 0.65, 0.65, 1.0) through internal conversion.
        """
        super().__init__(order, parent)
        self.ubo = ubo
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.emission = emission
        self.shininess = shininess
        self.reflection_strength = reflection_strength
        self.refraction_strength = refraction_strength
        self.refractive_index = refractive_index
        self.fresnel_power = fresnel_power
        self.f0_reflectance = f0_reflectance
        self.bump_strength = bump_strength

    def update_ubo(self):
        with self.ubo as ubo:
            for param in self.__dict__:
                if param != 'ubo':
                    setattr(ubo, param, getattr(self, param))

    def set_state(self) -> None:
        self.update_ubo()

    @staticmethod
    def _to_vec4(value):
        if isinstance(value, (int, float)):
            return Vec4(value, value, value, 1.0)
        elif isinstance(value, (tuple, list)) and len(value) == 4:
            return Vec4(*value)
        elif isinstance(value, Vec4):
            return value
        else:
            raise ValueError(f"Invalid value: {value}. Use single float or RGBA tuple.")

    def __setattr__(self, name, value):
        if name in ['ambient', 'diffuse', 'specular', 'emission', 'f0_reflectance']:
            value = self._to_vec4(value)
        super().__setattr__(name, value)


def get_model_matrix(
        position=Vec3(0, 0, 0),
        rotation_angle=0.0,
        rotation_dir=Vec3(0, 1, 0),
        scale=Vec3(1, 1, 1),
        origin=Vec3(0, 0, 0),
) -> Mat4:
    """Creates and returns model matrix from given parameters, using :py:mod:`pyglet.math` builtins."""

    identity = Mat4()

    # translation matrix
    translation = Mat4.from_translation(position) if position else identity

    # rotation matrix
    if rotation_angle:
        if rotation_dir is not Vec3(0, 1, 0):
            rotation_dir = rotation_dir.normalize()
        rotation = Mat4.from_rotation(rotation_angle, rotation_dir)
    else:
        rotation = identity

    # scale matrix
    scale = Mat4.from_scale(scale) if scale is not Vec3(1, 1, 1) else identity

    # compensating for origin
    if origin:
        translate_to_origin = Mat4.from_translation(-origin)
        translate_back = Mat4.from_translation(origin)
        return translate_back @ translation @ rotation @ translate_to_origin @ scale

    return translation @ rotation @ scale


def load_mesh(
        path: str,
        program: ShaderProgram,
        batch: Batch,
        group: Group,
        dynamic=True,
        add_tangents=True,
        position: tuple[float, float, float] = None,
        rotation: tuple[float, float, float] = None,
        scale: float = None,
        tex_scale: float = None,
        color: tuple[float, float, float, float] = None,
) -> Mesh:
    """
    Convenience method that loads and transforms OBJ data, then creates and returns Mesh instance.
    Useful when loaded data is used for single 3D Mesh. If loaded data is to be reused, manually call methods:
    data = transform_model_data(load_obj_model, *args)
    m1 = Mesh(data, *args)
    m2 = Mesh(data, *args) ...
    """
    # loading OBJ data
    data = load_obj_model(path)

    # transforming data if any transformation arguments are passed
    all_kwargs = locals()
    transform_param_names = ('position', 'rotation', 'scale', 'tex_scale', 'color')
    active_kwargs = {k: all_kwargs[k] for k in transform_param_names if all_kwargs[k] is not None}

    if active_kwargs:
        data = transform_model_data(data, add_tangents=add_tangents, **active_kwargs)
    else:
        # no transformation, but add tangents
        if add_tangents:
            add_tangents_bitangents(data)

    return Mesh(data, program, batch, group, dynamic)


def load_obj_model(filename) -> dict:
    """
    Load an OBJ model file and prepare vertex data for use with pyglet's VertexList

    Args:
        filename (str): Path to the .obj file

    Returns:
        dict: Dictionary containing flattened vertex data:
            - 'position': Flattened position coordinates [x0, y0, z0, x1, y1, z1, ...]
            - 'normal': Flattened normal vectors (if available)
            - 'tex_coord': Flattened texture coordinates (if available)
            - 'indices': Vertex indices for indexed rendering
    """
    # Temporary storage for the file data
    v_positions = []
    v_normals = []
    v_tex_coords = []

    # Final output vertex data (flattened for pyglet)
    final_positions = []
    final_normals = []
    final_tex_coords = []
    indices = []

    # Vertex cache for indexing optimization
    vertex_cache = {}
    index_count = 0

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':  # Vertex position
                v_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':  # Vertex normal
                v_normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':  # Texture coordinate
                # OBJ format can have 3 components (u, v, w), but we only need u, v
                if len(parts) >= 3:
                    v_tex_coords.append([float(parts[1]), float(parts[2])])
                else:
                    v_tex_coords.append([float(parts[1]), 0.0])
            elif parts[0] == 'f':  # Face
                # Parse face indices, handles different face formats:
                # f v1 v2 v3 [v4...]
                # f v1/vt1 v2/vt2 v3/vt3 [v4/vt4...]
                # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 [v4/vt4/vn4...]
                # f v1//vn1 v2//vn2 v3//vn3 [v4//vn4...]

                face_vertices = []

                # Process each vertex in the face
                for i in range(1, len(parts)):
                    vertex_data = parts[i].split('/')

                    # Extract vertex indices (OBJ indices start at 1, so subtract 1)
                    v_idx = int(vertex_data[0]) - 1 if vertex_data[0] else 0
                    vt_idx = int(vertex_data[1]) - 1 if len(vertex_data) > 1 and vertex_data[1] else -1
                    vn_idx = int(vertex_data[2]) - 1 if len(vertex_data) > 2 and vertex_data[2] else -1

                    # Create a unique key for this vertex combination
                    vertex_key = f"{v_idx}/{vt_idx}/{vn_idx}"

                    # Check if we've seen this vertex before
                    if vertex_key in vertex_cache:
                        # Reuse the existing index
                        face_vertices.append(vertex_cache[vertex_key])
                    else:
                        # Add a new vertex
                        vertex_cache[vertex_key] = index_count
                        face_vertices.append(index_count)
                        index_count += 1

                        # Add the vertex data to our final arrays
                        if v_idx < len(v_positions):
                            pos = v_positions[v_idx]
                            final_positions.extend(pos)

                        if vt_idx >= 0 and vt_idx < len(v_tex_coords):
                            tc = v_tex_coords[vt_idx]
                            final_tex_coords.extend(tc)
                        elif len(v_tex_coords) > 0:
                            # If we have some texture coordinates but not for this vertex
                            final_tex_coords.extend([0.0, 0.0])

                        if vn_idx >= 0 and vn_idx < len(v_normals):
                            norm = v_normals[vn_idx]
                            final_normals.extend(norm)
                        elif len(v_normals) > 0:
                            # If we have some normals but not for this vertex
                            final_normals.extend([0.0, 0.0, 0.0])

                # Triangulate the face
                # For a triangle, we just add the 3 vertices
                # For a quad or n-gon, we triangulate using a fan method
                if len(face_vertices) == 3:
                    indices.extend(face_vertices)
                else:
                    # Triangulate faces with more than 3 vertices (quads, n-gons)
                    for i in range(1, len(face_vertices) - 1):
                        indices.append(face_vertices[0])  # First vertex
                        indices.append(face_vertices[i])  # Current vertex
                        indices.append(face_vertices[i + 1])  # Next vertex

    # If no normals were provided in the file, calculate them
    if not final_normals and final_positions:
        final_normals = calculate_normals(final_positions, indices)

    # If no texture coordinates were provided, create default ones
    if not final_tex_coords and final_positions:
        for _ in range(len(final_positions) // 3):
            final_tex_coords.extend([0.0, 0.0])

    count = len(final_positions) // 3

    return {
        'position': final_positions,
        'normal': final_normals,
        'tex_coord': final_tex_coords,
        'indices': indices,
        'color': (1.0, 1.0, 1.0, 1.0) * count
    }


def calculate_normals(final_positions, indices) -> list:
    """Calculate triangle normals based on positions and indices."""
    final_normals = []
    # Calculate normals for each triangle and average them at shared vertices
    vertex_count = len(final_positions) // 3
    normals_sum = [[0.0, 0.0, 0.0] for _ in range(vertex_count)]
    normals_count = [0] * vertex_count

    # Process each triangle
    for i in range(0, len(indices), 3):
        if i + 2 < len(indices):
            # Get triangle indices
            idx1, idx2, idx3 = indices[i], indices[i + 1], indices[i + 2]

            # Get triangle vertices
            v1 = final_positions[idx1 * 3:idx1 * 3 + 3]
            v2 = final_positions[idx2 * 3:idx2 * 3 + 3]
            v3 = final_positions[idx3 * 3:idx3 * 3 + 3]

            # Calculate triangle edges
            edge1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
            edge2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]

            # Calculate cross product (normal)
            normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            ]

            # Normalize
            length = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
            if length > 0:
                normal = [normal[0] / length, normal[1] / length, normal[2] / length]
            else:
                normal = [0.0, 0.0, 1.0]  # Default normal if calculation failed

            # Add to each vertex of this triangle
            for idx in [idx1, idx2, idx3]:
                for j in range(3):
                    normals_sum[idx][j] += normal[j]
                normals_count[idx] += 1

    # Average the normals and add to final array
    for i in range(vertex_count):
        if normals_count[i] > 0:
            # Average
            for j in range(3):
                normals_sum[i][j] /= normals_count[i]

            # Normalize again
            vec = normals_sum[i]
            length = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
            if length > 0:
                for j in range(3):
                    final_normals.append(vec[j] / length)
            else:
                final_normals.extend([0.0, 0.0, 1.0])
        else:
            final_normals.extend([0.0, 0.0, 1.0])

    return final_normals


def calculate_tangents(positions, normals, tex_coords, indices) -> tuple[list, list]:
    """
    Calculate tangent and bitangent vectors for normal mapping from passed vertex data.

    Args:
        positions: Flattened position data [x0, y0, z0, x1, y1, z1, ...]
        normals: Flattened normal data
        tex_coords: Flattened texture coordinate data [u0, v0, u1, v1, ...]
        indices: Vertex indices

    Returns:
        tuple: (tangents, bitangents) as flattened lists
    """
    vertex_count = len(positions) // 3
    tangents = [0.0] * len(positions)
    bitangents = [0.0] * len(positions)

    # Process each triangle
    for i in range(0, len(indices), 3):
        if i + 2 >= len(indices):
            continue

        i1, i2, i3 = indices[i], indices[i + 1], indices[i + 2]

        # Get positions
        p1 = positions[i1 * 3:i1 * 3 + 3]
        p2 = positions[i2 * 3:i2 * 3 + 3]
        p3 = positions[i3 * 3:i3 * 3 + 3]

        # Get texture coordinates
        uv1 = tex_coords[i1 * 2:i1 * 2 + 2]
        uv2 = tex_coords[i2 * 2:i2 * 2 + 2]
        uv3 = tex_coords[i3 * 2:i3 * 2 + 2]

        # Calculate edges and UV differences
        edge1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        edge2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]]

        duv1 = [uv2[0] - uv1[0], uv2[1] - uv1[1]]
        duv2 = [uv3[0] - uv1[0], uv3[1] - uv1[1]]

        # Calculate tangent and bitangent
        denom = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        if abs(denom) < 0.0001:
            # Skip if the texture coordinates are degenerate
            continue

        r = 1.0 / denom

        tangent = [
            (edge1[0] * duv2[1] - edge2[0] * duv1[1]) * r,
            (edge1[1] * duv2[1] - edge2[1] * duv1[1]) * r,
            (edge1[2] * duv2[1] - edge2[2] * duv1[1]) * r
        ]

        bitangent = [
            (edge2[0] * duv1[0] - edge1[0] * duv2[0]) * r,
            (edge2[1] * duv1[0] - edge1[1] * duv2[0]) * r,
            (edge2[2] * duv1[0] - edge1[2] * duv2[0]) * r
        ]

        # Add to all vertices in this triangle
        for idx in [i1, i2, i3]:
            for j in range(3):
                tangents[idx * 3 + j] += tangent[j]
                bitangents[idx * 3 + j] += bitangent[j]

    # Orthogonalize and normalize tangents/bitangents
    for i in range(vertex_count):
        n = normals[i * 3:i * 3 + 3]
        t = tangents[i * 3:i * 3 + 3]

        # Gram-Schmidt orthogonalize
        dot_product = n[0] * t[0] + n[1] * t[1] + n[2] * t[2]
        t = [t[0] - n[0] * dot_product, t[1] - n[1] * dot_product, t[2] - n[2] * dot_product]

        # Normalize
        length = (t[0] ** 2 + t[1] ** 2 + t[2] ** 2) ** 0.5
        if length > 0.0001:
            for j in range(3):
                tangents[i * 3 + j] = t[j] / length
        else:
            # Default tangent if calculation failed
            tangents[i * 3:i * 3 + 3] = [1.0, 0.0, 0.0]

        # Calculate bitangent as cross product
        cross = [
            n[1] * tangents[i * 3 + 2] - n[2] * tangents[i * 3 + 1],
            n[2] * tangents[i * 3 + 0] - n[0] * tangents[i * 3 + 2],
            n[0] * tangents[i * 3 + 1] - n[1] * tangents[i * 3 + 0]
        ]

        # Normalize
        length = (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5
        if length > 0.0001:
            for j in range(3):
                bitangents[i * 3 + j] = cross[j] / length
        else:
            # Default bitangent if calculation failed
            bitangents[i * 3:i * 3 + 3] = [0.0, 1.0, 0.0]

    return tangents, bitangents


def create_transformation_matrix(position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0):
    """
    Create a 4x4 transformation matrix from position, rotation (Euler angles in degrees), and scale

    Args:
        position (tuple): (x, y, z) translation
        rotation (tuple): (x, y, z) rotation in degrees (Euler angles, XYZ order)
        scale (float or tuple): Uniform scale if float, or (x, y, z) scale if tuple

    Returns:
        list: 4x4 transformation matrix as a flat list in column-major order
        TODO: Rework for efficiency, add origin
    """
    # Convert rotation from degrees to radians
    rx, ry, rz = [math.radians(angle) for angle in rotation]

    # Create rotation matrices for each axis
    # X-axis rotation
    cx, sx = math.cos(rx), math.sin(rx)
    rot_x = [
        1, 0, 0, 0,
        0, cx, sx, 0,
        0, -sx, cx, 0,
        0, 0, 0, 1
    ]

    # Y-axis rotation
    cy, sy = math.cos(ry), math.sin(ry)
    rot_y = [
        cy, 0, -sy, 0,
        0, 1, 0, 0,
        sy, 0, cy, 0,
        0, 0, 0, 1
    ]

    # Z-axis rotation
    cz, sz = math.cos(rz), math.sin(rz)
    rot_z = [
        cz, sz, 0, 0,
        -sz, cz, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]

    # Combine rotation matrices (apply in order: X, Y, Z)
    rot = matrix_multiply(rot_z, matrix_multiply(rot_y, rot_x))

    # Handle both uniform and non-uniform scaling
    sx, sy, sz = (scale, scale, scale) if isinstance(scale, (int, float)) else scale

    # Create full transformation matrix with scale and translation
    matrix = [
        rot[0] * sx, rot[1] * sx, rot[2] * sx, rot[3] * sx,
        rot[4] * sy, rot[5] * sy, rot[6] * sy, rot[7] * sy,
        rot[8] * sz, rot[9] * sz, rot[10] * sz, rot[11] * sz,
        position[0], position[1], position[2], 1.0
    ]

    return matrix


def transform_model_data(
        data: dict,
        position=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=1.0,
        tex_scale: float = None,
        color=(1.0, 1.0, 1.0, 1.0),
        add_tangents=True
) -> dict:
    """
    Calculates tangents, bitangents, normals (if not present)
    and applies transformations to loaded OBJ model data.
    Returns a dictionary of prepared data, ready to use with IndexedVertexList abstraction.
    Use to hardcode transformations into vertex attributes for static objects in the scene.
    """

    # Create transformation matrix
    transform_matrix = create_transformation_matrix(position, rotation, scale)
    model_data = data.copy()

    # Apply transformations to vertex positions and normals
    transformed_positions = []
    transformed_normals = []
    original_positions = model_data['position']
    original_normals = model_data['normal']

    # Transform each vertex position
    for i in range(0, len(original_positions), 3):
        vertex = original_positions[i:i + 3]
        transformed = transform_vertex(vertex, transform_matrix)
        transformed_positions.extend(transformed)
        normal = original_normals[i:i + 3]
        transformed = transform_normal(normal, transform_matrix)
        transformed_normals.extend(transformed)

    count = len(transformed_positions) // 3

    model_data.update({
        'position': transformed_positions,
        'normal': transformed_normals,
        'color': color * count,
    })

    # Recalculate tangents and bitangents using the transformed positions and normals
    if add_tangents:
        add_tangents_bitangents(model_data)

    if tex_scale:
        model_data['tex_coord'] = [element * tex_scale for element in model_data['tex_coord']]

    return model_data


def matrix_multiply(a, b):
    """
    Multiply two 4x4 matrices (stored as flat lists in column-major order)

    Args:
        a (list): First 4x4 matrix
        b (list): Second 4x4 matrix

    Returns:
        list: Resulting 4x4 matrix as a flat list
    """
    result = [0] * 16

    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j]

    return result


def transform_vertex(vertex, matrix):
    """
    Transform a vertex by a 4x4 matrix

    Args:
        vertex (list): [x, y, z] vertex position
        matrix (list): 4x4 transformation matrix as a flat list

    Returns:
        list: Transformed [x, y, z] vertex
    """
    x = vertex[0] * matrix[0] + vertex[1] * matrix[4] + vertex[2] * matrix[8] + matrix[12]
    y = vertex[0] * matrix[1] + vertex[1] * matrix[5] + vertex[2] * matrix[9] + matrix[13]
    z = vertex[0] * matrix[2] + vertex[1] * matrix[6] + vertex[2] * matrix[10] + matrix[14]
    w = vertex[0] * matrix[3] + vertex[1] * matrix[7] + vertex[2] * matrix[11] + matrix[15]

    # Perform perspective division if w is not 1
    if abs(w - 1.0) > 0.00001:
        return [x / w, y / w, z / w]
    return [x, y, z]


def transform_normal(normal, matrix):
    """
    Transform a normal vector by a 4x4 matrix (ignoring translation)

    Args:
        normal (list): [nx, ny, nz] normal vector
        matrix (list): 4x4 transformation matrix as a flat list

    Returns:
        list: Transformed [nx, ny, nz] normal vector
    """
    # Apply only the rotation part of the matrix to the normal
    x = normal[0] * matrix[0] + normal[1] * matrix[4] + normal[2] * matrix[8]
    y = normal[0] * matrix[1] + normal[1] * matrix[5] + normal[2] * matrix[9]
    z = normal[0] * matrix[2] + normal[1] * matrix[6] + normal[2] * matrix[10]

    # Normalize the result
    length = math.sqrt(x * x + y * y + z * z)
    if length > 0.0001:
        return [x / length, y / length, z / length]
    return [0, 0, 1]  # Default fallback


def add_tangents_bitangents(data: dict) -> None:
    """Append existing data dictionary with tangent and bitanget vertex data."""
    tangents, bitangents = calculate_tangents(
        data['position'],
        data['normal'],
        data['tex_coord'],
        data['indices']
    )

    data.update({
        'tangent': tangents,
        'bitangent': bitangents
    })


def get_vertex_list(
        data: dict,
        program: ShaderProgram,
        batch: Batch,
        group: Group | None = None,
        mode=GL_TRIANGLES,
) -> IndexedVertexList:
    """
    Convenience function, to get vertex list from a dictionary of vertex attributes (such as output of model loaders)
    Data keys should match shader attribute names. Assumes all vertex data (except indices) are of 'float' type.
    """
    count, indices = None, None
    kwargs = {}
    for attr in data:
        # identify position attribute
        if 'pos' in attr.lower():
            count = len(data[attr]) // 3
        # identify indices attribute
        if 'ind' in attr.lower() and isinstance(data[attr][0], int):
            indices = data[attr]
            continue

        # prepare vertex attribute as key-word argument
        kwargs[attr] = ('f', data[attr])

    if count is None:
        raise KeyError("Cannot identify vertex 'position' attribute. If present, use key name that contains 'pos'.")

    if indices is None:
        raise KeyError("Cannot identify 'indices' attribute. If present, use key name that contains 'ind'.")

    return program.vertex_list_indexed(
        count,
        mode,
        indices,
        batch,
        group,
        **kwargs
    )


def get_cached_obj_data(
        path: str,
        save_dir="res/model/cached/",
        scale=1.0, position=(0, 0, 0), rotation=(0, 0, 0), tex_scale=1.0,
        force_reload=False,
        old_version_cleanup=True
) -> dict:
    """
    Fast load of a 3D model's transformed data from a cache file if available,
    otherwise process, cache, and return the transformed model data.
    Cached filename includes original model and a hash value based on passed parameters.
    Older versions of the same model are removed by default.
    """

    def save_model_data(save_path: Path, data: dict):
        with save_path.open('wb') as f:
            pickle.dump(data, f)

    def load_model_data(load_path: Path):
        with load_path.open('rb') as f:
            return pickle.load(f)

    def cleanup(model_base_name: str, keep_filename: str, save_dir="res/model/cached/"):
        """Remove all pickled files for a given model, except the one in use."""
        cache_path = Path(save_dir)
        for file in cache_path.glob(f"{model_base_name}_*.pkl"):
            if file.name != keep_filename:
                file.unlink()
                logger.info(f" Removed old cache file: {file}")

    # get filename
    filename = Path(path).name
    if filename.lower().endswith('.obj'):
        name = filename.rsplit('.', 1)[0]
    else:
        raise NameError("Expected '.obj' file format.")

    # generate has based on transformation parameters
    param_str = f"{scale}_{position}_{rotation}_{tex_scale}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]  # short hash
    hashed_name = f"{name}_{param_hash}.pkl"

    # generating cached file path
    save_dir_path = Path(save_dir)
    cached_file_path = save_dir_path / hashed_name

    # load and return cached file if exists, otherwise create cached file
    if force_reload or not cached_file_path.exists():
        save_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f" Generating and caching model to: {cached_file_path}")
        data = transform_model_data(
            load_obj_model(path),
            scale=scale, position=position, rotation=rotation, tex_scale=tex_scale
        )
        save_model_data(cached_file_path, data)

        # Clean cached old versions of this model
        if old_version_cleanup:
            cleanup(name, hashed_name, save_dir)
        return data

    logger.info(f" Loading cached model: {cached_file_path}")
    return load_model_data(cached_file_path)
