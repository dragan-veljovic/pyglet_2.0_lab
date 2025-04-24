"""
Updated OBJ model loading, allowing for quad or triangle faces.
Includes normal map calculation, if normals are not present.
Includes tangent and bitanget calculation for normal mapping.
TODO: implement 3 methods as a class, possibly fuse with pyglet model
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


import math


def load_obj_model(filename):
    """
    Load an OBJ model file and prepare vertex data for use with pyglet's VertexList

    Args:
        filename (str): Path to the .obj file

    Returns:
        dict: Dictionary containing flattened vertex data:
            - 'positions': Flattened position coordinates [x0, y0, z0, x1, y1, z1, ...]
            - 'normals': Flattened normal vectors (if available)
            - 'tex_coords': Flattened texture coordinates (if available)
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

    # If no texture coordinates were provided, create default ones
    if not final_tex_coords and final_positions:
        for _ in range(len(final_positions) // 3):
            final_tex_coords.extend([0.0, 0.0])

    return {
        'positions': final_positions,
        'normals': final_normals,
        'tex_coords': final_tex_coords,
        'indices': indices
    }


def calculate_tangents(positions, normals, tex_coords, indices):
    """
    Calculate tangent and bitangent vectors for normal mapping

    Args:
        positions (list): Flattened position data [x0, y0, z0, x1, y1, z1, ...]
        normals (list): Flattened normal data
        tex_coords (list): Flattened texture coordinate data [u0, v0, u1, v1, ...]
        indices (list): Vertex indices

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


def create_mesh_from_obj(filename, batch, group=None, shader_program=None,
                         position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0, alpha=255):
    """
    Create a pyglet VertexList from an OBJ file with pre-computed transformations

    Args:
        filename (str): Path to the .obj file
        batch (pyglet.graphics.Batch): Batch to add the mesh to
        group (pyglet.graphics.Group, optional): Group to add the mesh to
        shader_program (pyglet.graphics.Program, optional): Shader program to use
        position (tuple): (x, y, z) position for the object
        rotation (tuple): (x, y, z) rotation in degrees (Euler angles)
        scale (float or tuple): Uniform scale if float, or (x, y, z) scale factors if tuple

    Returns:
        pyglet.graphics.VertexList: The created vertex list
    """
    # Load the model data
    model_data = load_obj_model(filename)

    # Create transformation matrix
    transform_matrix = create_transformation_matrix(position, rotation, scale)

    # Apply transformations to vertex positions and normals
    transformed_positions = []
    transformed_normals = []
    original_positions = model_data['positions']
    original_normals = model_data['normals']

    # Transform each vertex position
    for i in range(0, len(original_positions), 3):
        vertex = original_positions[i:i + 3]
        transformed = transform_vertex(vertex, transform_matrix)
        transformed_positions.extend(transformed)

    # Transform each normal
    for i in range(0, len(original_normals), 3):
        normal = original_normals[i:i + 3]
        transformed = transform_normal(normal, transform_matrix)
        transformed_normals.extend(transformed)

    # Recalculate tangents and bitangents using the transformed positions and normals
    tangents, bitangents = calculate_tangents(
        transformed_positions,
        transformed_normals,
        model_data['tex_coords'],
        model_data['indices']
    )

    # Create vertex list
    if shader_program:
        # If we have a shader program, use it to create the vertex list
        vertex_list = shader_program.vertex_list_indexed(
            len(transformed_positions) // 3,
            pyglet.gl.GL_TRIANGLES,
            model_data['indices'],
            batch,
            group,
            position=('f', transformed_positions),
            normals=('f', transformed_normals),
            tex_coords=('f', model_data['tex_coords']),
            tangents=('f', tangents),
            bitangents=('f', bitangents)
        )
    else:
        # Otherwise use the default vertex list
        vertex_list = batch.add_indexed(
            len(transformed_positions) // 3,
            pyglet.gl.GL_TRIANGLES,
            group,
            model_data['indices'],
            ('v3f', transformed_positions),
            ('n3f', transformed_normals),
            ('t2f', model_data['tex_coords'])
        )

    return vertex_list


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

        pyglet.resource.path.append('res')
        pyglet.resource.reindex()

        self.program = ShaderProgram(
            pyglet.resource.shader("shaders/engine.vert"),
            pyglet.resource.shader("shaders/engine.frag"),
        )

        self.shadow_program = ShaderProgram(
            pyglet.resource.shader("shaders/shadow.vert"),
            pyglet.resource.shader("shaders/shadow.frag"),
        )

        # light
        self.light = PointLight(Vec3(-500, 600, 700), color=(255, 255, 255, 255))

        glEnable(GL_DEPTH_TEST)

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

        self.barrel_group = DiffuseNormalTextureGroup(
            pyglet.image.load("res/model/Barrel/barrel_BaseColor.png").get_texture(),
            pyglet.image.load("res/model/Barrel/barrel_Normal.png").get_texture(),
            self.program
        )

        rows, columns = 2, 2

        for i in range(rows):
            for j in range(columns):
                # scene domain
                self.plant = create_mesh_from_obj(
                    "res/model/Barrel/Barrel_OBJ.obj", self.batch, shader_program=self.program,
                    scale=300, group=self.barrel_group, position=(i*300, -500, j*300), rotation=(0, -90, 0))

                # shadow domain
                self.shadow_model = self.shadow_program.vertex_list_indexed(
                    count=len(self.plant.position)//3,
                    mode=GL_TRIANGLES,
                    indices=self.plant.indices,
                    batch=self.shadow_batch,
                    position=('f', self.plant.position[:]),
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
