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