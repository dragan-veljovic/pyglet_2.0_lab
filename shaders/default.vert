#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;

uniform float time;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    vec3 new_position = position;
    gl_Position = window.projection * window.view * vec4(new_position, 1.0);
    vertex_colors = colors;
}
