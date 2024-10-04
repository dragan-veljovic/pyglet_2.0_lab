#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_colors;
out vec3 vertex_positions;

uniform float time;

uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

void main() {
    vec3 new_position = position;
    new_position.y += 300*sin(new_position.x/50 - time);
    gl_Position = window.projection * window.view * vec4(new_position, 1.0);
    vertex_colors = colors;
    vertex_positions = normalize(new_position);
}
