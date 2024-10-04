#version 330 core
in vec4 vertex_colors;
in vec3 vertex_positions;
out vec4 final_colors;

uniform float time;

void main() {
    final_colors = vec4(vertex_positions, 1) * vec4(1, 1, 1, 1) * vec4(sin(time), -cos(time), -2*sin(time), 1);
}
