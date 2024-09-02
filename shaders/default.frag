#version 330 core
in vec4 vertex_colors;
out vec4 final_colors;

uniform float time;

void main() {
    final_colors = vertex_colors;
}
