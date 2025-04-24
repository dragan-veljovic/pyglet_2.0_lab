#version 330 core
in vec3 position;
in vec4 colors;

uniform mat4 light_proj;
uniform mat4 light_view;

out vec4 frag_colors;

void main(){
    gl_Position = light_proj * light_view * vec4(position, 1.0);
    frag_colors = colors;
}