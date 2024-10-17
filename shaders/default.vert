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

mat4 get_rotation_matrix(float theta){
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

void main() {
    mat4 rotation = get_rotation_matrix(time/2);
    vec3 new_position = position;
    gl_Position = window.projection * window.view * rotation * vec4(new_position, 1.0);
    vertex_colors = colors;
}
