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
    //
    float rounded_x = round(new_position.x);
    if (mod(rounded_x, 2.0) == 0){
        // for even x coordinates
        new_position.yz += new_position.yx * sin(time) * 0.1;
    } else {
        // for odd x coordinates
        new_position.zy -= new_position.zx * sin(time) * 0.1;
    }

    gl_Position = window.projection * window.view * vec4(new_position, 1.0);
    vertex_colors = colors;
}
