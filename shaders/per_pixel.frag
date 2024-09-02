#version 330 core
in vec4 vertex_colors;
out vec4 final_colors;
out vec4 fragColor;

uniform float time;

void main() {
    vec4 new_colors = vertex_colors;
    vec2 uv = gl_FragCoord.xy / vec2(1280.0, 720);
    if (new_colors.rgba != vec4(0.)){
        final_colors = vec4(uv.x, uv.y, 0.5, 1.0);
    } else {
        final_colors = new_colors;
    }
}
