#version 330 core
in vec4 frag_colors;

out vec4 frag_depth;

void main() {
    // set explicitly or not, this is what effectively happens behind the scene
    frag_depth = vec4(1 - gl_FragCoord.z);
}