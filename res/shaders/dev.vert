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

// perhaps pass these as a UBO to reduce number of update calls
uniform bool rendering_dynamic_object = true;
uniform bool transform_on_gpu = true;
uniform mat4 model;
uniform vec3 model_position = vec3(0.0);
uniform vec3 model_rotation = vec3(0.0);
uniform vec3 model_scale = vec3(1.0);
uniform vec3 model_origin = vec3(0.0);

mat4 get_model_matrix(){
    mat4 translate_to_origin = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -model_origin.x, -model_origin.y, -model_origin.z, 1.0
    );

    // Move object back after rotation
    mat4 translate_back = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        model_origin.x, model_origin.y, model_origin.z, 1.0
    );

    // x-rotation
    float cx = cos(model_rotation.x);
    float sx = sin(model_rotation.x);
    mat4 rotx = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, cx, -sx, 0.0,
        0.0, sx, cx, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    // y-rotation
    float cy = cos(model_rotation.y);
    float sy = sin(model_rotation.y);
    mat4 roty = mat4(
        cy, 0.0, sy, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -sy, 0.0, cy, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    // z-rotation
    float cz = cos(model_rotation.z);
    float sz = sin(model_rotation.z);
    mat4 rotz = mat4(
        cz, -sz, 0.0, 0.0,
        sz, cz, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    // scale matrix
    mat4 scale = mat4(
        model_scale.x, 0.0, 0.0, 0.0,
        0.0, model_scale.y, 0.0, 0.0,
        0.0, 0.0, model_scale.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    // translation matrix
    mat4 translation = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        model_position.x, model_position.y, model_position.z, 1.0
    );

    // Combine all transformations
    // Order matters: scale first, then rotate, then translate
    // return translation * rotz * roty * rotx * scale;
    return translation * translate_back * rotx * roty * rotz * translate_to_origin * scale;
}

void main() {
    // whole adjustment should only be done if rendering dynamic object
    // a large static mesh will still be multiplied by default mat3(1.0) wasting resources
    mat4 model = rendering_dynamic_object ?
        (transform_on_gpu ? get_model_matrix() : model)
        : mat4(1.0);
    mat3 normal_adjustment = rendering_dynamic_object ? mat3(transpose(inverse(model))) : mat3(1.0);

    // calculating TBN (should be done only if normal mapping is used)
    vec3 T = normalize(normal_adjustment * tangents);
    vec3 B = normalize(normal_adjustment * bitangents);
    vec3 N = normalize(normal_adjustment * normals);

    // out variables
    TBN = mat3(T, B, N);
    frag_tex_coords = tex_coords;
    frag_position = (model * vec4(position, 1.0)).xyz;
    frag_colors = colors;
    frag_normals = N;

    // Pass light-space coordinates to fragment shader
    // shadows seems not to work for moving object
    // perhaps because of different shader and transformation should be passed there
    // for this look into
    //"Shader Switching: When switching between shader programs,
     //uniform blocks can persist if they're bound to the same binding point"
    frag_shadow_coords = light_proj * light_view * model * vec4(position, 1.0);

    gl_Position = window.projection * window.view * model * vec4(position, 1.0);
}