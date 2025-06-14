#version 330 core
// vertex atrribute
layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec2 tex_coord;
layout (location = 3) in vec3 normal;
layout (location = 4) in vec3 tangent;
layout (location = 5) in vec3 bitangent;
// instance attribute
layout (location = 6) in vec4 instance_data_0;
layout (location = 7) in vec4 instance_data_1;
layout (location = 8) in vec4 instance_data_2;
layout (location = 9) in vec4 instance_data_3;
// outputs
out vec3 frag_position;
out vec4 frag_color;
out vec2 frag_tex_coord;
out vec3 frag_normal;
out vec4 frag_shadow_coord;
out mat3 TBN;

uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;

layout(std140) uniform LightBlock {
    vec3 position;
    vec3 target;
    vec3 color;
    bool directional;
    mat4 view;
    mat4 projection;
    float cutoff_start;
    float cutoff_end;
} light;

uniform float time;

// instance rendering
uniform bool instance_rendering = false;
// dynamic rendering
uniform bool rendering_dynamic_object = false;
uniform bool transform_on_gpu = false;
// CPU matrix or model parameters
uniform mat4 model_precalc;
uniform vec3 model_position = vec3(0.0);
uniform vec3 model_rotation = vec3(0.0);
uniform vec3 model_scale = vec3(1.0);
uniform vec3 model_origin = vec3(0.0);



mat4 get_model_matrix(
        vec3 model_position,
        vec3 model_rotation,
        vec3 model_scale,
        vec3 model_origin
) {
    // translation matrix
    mat4 translation = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        model_position.x, model_position.y, model_position.z, 1.0
    );

    // scale matrix
    mat4 scale = mat4(
        model_scale.x, 0.0, 0.0, 0.0,
        0.0, model_scale.x, 0.0, 0.0,
        0.0, 0.0, model_scale.x, 0.0,
        0.0, 0.0, 0.0, 1.0
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

    // Combine all transformations, allowing for possible custom origin
    // Order matters: scale first, then rotate, then translate
    if (model_origin != vec3(0.0))
    {
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

        return translation * translate_back * rotx * roty * rotz  * scale * translate_to_origin;
    }
    else
    {
        return translation * rotx * roty * rotz * scale;
    }
}

mat4 get_instance_model_matrix_gpu(){
    // Example how to extract data from instance attribute and get matrix on the GPU
    return get_model_matrix(
            vec3(instance_data_0.xyz), // extract transltation data
            vec3(instance_data_1.xyz), // extract rotation data
            vec3(instance_data_2.xyz), // extract scale data
            vec3(0.0)  // default origin
    );
}

mat4 get_instance_model_matrix(){
    // math is cheap, but reading this many attributes per vertex can be memory-bound
    return mat4(instance_data_0, instance_data_1, instance_data_2, instance_data_3);
}

void main() {
    // whole adjustment should only be done if rendering dynamic object
    // a large static mesh will still be multiplied by default mat3(1.0) wasting resources
    mat4 model;
    if (instance_rendering) {
        model = get_instance_model_matrix_gpu();
    } else {
        if (rendering_dynamic_object) {
            if (transform_on_gpu) {
                model = get_model_matrix(model_position, model_rotation, model_scale, model_origin);
            } else {
                model = model_precalc;
            }
        } else {
            model = mat4(1.0);
        }
    }

    // only if dynamic/ instance
    mat3 normal_adjustment = mat3(transpose(inverse(model)));
    // mat3 normal_adjustment = rendering_dynamic_object ? mat3(transpose(inverse(model))) : mat3(1.0);

    // calculating TBN (should be done only if normal mapping is used)
    vec3 T = normalize(normal_adjustment * tangent);
    vec3 B = normalize(normal_adjustment * bitangent);
    vec3 N = normalize(normal_adjustment * normal);

     // output attributes
    TBN = mat3(T, B, N);
    frag_tex_coord = tex_coord;
    frag_position = (model * vec4(position, 1.0)).xyz;
    frag_color = color;
    frag_normal = N;

    // Pass light-space coordinates to fragment shader
    // shadows seems not to work for moving object
    // perhaps because of different shader and transformation should be passed there
    // for this look into
    //"Shader Switching: When switching between shader programs,
     //uniform blocks can persist if they're bound to the same binding point"
    frag_shadow_coord = light.projection * light.view * model * vec4(position, 1.0);

    gl_Position = window.projection * window.view * model * vec4(position, 1.0);
}