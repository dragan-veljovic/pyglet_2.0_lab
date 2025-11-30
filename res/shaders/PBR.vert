#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec2 tex_coord;
layout (location = 3) in vec3 normal;

// outputs
out vec3 frag_position;
out vec4 frag_color;
out vec2 frag_tex_coord;
out vec3 frag_normal;
out float frag_disp;

// main app
uniform WindowBlock{
    mat4 projection;
    mat4 view;
    vec3 view_position;
    float time;
} window;

// wave declarations
uniform vec3 wave0;
uniform vec3 wave1;
uniform vec3 wave2;
uniform vec3 wave3;
uniform int no_waves;
uniform int wave_id;
//uniform float lambda = 300.0;

float speed = 150.0;
const float pi = 3.1416;

// transformations
uniform mat4 model_precalc;
uniform bool rendering_dynamic_object = false;
uniform bool wavy_transformation = true;

// wave data
vec3 wave_positions[] = vec3[](wave0, wave1, wave2, wave3);
uniform vec4 wave_amplitudes = vec4(20, 25,30, 10);
uniform vec4 wave_frequencies = vec4(0.5, 0.5, 0.75, 1.0);
uniform vec4 wave_phases = vec4(0, pi, 0, 0);
uniform vec4 source_activity_map = vec4(0.0);


float wave_displacement(vec3 pos) {
    float y = 0.0;

    for (int i = 0; i < no_waves; i++){
        if (source_activity_map[i] != 0.0) {
            float A = wave_amplitudes[i];
            float f = wave_frequencies[i];
            float p = wave_phases[i];
            float lambda = speed/f;
            float k = 2.0 * pi / lambda;
            float w = 2.0 * pi * f;

            float r = length(pos - wave_positions[i]);
            y += A * sin(k * r - w * window.time + p);
        }
    }
    return y;
}

vec3 wave_normal(vec3 pos)
{
    float dy_dx = 0.0;
    float dy_dz = 0.0;

    for (int i = 0; i < no_waves; i++)
    {
        if (source_activity_map[i] != 0.0){
            float A = wave_amplitudes[i];
            float f = wave_frequencies[i];
            float lambda = speed/f;
            float k = 2.0 * pi / lambda;
            float w = 2.0 * pi * f;
            float p = wave_phases[i];

            vec3 d = pos - wave_positions[i];// displacement vector
            float x = d.x;
            float z = d.z;

            float r = length(vec2(x, z));
            float phase = k * r - w * window.time + p;

            // avoid NaN at center
            float inv_r = (r > 0.0001) ? 1.0 / r : 0.0;

            // partial derivatives of this individual wave
            float c = A * cos(phase) * k * inv_r;
            dy_dx += c * x;
            dy_dz += c * z;
        }
    }
    return normalize(vec3(-dy_dx, 1.0, -dy_dz));
}

float source_displacement(int index){
    float A = wave_amplitudes[index];
    float f = wave_frequencies[index];
    float lambda = speed / f;
    float p = wave_phases[index];
    float w = 2.0 * pi * f;

    float k = 2.0 * pi / lambda;
    float r = length(wave_positions[wave_id]);

    float disp = A * sin(k * r - w * window.time + p);
    return disp;
}

void main() {
    mat4 model = rendering_dynamic_object ? model_precalc : mat4(1.0);

    float displacement = wave_displacement(position);
    vec3 new_position;

    if (wavy_transformation){
        new_position = position + vec3(0, displacement, 0);
        frag_normal = wave_normal(position);
        frag_disp = displacement;

    } else {
        if (source_activity_map[wave_id] != 0.0) {
            float disp = source_displacement(wave_id);
            new_position = position + vec3(0, disp, 0);
            mat3 normal_adjustment = mat3(transpose(inverse(model)));
            frag_normal = normalize(normal_adjustment * normal);
            frag_disp = disp;
        }
    }


    vec4 transformed = window.projection * window.view * model * vec4(new_position, 1.0);
    gl_Position = transformed;
    frag_position = (model * vec4(position, 1.0)).xyz;
    frag_color = color;
    frag_tex_coord = tex_coord;
}
