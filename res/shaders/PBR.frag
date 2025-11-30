#version 330 core
in vec3 frag_position;
in vec4 frag_color;
in vec2 frag_tex_coord;
in vec3 frag_normal;
in float frag_disp;

out vec4 final_color;

uniform WindowBlock{
    mat4 projection;
    mat4 view;
    vec3 view_position;
    float time;
} window;

uniform int no_lights = 2;

float gamma = 2.2;
vec3 light_color = vec3(1.0);
float light_intensity = 1.0/no_lights;
vec3 light_positions[] = vec3[](
    vec3(-5000, 2000, 5000),
    vec3(0, 500, 1000),
    vec3(-1000, 500, 1000),
    vec3(-500, 1000, 1000)
);

vec3 phong_lighting(){
    // ambient lighting
    float ambient = 0.15;
    float diffuse = 0.0;
    float specular = 0.0;
    float spec_strength = 2;
    vec3 lighting = vec3(ambient);
    for (int i=0; i<no_lights; i++){
        // diffuse lighting
        vec3 light_direction = normalize(light_positions[i] - frag_position);
        float diff = max(dot(frag_normal, light_direction), 0.0);
        // specular lighting
        vec3 view_direction = normalize(frag_position - window.view_position);
        vec3 reflected = normalize(reflect(light_direction, frag_normal));
        float spec = pow(max(dot(view_direction, reflected), 0.0), 128) * spec_strength;
        diffuse += diff;
        specular += spec;
    }

    vec4 new_frag_color = vec4(frag_color.r + frag_disp/50, frag_color.gba);

    lighting = ((ambient + diffuse) * new_frag_color.rgb + specular * light_color) * light_intensity;
    return lighting;
}



void main() {
    vec3 lighting = phong_lighting();
    lighting = clamp(pow(lighting, vec3(1.0 / gamma)), 0.0, 1.0);
    final_color = vec4(lighting, frag_color.a);
}
