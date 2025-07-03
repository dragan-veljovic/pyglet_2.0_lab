#version 330 core
// input variables
in vec3 frag_position;
in vec4 frag_color;
in vec2 frag_tex_coord;
in vec3 frag_normal;
in vec4 frag_shadow_coord;
in mat3 TBN;

// output variables
out vec4 final_color;

// texture uniforms
uniform sampler2D diffuse_texture;  // texture slot 0
uniform sampler2D normal_map;  // texture slot 1
uniform sampler2D shadow_map;  // texture slot 4
uniform samplerCube skybox;  // texture slot 5

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

// camera properties
uniform vec3 view_position;
uniform float z_far = 5000.0;
uniform float fade_length = 1000.0;

// light properties
//uniform vec3 light_position;
//uniform vec3 light_target = vec3(0.0);
//uniform bool light_directional = false;
//uniform vec3 light_color = vec3(1.0);

// material properties
uniform float ambient_strength = 0.25;
uniform float diffuse_strength = 0.50;
uniform float specular_strength = 1.0;
uniform float shininess = 128;

// rendering flags
uniform bool shadow_mapping = true;
uniform bool soft_shadows = true;
uniform bool lighting = true;
uniform bool lighting_diffuse = true;
uniform bool lighting_specular = true;
uniform bool texturing = true;
uniform bool normal_mapping = true;
uniform bool environment_mapping = true;
uniform bool fade = true;



float get_fade_factor(){
    float fade_start = z_far - fade_length;
    float fade_end = z_far;
    float distance = length(frag_position - view_position);
    return clamp(1.0 - (distance - fade_start) / (fade_end - fade_start), 0.0, 1.0);
}


float get_shadow_factor() {
    // Perform perspective divide
    vec3 proj_coord = frag_shadow_coord.xyz / frag_shadow_coord.w;
    int pcf_samples = 9;
    float shadow_bias = 0.005;
    // Transform to [0, 1] texture space
    proj_coord = proj_coord * 0.5 + 0.5;

    // Apply Percentage Closer Filtering (PCF)
    if (soft_shadows) {
        float shadow = 0.0;
        float sample_offset = 0.001;  // Adjust this for smoothness
        // PCF sampling around the current fragment
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 offset = vec2(x, y) * sample_offset;
                float depth = texture(shadow_map, proj_coord.xy + offset).r;
                shadow += (proj_coord.z > depth + shadow_bias) ? 0.5 : 1.0;  // Bias to reduce artifacts
            }
        }
        // Average the result
        shadow /= float(pcf_samples);
        return shadow;

    // Regular hard shadows
    } else {
         // Read the depth from the shadow map
        float closest_depth = texture(shadow_map, proj_coord.xy).r;
        // Current fragment depth in light space
        float current_depth = proj_coord.z;
        // Check if the fragment is in shadow
        return (current_depth > closest_depth + shadow_bias) ? 0.5 : 1.0; // Bias to reduce artifacts
    }
}

vec3 get_TBN_transformed_normals(){
    // sample normal map to get pixel normal values
    vec3 normal = texture(normal_map, frag_tex_coord).rgb;
    // obtain normal from normal map in range [0, 1]
    normal = normal * 2.0 - 1.0;
    // transform normals from tangetn space
    normal = normalize(TBN * normal);
    return normal;
}

vec3 get_phong_lighting_factors(vec3 normal){
    float ambient;
    float diffuse;
    float specular;

    // light direction from current fragment
    vec3 light_direction = light.directional ?
        normalize(light.position - light.target) : normalize(light.position - frag_position);

    // Ambient lighting
    ambient = ambient_strength;

    // check if fragment is outside of a spotlight's cutoff angle
    float cos_theta = dot(light_direction, -normalize(light.target - light.position));

    if (cos_theta < light.cutoff_end) {  // same as theta > cutoff
        diffuse = 0.0;
        specular = 0.0;

    } else {
        // calcualte intensity adjustment for spotlight cutoff effect
        float epsilon = light.cutoff_start - light.cutoff_end;
        float intensity = clamp((cos_theta - light.cutoff_end) / epsilon, 0.0, 1.0);

        // Diffuse lighting
        float diff = lighting_diffuse ? max(dot(light_direction, normal), 0.0) : 0.0;
        diffuse = intensity * diffuse_strength * diff;

        // Specular lighting
        float spec;
        if (lighting_specular) {
            vec3 view_dir = normalize(view_position - frag_position);  // direction to viewer
            vec3 reflect_dir = reflect(-light_direction, normal);  // reflection around normal
            spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);  // Specular factor
        } else {
            spec = 0.0;
        }
        specular = intensity * specular_strength * spec;
    }

    // Summarize and return result
    return vec3(ambient, diffuse, specular);
}

vec3 get_environment_mapping(vec3 normal){
    // refraction
    float ratio = 1.00 / 1.52;
    vec3 view_dir = normalize(frag_position - view_position);
    vec3 sample_vector = -refract(view_dir, normal, ratio);
    vec3 refraction = texture(skybox, sample_vector).rgb;
    // reflection
    sample_vector = -reflect(view_dir, normal);
    vec3 reflection = texture(skybox, sample_vector).rgb;
    return reflection;
}

vec3 get_lighting(float shadow_factor, vec3 normal, vec3 texture_diff){
    // Custom function that applies all lighting factors
    vec3 phong = get_phong_lighting_factors(normal);
    float ambient = phong.x;
    float diffuse = phong.y;
    float specular = phong.z;
    // specular not affected by object colors (only light color) is more realistic
    return ((ambient + diffuse * shadow_factor) * texture_diff + specular * shadow_factor) * light.color;
}


vec3 get_diffuse_texture(){
    return texture(diffuse_texture, frag_tex_coord).rgb;
}


void main() {
    // shadow mapping
    float shadow_factor = shadow_mapping ? get_shadow_factor() : 1.0;
    // Normal mapping
    vec3 normal = normal_mapping ? get_TBN_transformed_normals() : normalize(frag_normal);
    // Texturing
    vec3 texture_diff = texturing ? get_diffuse_texture() : vec3(0);

    // Environment mapping
    vec3 env_map = environment_mapping ? get_environment_mapping(normal) : vec3(0.0);

    // Phong lighting
    vec3 lighting = lighting ? get_lighting(shadow_factor, normal, texture_diff) : vec3(1.0) * shadow_factor;
    // Fade effect
    float fade_factor = fade ? get_fade_factor() : 1.0;

    // Combine lighting and shadow factors
    final_color = vec4(lighting  + env_map, frag_color.a * fade_factor);
}