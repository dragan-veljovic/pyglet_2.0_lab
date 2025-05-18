#version 330 core
// input variables
in vec3 frag_position;
in vec2 frag_tex_coords;
in vec3 frag_normals;
in vec4 frag_colors;
in vec4 frag_shadow_coords;
in mat3 TBN;

// output variables
out vec4 final_colors;

// texture uniforms
uniform sampler2D diffuse_texture;
uniform sampler2D normal_map;
uniform sampler2D shadow_map;

// lighting and material uniforms
uniform vec3 light_position;
uniform vec3 view_position;
uniform bool directional_light = false;
uniform vec3 light_target = vec3(0.0);
uniform vec3 light_color = vec3(1.0);
uniform float ambient_strength = 0.25;
uniform float diffuse_strength = 1.0;
uniform float specular_strength = 1.0;
uniform float shininess = 128.0;

// rendering flags
uniform bool shadow_mapping = true;
uniform bool soft_shadows = true;
uniform bool lighting = true;
uniform bool lighting_diffuse = true;
uniform bool lighting_specular = true;
uniform bool texturing = true;
uniform bool normal_mapping = true;
uniform bool fade = true;

// other uniforms
uniform float z_far = 5000.0;


float get_fade_factor(){
    float fade_length = 750;
    float fade_start = z_far - fade_length;
    float fade_end = z_far;
    float distance = length(frag_position - view_position);
    return clamp(1.0 - (distance - fade_start) / (fade_end - fade_start), 0.0, 1.0);
}


float get_shadow_factor() {
    // Perform perspective divide
    vec3 proj_coords = frag_shadow_coords.xyz / frag_shadow_coords.w;
    int pcf_samples = 9;
    float shadow_bias = 0.005;
    // Transform to [0, 1] texture space
    proj_coords = proj_coords * 0.5 + 0.5;

    // Apply Percentage Closer Filtering (PCF)
    if (soft_shadows) {
        float shadow = 0.0;
        float sample_offset = 0.001;  // Adjust this for smoothness
        // PCF sampling around the current fragment
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 offset = vec2(x, y) * sample_offset;
                float depth = texture(shadow_map, proj_coords.xy + offset).r;
                shadow += (proj_coords.z > depth + shadow_bias) ? 0.5 : 1.0;  // Bias to reduce artifacts
            }
        }
        // Average the result
        shadow /= float(pcf_samples);
        return shadow;

    // Regular hard shadows
    } else {
         // Read the depth from the shadow map
        float closest_depth = texture(shadow_map, proj_coords.xy).r;
        // Current fragment depth in light space
        float current_depth = proj_coords.z;
        // Check if the fragment is in shadow
        return (current_depth > closest_depth + shadow_bias) ? 0.5 : 1.0; // Bias to reduce artifacts
    }
}

vec3 get_TBN_transformed_normals(){
    // sample normal map to get pixel normal values
    vec3 normal = texture(normal_map, frag_tex_coords).rgb;
    // obtain normal from normal map in range [0, 1]
    normal = normal * 2.0 - 1.0;
    // transform normals from tangetn space
    normal = normalize(TBN * normal);
    return normal;
}

vec3 get_phong_lighting(float shadow_factor, vec3 normal){
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    //
    vec3 light_direction = directional_light ?
        normalize(light_position - light_target) : normalize(light_position - frag_position);

    // Ambient lighting
    ambient = ambient_strength * light_color;

    // Diffuse lighting
    float diff = lighting_diffuse ? max(dot(light_direction, normal), 0.0) : 0.0;
    diffuse = diffuse_strength * diff * light_color;

    // Specular lighting
    float spec;
    if (lighting_specular){
        vec3 view_dir = normalize(view_position - frag_position);  // direction to viewer
        vec3 reflect_dir = reflect(-light_direction, normal);  // reflection around normal
        spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);  // Specular factor
    } else {
        spec = 0.0;
    }
    specular = specular_strength * spec * light_color;

    // Summarize and return result
    return vec3(ambient + (diffuse + specular) * shadow_factor);
}


vec3 get_diffuse_texture(){
    return texture(diffuse_texture, frag_tex_coords).rgb;
}


void main() {
    // shadow mapping
    float shadow_factor = shadow_mapping ? get_shadow_factor() : 1.0;
    // Normal mapping
    vec3 normal = normal_mapping ? get_TBN_transformed_normals() : normalize(frag_normals);
    // Phong lighting
    vec3 lighting = lighting ? get_phong_lighting(shadow_factor, normal) : vec3(1.0) * shadow_factor;
    // Texturing
    vec3 texture_diff = texturing ? get_diffuse_texture() : frag_colors.rgb;
    // Fade effect
    float fade_factor = fade ? get_fade_factor() : 1.0;

    // Combine lighting and shadow factors
    final_colors = vec4(texture_diff * lighting, frag_colors.a * fade_factor);
}