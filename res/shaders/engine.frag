#version 330 core
in vec3 frag_position;
in vec2 frag_tex_coords;
in vec3 frag_normals;
in vec4 frag_colors;
in vec4 frag_shadow_coords;
in mat3 TBN;

out vec4 final_colors;

// texture uniforms
uniform sampler2D shadow_map;
uniform sampler2D diffuse_texture;
uniform sampler2D normal_map;

// vector uniforms
uniform vec3 light_position;
uniform vec3 view_position;

// flags
uniform bool shadowing = true;
uniform bool soft_shadows = true;
uniform bool lighting = true;
uniform bool lighting_diffuse = true;
uniform bool lighting_specular = true;
uniform bool texturing = true;
uniform bool normal_mapping = true;

// settings
float shadow_bias = 0.005;

// Number of samples for PCF
uniform int pcf_samples = 9;  // Adjust this for quality (more samples = smoother shadows)

float get_shadow_factor(vec4 shadow_coords) {
    // Perform perspective divide
    vec3 proj_coords = shadow_coords.xyz / shadow_coords.w;

    // Transform to [0, 1] texture space
    proj_coords = proj_coords * 0.5 + 0.5;

    if (soft_shadows) {
        // Apply Percentage Closer Filtering (PCF)
        float shadow = 0.0;
        float sample_offset = 0.001;  // Adjust this for smoothness

        // PCF sampling around the current fragment
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 offset = vec2(x, y) * sample_offset;
                float depth = texture(shadow_map, proj_coords.xy + offset).r;
                shadow += (proj_coords.z > depth + 0.005) ? 0.5 : 1.0;  // Bias to reduce artifacts
            }
        }

        // Average the result
        shadow /= float(pcf_samples);
        return shadow;
    } else {
         // Read the depth from the shadow map
        float closest_depth = texture(shadow_map, proj_coords.xy).r;

        // Current fragment depth in light space
        float current_depth = proj_coords.z;

        // Check if the fragment is in shadow
        return (current_depth > closest_depth + 0.005) ? 0.5 : 1.0; // Bias to reduce artifacts
    }
}

void main() {
    float shadow_factor;
    // Get pixel shadow information using PCF
    if (shadowing) {
        shadow_factor = get_shadow_factor(frag_shadow_coords);
    } else {
        shadow_factor = 1.0;
    }

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    if (lighting){
        vec3 normal;
        if (normal_mapping) {
            // obtain normal from normal map in range [0, 1]
            normal = texture(normal_map, frag_tex_coords).rgb;
            normal = normal * 2.0 - 1.0;
            normal = normalize(TBN * normal);
        } else {
            normal = normalize(frag_normals);
        }

        // normal = normalize(frag_normals);

        vec3 light_color = vec3(1.0);
        vec3 light_direction = normalize(light_position - frag_position);

        // Ambient lighting
        ambient = 0.3 * light_color;

        // Diffuse lighting
        float diff = lighting_diffuse ? max(dot(light_direction, normal), 0.0) : 0.0;
        diffuse = diff * light_color;

        // Specular lighting
        float spec;
        if (lighting_specular){
            vec3 view_dir = normalize(view_position - frag_position);  // direction to viewer
            vec3 reflect_dir = reflect(-light_direction, normal);  // reflection around normal
            spec = pow(max(dot(view_dir, reflect_dir), 0.0), 64);  // Specular factor
        } else {
            spec = 0.0;
        }
        specular = spec * light_color;
    } else {
        ambient = vec3(1.0);
        diffuse = vec3(0.0);
        specular = vec3(0.0);
    }

    vec3 texture_diff;
    if (texturing){
        // Apply textures
        texture_diff = texture(diffuse_texture, frag_tex_coords).rgb;
    } else {
        texture_diff = frag_colors.rgb;
    }

    // Combine lighting and shadow factors
    vec3 lighting = vec3(ambient + (diffuse + specular) * shadow_factor);
    final_colors = vec4(texture_diff * lighting, 1.0);
}