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

uniform WindowBlock {
    mat4 projection;
    mat4 view;
    vec3 view_position;
    float time;
    float z_far;
    float fade_length;
} window;

layout(std140) uniform LightBlock {
    vec3 position;
    vec3 target;
    vec4 color;
    bool directional;
    mat4 view;
    mat4 projection;
    float cutoff_start;
    float cutoff_end;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
} light;

layout(std140) uniform MaterialBlock {
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 emission;
    float shininess;
    float reflection_strength;
    float refraction_strength;
    float refractive_index;
    vec4 f0_reflectance;
    float fresnel_power;
    float bump_strength;
} material;

// global rendering flags
uniform bool shadow_mapping = true;
uniform bool soft_shadows = true;
uniform bool lighting = true;
uniform bool lighting_diffuse = true;
uniform bool lighting_specular = true;
uniform bool texturing = true;
uniform bool normal_mapping = true;
uniform bool environment_mapping = true;
uniform bool fade = true;
uniform bool fresnel = true;


float get_fade_factor(){
    float fade_start = window.z_far - window.fade_length;
    float fade_end = window.z_far;
    float distance = length(frag_position - window.view_position);
    // gradual fade with cosine
    float t = clamp((distance - fade_start) / (fade_end - fade_start), 0.0, 1.0);
    return 0.5 + 0.5 * cos(t * 3.14159);
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
    // bump amount
    normal = mix(vec3(0.0, 0.0, 1.0), normal, material.bump_strength);
    // transform normals from tangetn space
    normal = normalize(TBN * normal);
    return normal;
}


void update_phong_factors(
    vec3 normal,
    out vec4 ambient,
    out vec4 diffuse,
    out vec4 specular
){
    // light direction from current fragment
    vec3 light_dir = light.directional ?
        normalize(light.position - light.target) : normalize(light.position - frag_position);

    // Ambient lighting
    ambient = material.ambient * light.ambient;

    // check if fragment is outside of a spotlight's cutoff angle
    float cos_theta = dot(light_dir, -normalize(light.target - light.position));
    if (cos_theta < light.cutoff_end) {  // same as theta > cutoff
        diffuse = vec4(0.0);
        specular = vec4(0.0);

    } else {
        // calcualte intensity adjustment for spotlight cutoff effect (disable for directional?)
        float epsilon = light.cutoff_start - light.cutoff_end;
        float intensity = clamp((cos_theta - light.cutoff_end) / epsilon, 0.0, 1.0);

        // Diffuse lighting
        float diff = lighting_diffuse ? max(dot(light_dir, normal), 0.0) : 0.0;
        // material diffuse color is added later, so it's present even if lighting is not used
        diffuse = intensity * light.diffuse * diff;

        // Specular lighting
        float spec;
        if (lighting_specular) {
            vec3 view_dir = normalize(window.view_position - frag_position);  // direction to viewer
            vec3 reflect_dir = -reflect(light_dir, normal);  // reflection around normal
            spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);  // Specular factor
        } else {
            spec = 0.0;
        }
        specular = intensity * material.specular * light.specular * spec;
    }
}


void update_environment_factors(vec3 normal, out vec3 reflection, out vec3 refraction){
    vec3 view_dir = normalize(frag_position - window.view_position);
    vec3 sample_vector;
    if (material.reflection_strength != 0) {
        // calculate reflection
        sample_vector = -reflect(view_dir, normal);
        reflection = texture(skybox, sample_vector).rgb * material.reflection_strength;
    } else {
        reflection = vec3(0.0);
    }
    if (material.refraction_strength != 0){
        // calculate refraction
        float ratio = 1.00 / material.refractive_index;
        sample_vector = -refract(view_dir, normal, ratio);
        refraction = texture(skybox, sample_vector).rgb * material.refraction_strength;
    } else {
        refraction = vec3(0.0);
    }
}


vec3 get_fresnel_factor(vec3 normal){
    // vector from fragment to camera position
    vec3 view_dir = normalize(window.view_position - frag_position);
    float cos_theta = clamp(dot(view_dir, normal), 0.0, 1.0);
    vec3 F0 = material.f0_reflectance.rgb;  // base reflectance 0.04 for non-metals
    return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5);
}


vec3 get_lighting(float shadow_factor, vec3 normal, vec3 texture_diff){
    vec4 ambient, diffuse, specular;
    if (lighting){
        update_phong_factors(normal, ambient, diffuse, specular);
    } else {
        ambient = vec4(0.0);
        diffuse = vec4(1.0);
        specular = vec4(0.0);
    }
    // separate specular from effect of base color as it's only affected by specular setting and light color
    vec3 reflection, refraction;
    if (environment_mapping) {
        update_environment_factors(normal, reflection, refraction);
    } else {
        // No environment data, use reflection of 1.0 for white halo if fresnel effect is used
        reflection = vec3(1.0);  // how about F0 reflectance here?
        refraction = vec3(0.0);
    }

    vec3 fresnel_factor, env_term;
    if (fresnel) {
        fresnel_factor = get_fresnel_factor(normal);
        env_term = reflection * fresnel_factor + refraction;
    } else {
        fresnel_factor = vec3(0.0);
        env_term = vec3(0.0);
    }

    // base_color adjusted by fresnel difference to avoid burnout with high F0
    vec3 base_color = material.diffuse.rgb * frag_color.rgb * light.color.rgb * texture_diff * (1.0 - fresnel_factor); // * reflection ; // for interesting efffects
    return (ambient + diffuse * shadow_factor).rgb * base_color + env_term + specular.rgb * shadow_factor * light.color.rgb;
}

// Confusing relationship between f0 reflectance and reflection_strength, which controls reflection?
// currently all is fresnel, it's off entire object no reflection

vec3 get_diffuse_texture(){
    return texture(diffuse_texture, frag_tex_coord).rgb;
}


void main() {
    // shadow mapping
    float shadow_factor = shadow_mapping ? get_shadow_factor() : 1.0;

    // Normal mapping
    vec3 normal = normal_mapping ? get_TBN_transformed_normals() : normalize(frag_normal);

    // Texturing
    vec3 texture_diff = texturing ? get_diffuse_texture() : vec3(1.0);

    // Phong lighting
    vec3 lighting = get_lighting(shadow_factor, normal, texture_diff);

    // Fade effect
    float fade_factor = fade ? get_fade_factor() : 1.0;

    // Combine lighting and shadow factors
    final_color = vec4(lighting, frag_color.a * material.diffuse.a * fade_factor);
}