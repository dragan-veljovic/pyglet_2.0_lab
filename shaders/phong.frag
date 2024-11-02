#version 330 core
    in vec3 frag_position;
    in vec3 frag_normals;
    in vec4 object_colors;
    in vec2 texture_coords;

    out vec4 final_colors;

    uniform sampler2D our_texture;

    // light uniforms
    uniform vec3 light_position;
    uniform vec3 light_color;
    uniform float ambient_strength;
    uniform vec3 view_position;
    uniform float specular_strength;
    uniform float shininess;

    // texturing or shading
    uniform bool texturing = true;

    void main()
    {
        // Ambient lighting
        vec3 ambient = ambient_strength * light_color;  // scale light by ambient strength

        // Diffuse lighting
        vec3 norm = normalize(frag_normals);
        vec3 light_dir = normalize(light_position - frag_position);
        float diff = max(dot(light_dir, norm), 0.0);  // diffusion factor is positive cos(theta)
        vec3 diffuse = diff * light_color;  // Scale light by diffusion factor

        // Specular lighting
        vec3 view_dir = normalize(view_position - frag_position);  // direction to viewer
        vec3 reflect_dir = reflect(-light_dir, norm);  // reflection around normal
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);  // Specular factor
        vec3 specular = specular_strength * spec * light_color;  // Scale light by specular factor

        // Inverse square-distance attenuation
        float distance = length(light_position - frag_position);
        float attenuation = min(1000 / (distance), 1.0);
        diffuse *= attenuation;
        specular *= attenuation;

        // Combine results
        vec3 result = (ambient + diffuse + specular) * object_colors.rgb;

        // texturing or monochrome shading
        if (texturing){
            vec4 tex_color = texture(our_texture, texture_coords);
            final_colors = tex_color * vec4(result, 1.0);
        } else {
            final_colors = vec4(result, 1.0);
        }
    }