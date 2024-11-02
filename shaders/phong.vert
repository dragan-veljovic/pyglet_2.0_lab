#version 330 core
    in vec3 position;
    in vec3 normals;
    in vec4 colors;
    in vec2 tex_coords;

    out vec3 frag_position;
    out vec3 frag_normals;
    out vec4 object_colors;
    out vec2 texture_coords;

    uniform WindowBlock
    {                       // This UBO is defined on Window creation, and available
        mat4 projection;    // in all Shaders. You can modify these matrixes with the
        mat4 view;          // Window.view and Window.projection properties.
    } window;

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1);
        frag_position = position;
        frag_normals = normals;
        object_colors = colors;
        texture_coords = tex_coords;
    }