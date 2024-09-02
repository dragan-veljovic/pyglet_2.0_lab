import pyglet
from pyglet.gl import *
from pyglet.math import Mat4

window = pyglet.window.Window(width=540, height=540, resizable=True)
batch = pyglet.graphics.Batch()
print(f"OpenGL Context: {window.context.get_info().version}")

vertex_source = """
    #version 330 core
    in vec3 position;
    in vec4 colors;
    in vec3 tex_coords;
    out vec4 vertex_colors;
    out vec3 texture_coords;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;  

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1.0);

        vertex_colors = colors;
        texture_coords = tex_coords;
    }
"""

fragment_source = """
    #version 330 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        final_colors = texture(our_texture, texture_coords.xy) + vertex_colors;
    }
"""

# pyglet.options['debug_gl_shaders'] = True

# manual shader program
vert_shader = pyglet.graphics.shader.Shader(vertex_source, 'vertex')
frag_shader = pyglet.graphics.shader.Shader(fragment_source, 'fragment')
program = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
vlist = program.vertex_list(3, GL_TRIANGLES, batch=batch, colors='Bn')
vlist.position[:] = (200, 400, 0, 300, 350, 0, 300, 450, 0)
vlist.colors[:] = (255, 0, 0, 255,  0, 255, 0, 255,  75, 75, 255, 255)
window.view = window.view.scale((0.4, 0.4, 1))
window.projection = Mat4.perspective_projection(3/4, 0.1, 1000, fov=80)
##########################################################
#   TESTS !
##########################################################
label = pyglet.text.Label("This is a test", x=0, y=180, dpi=200, color=(255, 25, 255, 150), batch=batch)

vertex_list = program.vertex_list(3, GL_TRIANGLES,
                                  position=('f', (100, 300, 0,  200, 250, 0,  200, 350, 0)),
                                  colors=('f', (1, 0, 0, 1,  0, 1, 0, 1,  0.3, 0.3, 1, 1)))


def create_quad_vertex_list(x, y, z, width, height):
    return x, y, z, x + width, y, z, x + width, y + height, z, x, y + height, z


program.vertex_list_indexed(4, GL_TRIANGLES, [0, 1, 2, 0, 2, 3], batch, None,
                            position=('f', create_quad_vertex_list(480, 270, -11, 50, 50)),
                            colors=('f', (1, 0.5, 0.2, 1, 1, 0.5, 0.2, 1, 1, 0.5, 0.2, 1, 1, 0.5, 0.2, 1)))

program.vertex_list_indexed(4, GL_TRIANGLES, [0, 1, 2, 0, 2, 3], batch, None,
                            position=('f', (400, 400, 0, 400+50, 400, 0, 400+50, 400+50, 0, 400, 400+50, 0)),
                            colors=('f', (1, 0.5, 0.2, 1, 1, 0.5, 0.2, 1, 1, 0.5, 0.2, 1, 1, 0.5, 0.2, 1)))


img = pyglet.image.load("Ball.png")
img.anchor_x = img.width // 2
img.anchor_y = img.height // 2
red = pyglet.image.SolidColorImagePattern((255, 0, 0, 255)).create_image(50, 50)
green = pyglet.image.SolidColorImagePattern((0, 255, 0, 255)).create_image(50, 50)
blue = pyglet.image.SolidColorImagePattern((0, 0, 255, 255)).create_image(50, 50)
white = pyglet.image.SolidColorImagePattern((255, 255, 255, 255)).create_image(50, 50)

sprites = [pyglet.sprite.Sprite(img=img, x=60, y=80, batch=batch),
           pyglet.sprite.Sprite(img=img, x=110, y=90, batch=batch),
           pyglet.sprite.Sprite(img=img, x=160, y=100, batch=batch),
           pyglet.sprite.Sprite(img=img, x=210, y=110, batch=batch)]
for sprite in sprites:
    sprite.opacity = 220

# sprite2 = pyglet.sprite.Sprite(img=red, x=200, y=400, batch=batch)
# sprite3 = pyglet.sprite.Sprite(img=green, x=300, y=300, batch=batch)
# sprite4 = pyglet.sprite.Sprite(img=blue, x=400, y=200, batch=batch)
#sprite5 = pyglet.sprite.Sprite(img=white, x=500, y=100, batch=batch)

#standalone_sprite = pyglet.sprite.Sprite(img=white, x=600, y=0)


##########################################################
# Modify the sprite scale value by scrolling the mouse
##########################################################

@window.event
def on_mouse_scroll(x, y, mouse, direction):
    for spr in sprites:
        spr.scale += direction / 10


###########################################################
#
###########################################################
@window.event
def on_draw():
    window.clear()

    # pyglet.graphics.draw(3, GL_TRIANGLES, ('position3f', (100, 100, 0,  200, 100, 0,  150, 200, 0)),
    #                                       ('colors3f', (1, 0.5, 0.2,  1, 0.5, 0.2,  1, 0.5, 0.2)))
    #
    # pyglet.graphics.draw_indexed(4, GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
    #                              ('position2i', (225, 300,   250, 300,   250, 325,   225, 325)),
    #                              ('colors3f', (0.5, 1, 0.2,  0.5, 0.2, 1,  0.2, 0.5, 1, 1, 0.5, 0.2)))

    vertex_list.draw(GL_TRIANGLES)
    batch.draw()
    #standalone_sprite.draw()

def update(dt):
    for sprite in sprites:
        sprite.rotation += 100 * dt % 360


if __name__ == "__main__":
    pyglet.gl.glClearColor(0.2, 0.3, 0.3, 1)
    #pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()