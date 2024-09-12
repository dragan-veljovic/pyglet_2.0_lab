"""Useful definitions for pyglet Window"""
import textwrap
from pathlib import Path
from typing import Callable

import pyglet
import timeit


def set_background_color(r: int = 20, g: int = 30, b: int = 40, a=255):
    """ Sets glClearColor for passed rgb(a) 0-255 values."""
    pyglet.gl.glClearColor(r / 255, g / 255, b / 255, a / 255)


def center_window(window: pyglet.window.Window):
    """ Centers passed window if entire window can fit in a current display resolution."""
    desk_res = pyglet.canvas.Display().get_default_screen()
    width, height = window.width, window.height
    if desk_res.width > width and desk_res.height > height:
        window.set_location((desk_res.width - window.width) // 2,
                            (desk_res.height - window.height) // 2)


def get_config(
        samples=4,
        sample_buffers=1,
        depth_size=16,
        double_buffer=True
):
    return pyglet.gl.Config(
        samples=samples,
        sample_buffers=sample_buffers,
        depth_size=depth_size,
        double_buffer=double_buffer
    )


def start_default_display_mode(window_handle: Callable[..., pyglet.window.Window], **kwargs):
    """
    Convenience method to create a subclassed pyglet Window in current
    desktop resolution and start pyglet loop at monitor's max refresh rate.

    :param window_handle: a callable to a pyglet window to start, ex. MyWindow (without brackets)
    :param kwargs: additional keyword arguments to be passed to window

    Example:
        if __name__ == "__main__":
            start_default_display_mode(App, fullscreen=True, config=get_config())
    """
    display = pyglet.canvas.get_display()
    screen = display.get_default_screen()
    current_mode = screen.get_mode()
    width, height = current_mode.width, current_mode.height
    refresh_rate = current_mode.rate
    window_handle(width=width, height=height, **kwargs)
    pyglet.app.run(1/refresh_rate)


def get_default_shader_program() -> pyglet.graphics.shader.ShaderProgram:
    """
    Get shader program from default.vert and default.frag files, if they exist in root's "shaders" folder.
    If files (or folder) is not found, they are created.
    """
    project_root = Path.cwd()
    shaders_dir = project_root / "shaders"

    if not shaders_dir.exists():
        shaders_dir.mkdir()

    vert_file = shaders_dir / "default.vert"
    frag_file = shaders_dir / "default.frag"

    if not vert_file.exists():
        default_vert_content = textwrap.dedent("""\
            #version 330 core
            in vec3 position;
            in vec4 colors;
            out vec4 vertex_colors;

            uniform float time;

            uniform WindowBlock
                {
                    mat4 projection;
                    mat4 view;
                } window;

            void main() {
                vec3 new_position = position;
                gl_Position = window.projection * window.view * vec4(new_position, 1.0);
                vertex_colors = colors;
            }
        """)
        with open(vert_file, 'w') as file:
            file.write(default_vert_content)

    if not frag_file.exists():
        default_frag_content = textwrap.dedent("""\
            #version 330 core
            in vec4 vertex_colors;
            out vec4 final_colors;

            uniform float time;

            void main() {
                final_colors = vertex_colors;
            }
        """)
        with open(frag_file, 'w') as file:
            file.write(default_frag_content)

    pyglet.resource.path = ["shaders/"]
    pyglet.resource.reindex()

    vertex_shader = pyglet.resource.shader("default.vert")
    fragment_shader = pyglet.resource.shader("default.frag")
    shader_program = pyglet.graphics.shader.ShaderProgram(vertex_shader, fragment_shader)
    return shader_program


def time_it(func_handle):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func_handle(*args, **kwargs)
        print(f"{func_handle.__name__} Execution time: ", timeit.default_timer() - start_time)
        return result

    return wrapper

# simple palette - use color module for more colors
palette = {"orange": (209, 60, 23, 255), "yellow": (219, 204, 101, 255), "grey": (112, 112, 112, 255),
           "light-grey": (185, 185, 186, 255), "brown": (71, 56, 44, 255), "teal": (0, 128, 128, 255),
           "mild-white": (240, 240, 240, 255), "white": (255, 255, 255, 255),
           "blue": (2, 145, 173, 255), "purple": (119, 101, 219, 255), "bkg": (24, 39, 54, 255),
           "app1": (159, 231, 245, 255), "app2": (66, 158, 189, 255), "app3": (5, 63, 92, 255),
           "app4": (247, 173, 25, 255), "app5": (242, 127, 12, 255), "app6": (255, 132, 0, 255)}


def show_palette(batch, group=None):
    color_display = []
    color_labels = []
    for i, (name, rgb) in enumerate(palette.items()):
        color_display.append(pyglet.shapes.Rectangle(100, 640 - i * 35, 30, 30, rgb[:3], batch=batch, group=group))
        color_labels.append(pyglet.text.Label(f"'{name}'", x=150, y=640 - i * 35 + 10, batch=batch, group=group))
    return color_display
