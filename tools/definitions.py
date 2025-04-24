"""Useful convenience definitions for Pyglet."""
import textwrap
from pathlib import Path
from typing import Type
from pyglet.graphics.shader import Shader, ShaderProgram
import pyglet
import timeit
from pyglet.graphics.shader import ShaderProgram, Shader


def set_background_color(r: int = 20, g: int = 30, b: int = 40, a=255):
    """ Sets glClearColor for passed rgb(a) 0-255 values."""
    pyglet.gl.glClearColor(r / 255, g / 255, b / 255, a / 255)


def center_window(window: pyglet.window.Window):
    pass
    """ Centers passed window if entire window can fit in a current display resolution."""
    desk_res = pyglet.display.Display().get_default_screen()
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


def start_app(window_ref: Type[pyglet.window.Window], arguments: dict = None) -> pyglet.window.Window:
    """
    A convenience method of initializing pyglet loop and Window with custom arguments.

    :param window_ref: a reference to your pyglet.window.Window subclass to start
    :param arguments: a dictionary with arguments to be passed to your window instance
        This dictionary accepts two special keys:
        "default_mode" - if its value is True, all other settings are overridden
            and window is created at desktop resolution, ran in full screen at max refresh rate
        "fps" - custom fps at which to run pyglet loop

    Example:

        SETTINGS = {
            "default_mode": False,
            "fps": 60,
            "width": 1920,
            "height": 1080,
            "resizable": True,
            "config": get_config(samples=2)
        }

        class App(pyglet.window.Window):
            def __init__(self, **kwargs):
            super(App, self).__init__(**kwargs)
            # ... contents

        if __name__ == "__main__":
        start_app(App, SETTINGS)
    """
    if arguments is None:
        window_arguments = {'width': 1280, 'height': 720, 'resizable': True}
    else:
        window_arguments = arguments.copy()

    default_mode = window_arguments.pop('default_mode') if 'default_mode' in window_arguments else False
    fps = window_arguments.pop('fps') if 'fps' in window_arguments else 60

    if default_mode:
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        current_mode = screen.get_mode()
        width, height = current_mode.width, current_mode.height
        fps = current_mode.rate
        window = window_ref(width=width, height=height, fullscreen=True)
    else:
        window = window_ref(**window_arguments)

    pyglet.app.run(1/fps)
    return window


def get_default_shader_program() -> ShaderProgram:
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
    shader_program = ShaderProgram(vertex_shader, fragment_shader)
    return shader_program


def get_shader_program(*shader_files: str, path="shaders/") -> ShaderProgram:
    """Get shader program from passed shader file name[s], placed in root/shaders/ directory."""
    shaders = [pyglet.resource.shader(path + file) for file in shader_files]
    return ShaderProgram(*shaders)


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


def get_palette(batch, group=None) -> list:
    """Returns a list of pyglet.shapes to represent a simple color palette."""
    color_display = []
    color_labels = []
    for i, (name, rgb) in enumerate(palette.items()):
        color_display.append(pyglet.shapes.Rectangle(100, 640 - i * 35, 30, 30, rgb[:3], batch=batch, group=group))
        color_labels.append(pyglet.text.Label(f"'{name}'", x=150, y=640 - i * 35 + 10, batch=batch, group=group))
    return color_display
