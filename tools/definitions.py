"""Useful convenience definitions for Pyglet."""
import textwrap
from pathlib import Path
import pyglet
import time
from pyglet.graphics.shader import ShaderProgram, Shader
import logging
from typing import Callable
from pyglet.window import Window
import threading
import code

DEBUG_MODE = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(levelname)s: %(name)s: %(message)s'
)


def get_logger(name=None):
    return logging.getLogger(name)


def set_background_color(r: int = 20, g: int = 30, b: int = 40, a=255):
    """ Sets glClearColor for passed rgb(a) 0-255 values."""
    pyglet.gl.glClearColor(r / 255, g / 255, b / 255, a / 255)


def center_window(window: pyglet.window.Window):
    pass
    """ Centers passed window if entire window can fit in a current display resolution."""
    desk_res = pyglet.display.get_display().get_default_screen()
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


def start_app(
        window_ref: Callable[..., Window],
        default_mode=False,
        enable_console=False,
        fps: int | None = None,
        **kwargs
) -> Window:
    """
    A convenience method of initializing pyglet.window.Window and starting a pyglet loop with custom arguments.

    :param window_ref: A reference to your pyglet.window.Window subclass to start.
    :param default_mode: Create window at current desktop resolution, run in fullscreen at max refresh rate.
    :param enable_console: Start interactive console for real-time attribute changes.
        This may require starting app from your terminal ('python main.py').
        If you run through Pycharm, check "Edit Configurations -> Execution -> Emulate terminal in output console".
    :param fps: Custom desired fps at which to run pyglet loop. Defaults to 60 if left 'None'.
        It will get overridden with your monitor's max refresh rate if 'default_mode = True'.
    :param kwargs: Any other kwargs to be passed to 'Window' class during its creation.
    :param window_centered: Created 'Window' will be centered, if its resolution is less that the default screen's.

    :return: your custom pyglet.window.Window subclass

    Usage example:
            SETTINGS = {
            "default_mode": False,  # setting this to True will override width, height, fullscreen and fps settings
            'width': 1920,
            'height': 1080,
            'config': get_config(samples=2),
            'vsync': False,
            'fullscreen': True
        }

        class App(pyglet.window.Window):
            def __init__(self, **kwargs):
            super(App, self).__init__(**kwargs)
            # ... contents

        if __name__ == "__main__":
            app = start_app(App, fps=120, enable_console=True, **SETTINGS)
            # or simply 'start_app(App)' for reasonable defaults.
    """
    if default_mode:
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        current_mode = screen.get_mode()
        width, height = current_mode.width, current_mode.height
        fps = current_mode.rate
        kwargs.update({'width': width, 'height': height, 'fullscreen': True})
    else:
        fps = fps if fps is not None else 60
        if not kwargs:
            kwargs = {'width': 1280, 'height': 720, 'resizable': True}

    app = window_ref(**kwargs)

    def start_console() -> None:
        """
        App instance can be accessed with variable 'app' or 'self' in console, for convenient attribute changes.
        Example: 'app.material.diffuse=0.8' or more naturally 'self.material.diffuse=0.8"
        """
        local_vars = {'app': app, 'self': app}
        banner = "Interactive console (type `app.some_param = x` to change parameters)"
        code.interact(local=globals() | local_vars, banner=banner)

    if enable_console:
        threading.Thread(target=start_console, daemon=True).start()

    pyglet.app.run(1 / fps)

    return app


def get_default_shaders_program():
    default_vert_content = """
        #version 330 core
        in vec3 position;
        in vec4 color;
        out vec4 frag_color;
    
        uniform WindowBlock
            {
                mat4 projection;
                mat4 view;
            } window;
    
        void main() {
            vec3 new_position = position;
            gl_Position = window.projection * window.view * vec4(new_position, 1.0);
            frag_color = color;
        }
    """

    default_frag_content = """
    #version 330 core
    in vec4 frag_color;
    out vec4 final_color;
    
    void main() {
        final_color = frag_color;
    }
    """

    return ShaderProgram(
        Shader(default_vert_content, 'vertex'),
        Shader(default_frag_content, 'fragment')
    )


def create_default_shader_program_from_files() -> ShaderProgram:
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
            in vec4 color;
            
            out vec4 frag_color;

            uniform WindowBlock
                {
                    mat4 projection;
                    mat4 view;
                } window;

            void main() {
                vec3 new_position = position;
                gl_Position = window.projection * window.view * vec4(new_position, 1.0);
                frag_color = color;
            }
        """)
        with open(vert_file, 'w') as file:
            file.write(default_vert_content)

    if not frag_file.exists():
        default_frag_content = textwrap.dedent("""\
            #version 330 core
            in vec4 frag_color;
            out vec4 final_color;

            void main() {
                final_color = frag_color;
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


def load_shader_program_from_files(*shader_files: str, path="shaders/") -> ShaderProgram:
    """Get shader program from passed shader file name[s], placed in root/shaders/ directory."""
    shaders = [pyglet.resource.shader(path + file) for file in shader_files]
    return ShaderProgram(*shaders)


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
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
