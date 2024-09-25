"""Useful definitions for pyglet Window"""
import pyglet
import timeit
from pyglet.graphics.shader import ShaderProgram, Shader


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


def get_shader_program(*shader_files: str, path="shaders/") -> ShaderProgram:
    shaders = [pyglet.resource.shader(path + file) for file in shader_files]
    return ShaderProgram(*shaders)


def time_it(func_handle):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func_handle(*args, **kwargs)
        print(f"{func_handle.__name__} Execution time: ", timeit.default_timer() - start_time)
        return result

    return wrapper


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
