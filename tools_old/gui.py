import imgui
import pyglet.window
from imgui.integrations.pyglet import create_renderer


class IntegrateImGui:
    def __init__(self, window: pyglet.window.Window):
        """
        This module separates ImGui integration from the main pyglet window.
        It handles gui initialization (in initializer), gui elements and
        functions (in gui_design() method), and rendering (in render() method).

        To enable it with pyglet, instantiate this class and pass pyglet window,
        and call this class' render() method in your window.on_draw().
        Then include gui_design() method in your pyglet window script to
        make your own gui design.
        """
        # imgui initialization
        self.window = window
        self.imgui_context = imgui.create_context()
        self.impl = create_renderer(self.window)
        self.io = imgui.get_io()
        # use set_new_font convenience method if needed
        self.new_font = None
        # check for override method
        if hasattr(self.window, "gui_design"):
            self.gui_design = self.window.gui_design

    def set_new_font(self, path_to_tff, font_size: int):
        new_font = self.io.fonts.add_font_from_file_ttf(f"{path_to_tff}", font_size)
        self.impl.refresh_font_texture()
        self.new_font = new_font

    def render(self):
        imgui.new_frame()
        self.gui_design()
        # generate raw data from gui_design instructions using native code
        imgui.render()
        # get data and actually render it to the screen using implementation
        self.impl.render(imgui.get_draw_data())

    @staticmethod
    def gui_design():
        """ This will be overridden If gui_design() method is detected in passed window."""
        imgui.set_next_window_size(400, 150, imgui.ONCE)
        imgui.set_next_window_position(100, 100, imgui.ONCE)
        imgui.begin("Test window", True, imgui.WINDOW_NO_SAVED_SETTINGS)
        imgui.text_wrapped("Imgui implementation successful!\n\n"
            "Override this with your own gui_design() in pyglet window code.")
        imgui.end()










