import pyglet.window
import imgui
from imgui.integrations.pyglet import create_renderer
from pyglet.event import EVENT_HANDLED
from typing import Callable


class _ImGuiEventFilter:
    """
    Window input handler class that discards mouse and key presses down
    the event stack, if they are used to control any imgui element.

    Added by default during :py:class:`ImplementImGUI` instantiation,
    but should be pushed at the top of the stack to function properly.
    """
    def __init__(self, impl):
        self.impl = impl  # The PygletRenderer instance

    # Mouse events
    def on_mouse_press(self, x, y, button, modifiers):
        self.impl.on_mouse_press(x, y, button, modifiers)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    def on_mouse_release(self, x, y, button, modifiers):
        self.impl.on_mouse_release(x, y, button, modifiers)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.impl.on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.impl.on_mouse_scroll(x, y, scroll_x, scroll_y)
        if imgui.get_io().want_capture_mouse:
            return EVENT_HANDLED

    # Keyboard events
    def on_key_press(self, symbol, modifiers):
        self.impl.on_key_press(symbol, modifiers)
        if imgui.get_io().want_capture_keyboard:
            return EVENT_HANDLED

    def on_key_release(self, symbol, modifiers):
        self.impl.on_key_release(symbol, modifiers)
        if imgui.get_io().want_capture_keyboard:
            return EVENT_HANDLED

    def on_text(self, text):
        self.impl.on_text(text)
        if imgui.get_io().want_capture_keyboard:
            return EVENT_HANDLED


class ImplementImGUI:
    """
    Implement ImGUI in existing pyglet Window.

    By default, it will push new handler to the window, discarding mouse clicks
    and other inputs directed at the GUI element.

    Call :py:meth:`render` after drawing batches in your window on_draw,
    in order to display the gui. Define your custom imgui design function
    and pass its handle as :py:param:`imgui_design`.
    """
    def __init__(
            self,
            window: pyglet.window.Window,
            imgui_design: Callable = None,
            scale=1.0,
            save_state=False,
    ):
        self._window = window
        if imgui_design:
            self.imgui_design = imgui_design
        else:
            self.imgui_design = self.default_design

        imgui.create_context()
        self._impl = create_renderer(self._window)
        self._filter = _ImGuiEventFilter(self._impl)
        self.push_imgui_handler_to_top()

        io = imgui.get_io()
        io.font_global_scale = scale
        if not save_state:
            io.ini_file_name = None

    def render(self):
        # Start new ImGui frame
        imgui.new_frame()
        self.imgui_design()
        # Render ImGui
        imgui.render()
        self._impl.render(imgui.get_draw_data())

    @staticmethod
    def default_design():
        """
        Default demo window for error checking.
        Pass your own gui design function handle to be rendered.
        """
        imgui.show_demo_window()

    def push_imgui_handler_to_top(self):
        # Remove if already present
        try:
            self._window.remove_handlers(self._filter)
        except ValueError:
            pass  # not in stack yet

        # Push again (as LAST = highest priority)
        self._window.push_handlers(self._filter)




