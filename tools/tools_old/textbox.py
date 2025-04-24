import pyglet
from pyglet.window import mouse
from pyglet.window import key


class TextBox:
    textboxes = []
    lorem_impsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mauris vitae sapien finibus, tempus massa non, cursus nibh. Nulla elit lorem, rhoncus quis congue quis, ultricies sit amet neque. Morbi ac urna eget orci mattis sodales."

    def __init__(self, window: pyglet.window.Window, x: int, y: int, batch=None, group=None):
        print("Textbox instance created")
        self.window = window
        self.x, self.y = x, y
        self.batch = batch
        self.group = group
        self.doc = pyglet.text.document.UnformattedDocument(self.lorem_impsum)
        self.font_size = 20
        self.doc.set_style(0, 0, dict(font_name="Segoe UI", font_size=self.font_size,
                                      color=(255, 255, 255, 255),
                                      background_color=(30, 30, 30, 100)))

        # # adaptive initial textbox width
        # self.width_clearance = self.window.width - self.x
        # if self.width_clearance >= window.width // 2:
        #     self.initial_width = window.width // 2
        # else:
        #     self.initial_width = max(self.width_clearance, 300)
        self.initial_width = 700
        self.layout = pyglet.text.layout.IncrementalTextLayout(self.doc, width=self.initial_width,
                                                               height=(self.font_size*2*2), multiline=True,
                                                               wrap_lines=True, batch=self.batch, group=self.group)
        self.layout.position = (self.x, self.y - self.layout.height)
        self.caret = pyglet.text.caret.Caret(self.layout)
        self.caret.color = (255, 255, 255)
        self.active = None
        self.rectangle = None

        self.textboxes.append(self)
        self.activate()

    def event_dispatcher(self, event: str, args: list):
        if event == 'on_mouse_press':
            x, y, button, mods = args[0], args[1], args[2], args[3]
            if button == mouse.LEFT:
                # if is_double_click(x, y, button):
                #     self.activate()
                self.activate()

        elif event == 'on_mouse_release':
            pass

        elif event == 'on_mouse_drag':
            x, y, dx, dy, buttons, mods = args[0], args[1], args[2], args[3], args[4], args[5]
            if buttons == mouse.LEFT:
                self.layout.x += dx
                self.layout.y += dy
            if buttons == mouse.RIGHT:
                self.layout.width += dx
                self.layout.height += dy
                self.height_adjustment()
                # empty set_style call to apply changes (when not using tick updating)
                self.doc.set_style(0, 0, dict())

        elif event == 'on_mouse_scroll':
            x, y, scroll_x, scroll_y = args[0], args[1], args[2], args[3]
            current_font_size = self.doc.get_style('font_size', 0)
            new_font_size = current_font_size + scroll_y
            self.doc.set_style(0, 10, dict(font_size=new_font_size))
            self.height_adjustment()

    @classmethod
    def on_key_press_dispatcher(cls, button, modifiers):
        # x, y = window._mouse_x, window._mouse_y
        # selection = check_selection(x, y)
        # if button == key.F1:
        #     TextBox(window, x, y)
        if button == key.F2:
            for tb in cls.textboxes:
                if tb.active:
                    tb.deactivate()
        elif button == key.ESCAPE:
            for tb in cls.textboxes:
                if tb.active:
                    tb.deactivate()
        # elif button == key.F3:
        #     if selection:
        #         selection.delete()
        # elif button == key.F4:
        #     if selection:
        #         selection.doc.text = cls.lorem_impsum
        #         print(selection.doc.text)

    def deactivate(self):
        self.active = False
        self.window.remove_handlers(self.caret)
        self.caret.visible = False
        if self.rectangle:
            self.rectangle.delete()
        self.height_adjustment()
        # calling an empty set_style to apply changes (for when not using tick updating)
        self.doc.set_style(0, 0, dict())

    def activate(self):
        self.active = True
        self.rectangle = pyglet.shapes.Rectangle(self.layout.x, self.layout.y, self.layout.width, self.layout.height,
                                                 batch=self.batch, color=(200, 100, 0), group=self.group)
        self.rectangle.opacity = 50
        self.window.push_handlers(self.caret)
        self.caret.visible = True

    def height_adjustment(self):
        # resizing layout based number of lines
        old_height = self.layout.height
        lines = self.layout.get_line_count()
        font_size = self.doc.get_style('font_size', 0)
        new_height = int(lines * font_size * 2)  # crashes if not int
        self.layout.height = new_height
        # adjusting layout.y so that first line stays at same height
        self.layout.y -= new_height - old_height

    def delete(self):
        self.doc.delete_text(0, len(self.doc.text))
        self.layout.delete()
        self.textboxes.remove(self)