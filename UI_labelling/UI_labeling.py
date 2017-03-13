"""kivy
"""
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, ListProperty, OptionProperty, BooleanProperty
from kivy.vector import Vector
from kivy.clock import Clock
import kivy
kivy.require('1.9.0')

class Cursor(Widget):
    """yeah
    """
    rgba = property(None)


class SliderEngine(Widget):
    """yeah
    """
    # slider bar
    cursorStart = ObjectProperty(None)
    cursorEnd = ObjectProperty(None)
    offset_x = NumericProperty(50.0)
    offset_y = NumericProperty(50.0)
    # Smoothe Line
    points = ListProperty([])
    selectedPoints = ListProperty([])
    alpha = NumericProperty(0.5)
    joint = OptionProperty('none', options=('round', 'miter', 'bevel', 'none'))
    cap = OptionProperty('none', options=('round', 'square', 'none'))
    linewidth = NumericProperty(5.0)
    close = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(SliderEngine, self).__init__(**kwargs)
        self.cursorStart.rgba = (1, 1, 0, 1)
        self.cursorEnd.rgba = (1, 0, 1, 1)
        self.points = [600, 200, 200, 200, 100, 400, 300, 500,
                       500, 500, 300, 300, 500, 300,
                       500, 400, 600, 400, 700, 500]
        defaultStartIdx = int(len(self.points)/2*1/4)
        defaultEndIdx = int(len(self.points)/2*3/4+1)
        self.selectedPoints = self.points[defaultStartIdx*2:defaultEndIdx*2]

    def on_touch_move(self, touch):
        dist2CursorStart = abs(touch.x - self.cursorStart.x)
        dist2CursorEnd = abs(touch.x - self.cursorEnd.x)

        if dist2CursorStart > dist2CursorEnd:
            if touch.x < (self.width-self.offset_x) \
                and touch.x > self.cursorStart.center_x:
                self.cursorEnd.center_x = touch.x
        else:
            if touch.x < self.cursorEnd.center_x \
                and touch.x > self.offset_x:
                self.cursorStart.center_x = touch.x

        self.updateSelectedPoints()

    def getCursorMatchedPoint_idx(self, cursor):
        normalizedX = (cursor.center_x - self.offset_x) / (self.width - self.offset_x * 2)
        pointsLength = len(self.points) / 2
        return int(normalizedX * pointsLength)

    def updateSelectedPoints(self):
        startPtIdx = self.getCursorMatchedPoint_idx(self.cursorStart)
        endPtIdx = self.getCursorMatchedPoint_idx(self.cursorEnd) + 1
        self.selectedPoints = self.points[startPtIdx*2:endPtIdx*2]


class LabelingApp(App):
    """yeah
    """
    def build(self):
        slider = SliderEngine()
        return slider


if __name__ == '__main__':
    LabelingApp().run()
