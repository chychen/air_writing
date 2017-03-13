"""kivy
"""
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, ListProperty, OptionProperty, BooleanProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.config import Config
import kivy
kivy.require('1.9.0')
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')

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
    # trajectory
    points = ListProperty([])
    selectedPoints = ListProperty([])
    alpha = NumericProperty(0.5)
    joint = OptionProperty('none', options=('round', 'miter', 'bevel', 'none'))
    cap = OptionProperty('none', options=('round', 'square', 'none'))
    linewidth = NumericProperty(5.0)
    close = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(SliderEngine, self).__init__(**kwargs)
        self.size_hint = kwargs['size_hin']
        # cursor
        self.cursorStart.rgba = (1, 1, 0, 1)
        self.cursorEnd.rgba = (1, 0, 1, 1)
        # trajectory
        self.points = [600, 200, 200, 200, 100, 400, 300, 500,
                       500, 500, 300, 300, 500, 300,
                       500, 400, 600, 400, 700, 500]
        defaultStartIdx = int(len(self.points)/2*1/4)
        defaultEndIdx = int(len(self.points)/2*3/4+1)
        self.selectedPoints = self.points[defaultStartIdx*2:defaultEndIdx*2]

    def on_touch_move(self, touch):
        super(SliderEngine, self).on_touch_down(touch)
        self.touchAction(touch)

    def on_touch_down(self, touch):
        super(SliderEngine, self).on_touch_down(touch)
        self.touchAction(touch)

    def getCursorMatchedPoint_idx(self, cursor):
        normalizedX = (cursor.center_x - self.offset_x) / (self.width - self.offset_x * 2)
        pointsLength = len(self.points) / 2
        return int(normalizedX * pointsLength)

    def updateSelectedPoints(self):
        startPtIdx = self.getCursorMatchedPoint_idx(self.cursorStart)
        endPtIdx = self.getCursorMatchedPoint_idx(self.cursorEnd) + 1
        self.selectedPoints = self.points[startPtIdx*2:endPtIdx*2]
    
    def touchAction(self, touch):
        dist2CursorStart = abs(touch.x - self.cursorStart.x)
        dist2CursorEnd = abs(touch.x - self.cursorEnd.x)
        if dist2CursorStart > dist2CursorEnd:
            if touch.x < (self.width-self.offset_x) \
                and touch.x > self.cursorStart.center_x \
                and touch.y < self.height:
                self.cursorEnd.center_x = touch.x
        else:
            if touch.x < self.cursorEnd.center_x \
                and touch.x > self.offset_x \
                and touch.y < self.height:
                self.cursorStart.center_x = touch.x

        self.updateSelectedPoints()


class LabelTest(FloatLayout):

    def __init__(self, *args, **kwargs):
        super(LabelTest, self).__init__(*args, **kwargs)
        slider = SliderEngine(
            size_hin=(1, .8),
            pos_hint={'x':0, 'y':0}
        )
        # button
        lastButton = Button(
            on_press=self.lastButtoncallback,
            text="Last",
            size_hin=(.5, .2),
            pos_hint={'x':0, 'y':.8}
        )
        lastButton.font_size = 24
        lastButton.background_color = (0, 1, 1, 1)
        nextButton = Button(
            on_press=self.nextButtoncallback,
            text="Next",
            size_hin=(.5, .2),
            pos_hint={'x':.5, 'y':.8}
        )
        self.add_widget(slider)
        self.add_widget(lastButton)
        self.add_widget(nextButton)
    
    def lastButtoncallback(self, instance):
        print 'The button <%s> is being pressed' % instance.text

    def nextButtoncallback(self, instance):
        print 'The button <%s> is being pressed' % instance.text
            

class LabelingApp(App):
    """yeah
    """
    def build(self):
        Label = LabelTest()
        return Label


if __name__ == '__main__':
    LabelingApp().run()
