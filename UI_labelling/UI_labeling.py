"""kivy
"""
import json
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, ListProperty, OptionProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.config import Config
import kivy
kivy.require('1.9.0')
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '800')

class Cursor(Widget):
    """yeah
    """
    pass

class DrawingBoard(FloatLayout):
    """yeah
    """
    # trajectory
    points = ListProperty([])
    selectedPoints = ListProperty([])
    alpha = NumericProperty(0.5)

    def __init__(self, **kwargs):
        super(DrawingBoard, self).__init__(**kwargs)
        # trajectory
        collectedDataDir = 'testing_data/'
        dataName = '2017-03-15 13:04:16.293349'
        fileName = collectedDataDir + dataName + ".json"
        f = open(fileName, 'r')
        jsonData = json.load(f)
        f.close()
        
        for i, v in enumerate(jsonData['trajectory_list']):
            jsonData['trajectory_list'][i] = v * 800
        self.points = jsonData['trajectory_list']
        defaultStartIdx = int(len(self.points)/2*1/4)
        defaultEndIdx = int(len(self.points)/2*3/4+1)
        self.selectedPoints = self.points[defaultStartIdx*2:defaultEndIdx*2]

    def updateselectedPoints(self, startIdx, endIdx):
        self.selectedPoints = self.points[startIdx: endIdx]


class SliderEngine(Widget):
    """yeah
    """
    # slider bar
    cursorStart = ObjectProperty(None)
    cursorEnd = ObjectProperty(None)
    offset_x = NumericProperty(50.0)
    offset_y = NumericProperty(50.0)
    # trajectory
    board = ObjectProperty(None)

    def on_touch_move(self, touch):
        super(SliderEngine, self).on_touch_down(touch)
        self.touchAction(touch)

    def on_touch_down(self, touch):
        super(SliderEngine, self).on_touch_down(touch)
        self.touchAction(touch)

    def getCursorMatchedPoint_idx(self, cursor):
        normalizedX = (cursor.center_x - self.offset_x) / (self.width - self.offset_x * 2)
        pointsLength = len(self.board.points) / 2
        return int(normalizedX * pointsLength)

    def updateSelectedPoints(self):
        startPtIdx = self.getCursorMatchedPoint_idx(self.cursorStart)
        endPtIdx = self.getCursorMatchedPoint_idx(self.cursorEnd) + 1
        self.board.updateselectedPoints(startPtIdx*2, endPtIdx*2)
    
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


class LabelEngine(FloatLayout):
    lastButton = ObjectProperty(None)
    nextButton = ObjectProperty(None)

    def __init__(self, *args, **kwargs):
        super(LabelEngine, self).__init__(*args, **kwargs)
        self.lastButton.bind(on_press=self.lastButtoncallback)
        self.nextButton.bind(on_press=self.nextButtoncallback)

    
    def lastButtoncallback(self, instance):
        print 'The button <%s> is being pressed' % instance.text

    def nextButtoncallback(self, instance):
        print 'The button <%s> is being pressed' % instance.text
            

class LabelingApp(App):
    """yeah
    """
    def build(self):
        Label = LabelEngine()
        return Label


if __name__ == '__main__':
    LabelingApp().run()