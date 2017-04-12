from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import codecs
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty, ListProperty, OptionProperty, BooleanProperty
from kivy.uix.button import Button
from kivy.config import Config
from kivy.graphics import Point, Color, Line
import kivy
kivy.require('1.9.0')
Config.set('graphics', 'width', '1800')
Config.set('graphics', 'height', '1200')


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NORMALIZED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'normalized_voc')
TARGET_FILE_PATH = os.path.join(NORMALIZED_DATA_DIR_PATH, 'User_0')


class StartCursor(Widget):
    def __init__(self, *args, **kwargs):
        super(StartCursor, self).__init__(*args, **kwargs)


class EndCursor(Widget):
    def __init__(self, *args, **kwargs):
        super(EndCursor, self).__init__(*args, **kwargs)


class SlideBar(Widget):
    y_offset = NumericProperty(30.0)

    def __init__(self, *args, **kwargs):
        super(SlideBar, self).__init__(*args, **kwargs)

    # def on_touch_down(self, touch):
    #     super(SlideBar, self).on_touch_down(touch)
    #     s1 = StartCursor(pos=(touch.x, self.height * 7 / 16))
    #     self.add_widget(s1)


class DrawingBoard(Widget):
    # trajectory
    y_offset = NumericProperty(200.0)
    points = ListProperty([])

    def __init__(self, *args, **kwargs):
        super(DrawingBoard, self).__init__(*args, **kwargs)

    def init_board(self, points, voc_length):
        self.canvas.clear()
        self.points = points
        voc_lines = Line(points=self.points, width=4)
        self.canvas.add(Color(.4, .4, 1, .3))
        self.canvas.add(voc_lines)
        voc_points = Point(points=self.points, pointsize=5)
        self.canvas.add(Color(.4, .4, 1, 1))
        self.canvas.add(voc_points)

        # add default cursors
        self.all_selected_points_list = []
        self.all_cursor_list = []
        for i in range(voc_length - 1):
            # start cursor
            start_x = ((i + 1) / voc_length - 1.0 / 4.0 / voc_length)
            temp_start_cursor = StartCursor(
                pos=(start_x * self.width, SlideBar().y_offset))
            self.add_widget(temp_start_cursor)
            self.all_cursor_list.append(temp_start_cursor)
            # end cursor
            end_x = ((i + 1) / voc_length + 1.0 / 4.0 / voc_length)
            temp_end_cursor = EndCursor(
                pos=(end_x * self.width, SlideBar().y_offset))
            self.add_widget(temp_end_cursor)
            self.all_cursor_list.append(temp_end_cursor)
            # selected points
            start_point_idx = int(len(self.points) / 2 * start_x)
            end_point_idx = int(len(self.points) / 2 * end_x) + 1
            temp_selected_points = self.points[start_point_idx *
                                               2: end_point_idx * 2]
            self.all_selected_points_list.append(temp_selected_points)

        # record pointers pointing to canvas's Point in
        # 'self.all_canvas_point_list'
        self.all_canvas_point_list = []
        for selected_points in self.all_selected_points_list:
            # temp_P = Point(points=selected_points, pointsize=5)
            temp_P = Line(points=selected_points, width=3)
            self.all_canvas_point_list.append(temp_P)
            self.canvas.add(Color(1., 0, 0))
            self.canvas.add(temp_P)

    def on_touch_move(self, touch):
        super(DrawingBoard, self).on_touch_down(touch)
        self.touch_action(touch)

    def on_touch_down(self, touch):
        super(DrawingBoard, self).on_touch_down(touch)
        self.touch_action(touch)

    def get_cursor_matched_point_idx(self, cursor):
        normalized_x = cursor.center_x / self.width
        pointsLength = len(self.points) / 2
        return int(normalized_x * pointsLength)

    def update_selected_points(self):
        for i, canvas_point in enumerate(self.all_canvas_point_list):
            startPtIdx = self.get_cursor_matched_point_idx(
                self.all_cursor_list[i * 2])
            endPtIdx = self.get_cursor_matched_point_idx(
                self.all_cursor_list[i * 2 + 1]) + 1
            canvas_point.points = self.points[startPtIdx * 2: endPtIdx * 2]

    def touch_action(self, touch):
        # select the cloest cursor to modify its center_x
        closest_cursor = None
        closest_cursor_id = -1
        min_dist = sys.maxint
        for i, cursor in enumerate(self.all_cursor_list):
            temp_dist = abs(touch.x - cursor.x)
            if temp_dist < min_dist:
                min_dist = temp_dist
                closest_cursor = cursor
                closest_cursor_id = i
        if touch.x < self.width and touch.y < self.height:
            closest_cursor.center_x = touch.x

        # make sure the cursor's center_x value won't exceed neighbors
        if closest_cursor_id > 0 and closest_cursor_id < len(self.all_cursor_list) - 1:
            if self.all_cursor_list[closest_cursor_id + 1].x < closest_cursor.center_x:
                closest_cursor.center_x = self.all_cursor_list[closest_cursor_id + 1].x - 1
            if self.all_cursor_list[closest_cursor_id - 1].x > closest_cursor.center_x:
                closest_cursor.center_x = self.all_cursor_list[closest_cursor_id - 1].x + 1

        self.update_selected_points()


class AppEngine(FloatLayout):
    """
    main app
    """
    lastButton = ObjectProperty(None)
    nextButton = ObjectProperty(None)
    board = ObjectProperty(None)

    def __init__(self, *args, **kwargs):
        super(AppEngine, self).__init__(*args, **kwargs)
        self.lastButton.bind(on_press=self.lastButtoncallback)
        self.nextButton.bind(on_press=self.nextButtoncallback)

        filename = TARGET_FILE_PATH + ".json"
        with codecs.open(filename, 'r', 'utf-8-sig') as f:
            json_data = json.load(f)
        self.vocs_idx_counter = -1
        self.all_vocs_data = json_data['data']
        self.vocs_amount = len(json_data['data'])

    def lastButtoncallback(self, instance):
        print ('The button <%s> is being pressed' % instance.text)
        temp_idx = self.vocs_idx_counter - 1
        if self.vocs_idx_counter >= 0:
            points, voc_length = self.read_voc_from_json( self.all_vocs_data.keys()[self.vocs_idx_counter])
            self.board.init_board(points, voc_length)
            self.vocs_idx_counter = temp_idx
        else:
            # end
            pass

    def nextButtoncallback(self, instance):
        print ('The button <%s> is being pressed' % instance.text)
        temp_idx = self.vocs_idx_counter + 1
        if temp_idx < self.vocs_amount:
            points, voc_length = self.read_voc_from_json( self.all_vocs_data.keys()[self.vocs_idx_counter])
            self.board.init_board(points, voc_length)
            self.vocs_idx_counter = temp_idx
        else:
            # end
            pass

    def read_voc_from_json(self, voc):
        print (voc)
        print (len(str(voc)))
        voc_length = len(str(voc))
        voc_pos_list = []
        voc_timestep_list = self.all_vocs_data[voc]
        for time_step_dict in voc_timestep_list:
            # i: timestep in one voc as dict format
            voc_pos_list.append(time_step_dict['pos'])
        # TODO:!!!!!!!!!!!!!!!!!!!!!! auto next voc
        scaled_pos = np.array(voc_pos_list)
        # normalization
        x_amax = np.amax(scaled_pos[:, 0])
        x_amin = np.amin(scaled_pos[:, 0])
        x_range = x_amax - x_amin
        x_scale = 1.0 / x_range
        scaled_pos[:, 0] = scaled_pos[:, 0] * x_scale * self.board.width
        scaled_pos[:, 1] = scaled_pos[:, 1] * x_scale * self.board.height + self.board.y_offset

        return scaled_pos.flatten().tolist(), voc_length


class LabelingApp(App):
    """
    main app builder
    """

    def build(self):
        Label = AppEngine()
        return Label


if __name__ == '__main__':
    LabelingApp().run()
