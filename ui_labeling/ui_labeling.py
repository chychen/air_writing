from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import codecs
import random
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
LABELED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'labeled_voc')
USER_FILE_NAME = 'User_0'
TARGET_FILE_PATH = os.path.join(NORMALIZED_DATA_DIR_PATH, USER_FILE_NAME)
RESULT_FILE_PATH = os.path.join(LABELED_DATA_DIR_PATH, USER_FILE_NAME)


class StartCursor(Widget):
    def __init__(self, *args, **kwargs):
        super(StartCursor, self).__init__(*args, **kwargs)
        self.rgb = kwargs['color']


class EndCursor(Widget):
    def __init__(self, *args, **kwargs):
        super(EndCursor, self).__init__(*args, **kwargs)
        self.rgb = kwargs['color']


class SlideBar(Widget):
    y_offset = NumericProperty(30.0)

    def __init__(self, *args, **kwargs):
        super(SlideBar, self).__init__(*args, **kwargs)


class DrawingBoard(Widget):

    def __init__(self, *args, **kwargs):
        super(DrawingBoard, self).__init__(*args, **kwargs)
        self.y_offset = 200.0
        self.points = []
        self.all_selected_points_list = []
        self.all_selected_points_idx_list = []
        self.all_cursor_list = []
        self.all_canvas_point_list = []
        self.all_connectionist_color_list = []

    def init_board(self, points, voc_length):
        self.canvas.clear()
        self.points = points
        voc_lines = Line(points=self.points, width=4)
        self.canvas.add(Color(.6, .6, .6, .6))
        self.canvas.add(voc_lines)
        voc_points = Point(points=self.points, pointsize=5)
        self.canvas.add(Color(.6, .6, .6, 1))
        self.canvas.add(voc_points)

        # add default cursors and correspond selected points
        self.all_selected_points_list = []
        self.all_selected_points_idx_list = []
        self.all_cursor_list = []
        self.all_connectionist_color_list = []
        for i in range(voc_length - 1):
            # random colors
            # start from 7 for brighter color space
            r = random.randint(7, 11) / 10.0
            g = random.randint(0, 11) / 10.0
            b = random.randint(0, 11) / 10.0
            self.all_connectionist_color_list.append(Color(r, g, b))
            # start cursor
            start_x = ((i + 1) / voc_length - 1.0 / 4.0 / voc_length)
            temp_start_cursor = StartCursor(
                pos=(start_x * self.width, SlideBar().y_offset), color=[r, g, b])
            self.add_widget(temp_start_cursor)
            self.all_cursor_list.append(temp_start_cursor)
            # end cursor
            end_x = ((i + 1) / voc_length + 1.0 / 4.0 / voc_length)
            temp_end_cursor = EndCursor(
                pos=(end_x * self.width, SlideBar().y_offset), color=[r, g, b])
            self.add_widget(temp_end_cursor)
            self.all_cursor_list.append(temp_end_cursor)
            # selected points
            start_point_idx = int(len(self.points) / 2 * start_x)
            end_point_idx = int(len(self.points) / 2 * end_x) + 1
            temp_selected_points = self.points[start_point_idx *
                                               2: end_point_idx * 2]
            self.all_selected_points_list.append(temp_selected_points)
            for selected_idx in range(start_point_idx, end_point_idx, 1):
                self.all_selected_points_idx_list.append(selected_idx)

        # record pointers pointing to canvas's Point in
        # 'self.all_canvas_point_list'
        self.all_canvas_point_list = []
        for i, selected_points in enumerate(self.all_selected_points_list):
            # temp_P = Point(points=selected_points, pointsize=5)
            temp_P = Line(points=selected_points, width=3)
            self.all_canvas_point_list.append(temp_P)
            self.canvas.add(self.all_connectionist_color_list[i])
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
        # '+-10' is for foolproof
        if closest_cursor_id > 0 and closest_cursor_id < len(self.all_cursor_list) - 1:
            if self.all_cursor_list[closest_cursor_id + 1].x < closest_cursor.center_x + 10:
                closest_cursor.center_x = self.all_cursor_list[closest_cursor_id + 1].x - 10
            if self.all_cursor_list[closest_cursor_id - 1].x > closest_cursor.center_x - 10:
                closest_cursor.center_x = self.all_cursor_list[closest_cursor_id - 1].x + 10
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
        # copy all data from original json
        # all we need to do is mark each timestep with 'isL'(isLabeled) value
        self.final_dict = json_data

    def lastButtoncallback(self, instance):
        print ('!!!! Move to %s Word !!!!' % instance.text)
        temp_idx = self.vocs_idx_counter - 1
        if temp_idx >= 0:
            self.vocs_idx_counter = temp_idx
            points, voc_length = self.read_voc_from_json(
                self.all_vocs_data.keys()[self.vocs_idx_counter])
            self.board.init_board(points, voc_length)
        else:
            # end
            pass

    def nextButtoncallback(self, instance):
        # save labeled data into 'final_dict' before move next word
        finished_voc = self.all_vocs_data.keys()[self.vocs_idx_counter]
        voc_dict = self.final_dict['data'][finished_voc]
        for _, timestep_dict in enumerate(voc_dict):
            timestep_dict['isL'] = False # default value with False: not labeled
        for labeled_idx in self.board.all_selected_points_idx_list:
            voc_dict[labeled_idx]['isL'] = True # selected timestep idx with True: labeled

        print ('!!!! Move to %s Word !!!!' % instance.text)
        temp_idx = self.vocs_idx_counter + 1
        if temp_idx < self.vocs_amount:
            self.vocs_idx_counter = temp_idx
            points, voc_length = self.read_voc_from_json(
                self.all_vocs_data.keys()[self.vocs_idx_counter])
            self.board.init_board(points, voc_length)
        else:
            # end
            result_filename = RESULT_FILE_PATH + ".json"
            with codecs.open(result_filename, 'w', 'utf-8') as out:
                json.dump(self.final_dict, out,
                          encoding="utf-8", ensure_ascii=False)
            print ("Saved to file path::", result_filename)

    def read_voc_from_json(self, voc):
        """
        params: voc: string, the traget word
        return:
        """
        print (voc)
        print (len(str(voc)))
        voc_length = len(str(voc))
        voc_pos_list = []
        voc_timestep_list = self.all_vocs_data[voc]
        for time_step_dict in voc_timestep_list:
            # i: timestep in one voc as dict format
            voc_pos_list.append(time_step_dict['pos'])
        scaled_pos = np.array(voc_pos_list)
        # normalization
        x_amax = np.amax(scaled_pos[:, 0])
        x_amin = np.amin(scaled_pos[:, 0])
        x_range = x_amax - x_amin
        x_scale = 1.0 / x_range
        scaled_pos[:, 0] = scaled_pos[:, 0] * x_scale * self.board.width
        scaled_pos[:, 1] = scaled_pos[:, 1] * x_scale * \
            self.board.height + self.board.y_offset

        return scaled_pos.flatten().tolist(), voc_length


class LabelingApp(App):
    """
    main app builder
    """

    def build(self):
        Label = AppEngine()
        return Label


if __name__ == '__main__':
    if not os.path.exists(NORMALIZED_DATA_DIR_PATH):
        os.makedirs(NORMALIZED_DATA_DIR_PATH)
    if not os.path.exists(LABELED_DATA_DIR_PATH):
        os.makedirs(LABELED_DATA_DIR_PATH)

    LabelingApp().run()
