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
from kivy.properties import NumericProperty, StringProperty, ObjectProperty, ListProperty
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.config import Config
from kivy.uix.textinput import TextInput
from kivy.graphics import Point, Color, Line
import kivy
kivy.require('1.9.0')
Config.set('graphics', 'width', '1800')
Config.set('graphics', 'height', '1200')


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NORMALIZED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'normalized_voc')
LABELED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'labeled_voc')
USER_FILE_NAME = 'User_'
TARGET_FILE_PATH = os.path.join(NORMALIZED_DATA_DIR_PATH, USER_FILE_NAME)
RESULT_FILE_PATH = os.path.join(LABELED_DATA_DIR_PATH, USER_FILE_NAME)


class UserIDTextInput(BoxLayout):
    button_text = StringProperty("")

    def __init__(self, *args, **kwargs):
        super(UserIDTextInput, self).__init__(*args, **kwargs)
        self.on_enter = kwargs['on_enter']


class ContentWithButton(BoxLayout):
    content_text = StringProperty("")
    button_text = StringProperty("")

    def __init__(self, *args, **kwargs):
        super(ContentWithButton, self).__init__(*args, **kwargs)
        self.content_text = kwargs['content_text']
        self.button_text = kwargs['button_text']

    def exit(self):
        App.get_running_app().stop()


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
        self.all_canvas_selected_line_list = []
        self.all_connectionist_color_list = []
        self.all_cursor_lines_list = []
        self.voc_length = None
        self.isInit = False

    def init_board(self, points, voc_length, restored_labeled_list=None):
        self.isInit = True
        self.canvas.clear()
        self.points = points
        self.voc_length = voc_length
        voc_lines = Line(points=self.points, width=4)
        self.canvas.add(Color(.6, .6, .6, .6))
        self.canvas.add(voc_lines)
        voc_points = Point(points=self.points, pointsize=5)
        self.canvas.add(Color(.6, .6, .6, 1))
        self.canvas.add(voc_points)

        if restored_labeled_list is not None:
            self.init_restored(restored_labeled_list)
        else:
            self.init_default()

    def init_restored(self, restored_labeled_list):
        # restore cursors and correspond selected points
        self.all_connectionist_color_list = []
        for i in range(self.voc_length - 1):
            # random colors
            # start from 7 for brighter color space
            r = random.randint(7, 11) / 10.0
            g = random.randint(0, 11) / 10.0
            b = random.randint(0, 11) / 10.0
            self.all_connectionist_color_list.append(Color(r, g, b))

        self.all_selected_points_idx_list = []
        self.all_selected_points_list = []
        self.all_cursor_list = []
        self.all_cursor_lines_list = []
        temp_flag = False
        temp_start_idx = None
        temp_end_idx = None
        counter = 0
        for i, value in enumerate(restored_labeled_list):
            # all_selected_points_idx_list
            if value is True:
                self.all_selected_points_idx_list.append(i)
            # all_selected_points_list
            if temp_flag is False:
                if value is True:
                    temp_start_idx = i
                    temp_flag = True
            elif temp_flag is True:
                if value is False:
                    temp_end_idx = i
                    temp_flag = False
                    temp_selected_points = self.points[temp_start_idx *
                                                       2: temp_end_idx * 2]
                    self.all_selected_points_list.append(temp_selected_points)
                    # start cursor
                    start_x = (temp_start_idx / len(restored_labeled_list))
                    temp_start_cursor = StartCursor(
                        pos=(start_x * self.width, SlideBar().y_offset), color=self.all_connectionist_color_list[counter].rgb)
                    self.add_widget(temp_start_cursor)
                    self.all_cursor_list.append(temp_start_cursor)
                    # end cursor
                    end_x = (temp_end_idx / len(restored_labeled_list))
                    temp_end_cursor = EndCursor(
                        pos=(end_x * self.width, SlideBar().y_offset), color=self.all_connectionist_color_list[counter].rgb)
                    self.add_widget(temp_end_cursor)
                    self.all_cursor_list.append(temp_end_cursor)
                    # line between cursors
                    temp_line_pos_list = [
                        start_x * self.width + 10, SlideBar().y_offset + 5, end_x * self.width, SlideBar().y_offset + 5]
                    temp_line = Line(points=temp_line_pos_list, width=5)
                    self.all_cursor_lines_list.append(temp_line)
                    self.canvas.add(
                        self.all_connectionist_color_list[counter])  # add Color
                    self.canvas.add(temp_line)  # add Line
                    # color index
                    counter += 1

        # visulize selected points on canvas and record pointers of those
        # canvas's Lines
        self.all_canvas_selected_line_list = []
        for i, selected_points in enumerate(self.all_selected_points_list):
            # temp_P = Point(points=selected_points, pointsize=5)
            temp_P = Line(points=selected_points, width=3)
            self.all_canvas_selected_line_list.append(temp_P)
            self.canvas.add(self.all_connectionist_color_list[i])  # add Color
            self.canvas.add(temp_P)  # add Line

    def init_default(self):
        # add default cursors and correspond selected points
        self.all_selected_points_list = []
        self.all_selected_points_idx_list = []
        self.all_cursor_list = []
        self.all_connectionist_color_list = []
        self.all_cursor_lines_list = []
        for i in range(self.voc_length - 1):
            # random colors
            # start from 7 for brighter color space
            r = random.randint(7, 11) / 10.0
            g = random.randint(0, 11) / 10.0
            b = random.randint(0, 11) / 10.0
            self.all_connectionist_color_list.append(Color(r, g, b))
            # start cursor
            start_x = ((i + 1) / self.voc_length - 1.0 / 4.0 / self.voc_length)
            temp_start_cursor = StartCursor(
                pos=(start_x * self.width, SlideBar().y_offset), color=[r, g, b])
            self.add_widget(temp_start_cursor)
            self.all_cursor_list.append(temp_start_cursor)
            # end cursor
            end_x = ((i + 1) / self.voc_length + 1.0 / 4.0 / self.voc_length)
            temp_end_cursor = EndCursor(
                pos=(end_x * self.width, SlideBar().y_offset), color=[r, g, b])
            self.add_widget(temp_end_cursor)
            self.all_cursor_list.append(temp_end_cursor)
            # line between cursors
            temp_line_pos_list = [
                start_x * self.width + 10, SlideBar().y_offset + 5, end_x * self.width, SlideBar().y_offset + 5]
            temp_line = Line(points=temp_line_pos_list, width=5)
            self.all_cursor_lines_list.append(temp_line)
            self.canvas.add(Color(r, g, b))  # add Color
            self.canvas.add(temp_line)  # add Line
            # selected points
            start_point_idx = int(len(self.points) / 2 * start_x)
            end_point_idx = int(len(self.points) / 2 * end_x)
            temp_selected_points = self.points[start_point_idx *
                                               2: end_point_idx * 2]
            self.all_selected_points_list.append(temp_selected_points)
            for selected_idx in range(start_point_idx, end_point_idx, 1):
                self.all_selected_points_idx_list.append(selected_idx)

        # record pointers pointing to canvas's Point in
        # 'self.all_canvas_point_list'
        self.all_canvas_selected_line_list = []
        for i, selected_points in enumerate(self.all_selected_points_list):
            # temp_P = Point(points=selected_points, pointsize=5)
            temp_P = Line(points=selected_points, width=3)
            self.all_canvas_selected_line_list.append(temp_P)
            self.canvas.add(self.all_connectionist_color_list[i])
            self.canvas.add(temp_P)

    def on_touch_move(self, touch):
        super(DrawingBoard, self).on_touch_down(touch)
        if self.isInit:
            self.touch_action(touch)

    def on_touch_down(self, touch):
        super(DrawingBoard, self).on_touch_down(touch)
        if self.isInit:
            self.touch_action(touch)

    def get_cursor_matched_point_idx(self, cursor):
        normalized_x = cursor.center_x / self.width
        pointsLength = len(self.points) / 2
        return int(normalized_x * pointsLength)

    def update_selected_points(self):
        self.all_selected_points_idx_list = []  # clear
        for i, canvas_selected_line in enumerate(self.all_canvas_selected_line_list):
            startPtIdx = self.get_cursor_matched_point_idx(
                self.all_cursor_list[i * 2])
            endPtIdx = self.get_cursor_matched_point_idx(
                self.all_cursor_list[i * 2 + 1])
            canvas_selected_line.points = self.points[startPtIdx *
                                                      2: endPtIdx * 2]
            for selected_idx in range(startPtIdx, endPtIdx, 1):
                self.all_selected_points_idx_list.append(selected_idx)

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
            temp = self.all_cursor_lines_list[int(
                closest_cursor_id / 2)].points
            if closest_cursor_id % 2 is 0:
                temp[0] = touch.x + 5  # start cursor
                self.all_cursor_lines_list[int(
                    closest_cursor_id / 2)].points = temp
            elif closest_cursor_id % 2 is 1:
                temp[2] = touch.x - 5  # end cursor
                self.all_cursor_lines_list[int(
                    closest_cursor_id / 2)].points = temp

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
    saveButton = ObjectProperty(None)
    nextButton = ObjectProperty(None)
    board = ObjectProperty(None)
    word = StringProperty("None")
    word_idx = StringProperty("None")

    def __init__(self, *args, **kwargs):
        super(AppEngine, self).__init__(*args, **kwargs)

        self.all_vocs_data = None
        self.vocs_amount = None
        self.final_dict = None
        self.vocs_idx_counter = None
        self.user_id = 0

        # create content and add to the popup
        self.create_userid_textinput(title="User ID")

    def create_userid_textinput(self, title):
        content = UserIDTextInput(on_enter=self.on_enter)
        self.popupUserID = Popup(title=title,
                                 title_size='48sp',
                                 title_align='center',
                                 title_color=[1, 1, 1, 1],
                                 content=content,
                                 auto_dismiss=False,
                                 size_hint=(.15, .2))
        # open the popup
        self.popupUserID.open()

    def on_enter(self, user_id):
        self.popupUserID.dismiss()
        self.init(user_id)

    def init(self, user_id):
        self.lastButton.bind(on_press=self.lastButtonCallback)
        self.saveButton.bind(on_press=self.saveButtonCallback)
        self.nextButton.bind(on_press=self.nextButtonCallback)
        self.user_id = user_id

        filename = TARGET_FILE_PATH + user_id + ".json"
        if not os.path.isfile(filename):
            self.create_userid_textinput(title="Try Again")
            return
        else:
            with codecs.open(filename, 'r', 'utf-8-sig') as f:
                json_data = json.load(f)
            self.vocs_idx_counter = -1
            self.all_vocs_data = json_data['data']
            self.vocs_amount = len(json_data['data'])
            # copy all data from original json
            # all we need to do is mark each timestep with 'isL'(isLabeled)
            # value
            self.final_dict = json_data

            self.move_next_voc()

    def lastButtonCallback(self, instance):
        # save labeled data into 'final_dict' before move next/ last word
        self.update_final_dict()

        # move to last word
        print ('!!!! Move to <Last> Word !!!!')
        self.move_last_voc()

    def saveButtonCallback(self, instance):
        result_filename = RESULT_FILE_PATH + self.user_id + ".json"
        with codecs.open(result_filename, 'w', 'utf-8') as out:
            json.dump(self.final_dict, out,
                      encoding="utf-8", ensure_ascii=False)
        print ("Saved to file path::", result_filename)

        # create content and add to the popup
        content = Label(
            text="Labeled data have saved to following path:\n" + result_filename,
            text_size=(self.width, None),
            halign='center',
            font_size='32sp')
        popup = Popup(title="Successully Saved",
                      title_size='56sp',
                      title_align='center',
                      title_color=[1, 1, 1, 1],
                      content=content,
                      auto_dismiss=True,
                      size_hint=(.65, .25))
        # open the popup
        popup.open()

    def nextButtonCallback(self, instance):
        # save labeled data into 'final_dict' before move next/ last word
        self.update_final_dict()

        # move to next word
        print ('!!!! Move to <Next> Word !!!!')
        self.move_next_voc()

    def move_last_voc(self):
        temp_idx = self.vocs_idx_counter - 1
        if self.is_idx_valid(temp_idx):
            self.vocs_idx_counter = temp_idx

            # restore data if the word had been label
            restored_labeled_list = self.restore_labeled_index()

            points, voc_length = self.read_voc_from_json(
                self.all_vocs_data.keys()[self.vocs_idx_counter])
            self.board.init_board(points, voc_length, restored_labeled_list)
        else:
            # end
            pass

    def move_next_voc(self):
        temp_idx = self.vocs_idx_counter + 1
        if self.is_idx_valid(temp_idx):
            self.vocs_idx_counter = temp_idx

            # restore data if the word had been label
            restored_labeled_list = self.restore_labeled_index()

            points, voc_length = self.read_voc_from_json(
                self.all_vocs_data.keys()[self.vocs_idx_counter])
            self.board.init_board(points, voc_length, restored_labeled_list)
        else:
            # end
            result_filename = RESULT_FILE_PATH + self.user_id + ".json"
            with codecs.open(result_filename, 'w', 'utf-8') as out:
                json.dump(self.final_dict, out,
                          encoding="utf-8", ensure_ascii=False)
            print ("Saved to file path::", result_filename)

            # create content and add to the popup
            content = ContentWithButton(
                content_text="Many Thanks!\nLabeled data have saved to following path:\n" + result_filename, button_text='Close App')
            popup = Popup(title="!!Congrat!!",
                          title_size='56sp',
                          title_align='center',
                          title_color=[1, 1, 1, 1],
                          content=content,
                          auto_dismiss=False,
                          size_hint=(.4, .4))
            # open the popup
            popup.open()

    def is_idx_valid(self, index):
        return index >= 0 and index < self.vocs_amount

    def update_final_dict(self):
        if self.is_idx_valid(self.vocs_idx_counter):
            finished_voc = self.final_dict['data'].keys()[
                self.vocs_idx_counter]
            voc_dict = self.final_dict['data'][finished_voc]
            for _, timestep_dict in enumerate(voc_dict):
                # default value with False: not labeled
                timestep_dict['isL'] = False
            for labeled_idx in self.board.all_selected_points_idx_list:
                # selected timestep idx with True: labeled
                voc_dict[labeled_idx]['isL'] = True

    def restore_labeled_index(self):
        next_word = self.final_dict['data'].keys()[
            self.vocs_idx_counter]
        restored_labeled_list = []
        if 'isL' in self.final_dict['data'][next_word][0]:
            for timestep_dict in self.final_dict['data'][next_word]:
                restored_labeled_list.append(timestep_dict['isL'])
            return restored_labeled_list
        else:
            return None

    def read_voc_from_json(self, voc):
        """
        params: voc: string, the traget word
        return:
        """
        print ("voc::", voc)
        print ("length::", len(str(voc)))
        self.word = str(voc)
        self.word_idx = str(self.vocs_idx_counter)
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
        LabelApp = AppEngine()
        return LabelApp


if __name__ == '__main__':
    if not os.path.exists(NORMALIZED_DATA_DIR_PATH):
        os.makedirs(NORMALIZED_DATA_DIR_PATH)
    if not os.path.exists(LABELED_DATA_DIR_PATH):
        os.makedirs(LABELED_DATA_DIR_PATH)

    LabelingApp().run()
