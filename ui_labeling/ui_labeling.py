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
kivy.require('1.8.0')
Config.set('graphics', 'width', '1800')
Config.set('graphics', 'height', '1000')
Config.set('input', 'mouse', 'mouse, disable_multitouch')
from preprocessing.sphere_fitting import fit_sphere


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.join(DIR_PATH, 'preprocessing/voc')
NORMALIZED_DATA_DIR_PATH = os.path.join(
    DIR_PATH, 'preprocessing/normalized_voc')
LABELED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'labeled_voc')


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
        self.high_contrast_colors_index = 0
        self.high_contrast_colors = [Color(1, .3, .3),
                                     Color(.3, 1, .3),
                                     Color(.3, .3, 1),
                                     Color(1, .1, 1),
                                     Color(1, 1, .3)]
        self.closest_cursor = None
        self.closest_cursor_id = None

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

        self.update_selected_points()

    def get_color(self):
        return_color = self.high_contrast_colors[self.high_contrast_colors_index]
        self.high_contrast_colors_index += 1
        if self.high_contrast_colors_index == len(self.high_contrast_colors):
            self.high_contrast_colors_index = 0
        return return_color

    def init_restored(self, restored_labeled_list):
        # restore cursors and correspond selected points
        self.all_connectionist_color_list = []
        for i in range(self.voc_length):
            self.all_connectionist_color_list.append(self.get_color())

        self.all_selected_points_list = []
        self.all_cursor_list = []
        self.all_cursor_lines_list = []
        temp_flag = False
        temp_start_idx = None
        temp_end_idx = None
        counter = 0
        for i, value in enumerate(restored_labeled_list):
            # all_selected_points_list
            if temp_flag is False:
                if value is True:
                    temp_start_idx = i
                    temp_flag = True
            elif temp_flag is True:
                if value is False or i == (len(restored_labeled_list) - 1):
                    # selected points
                    temp_end_idx = i
                    temp_flag = False
                    temp_selected_points = self.points[temp_start_idx *
                                                       2: temp_end_idx * 2]
                    self.all_selected_points_list.append(temp_selected_points)
                    # start cursor
                    start_x = (float(temp_start_idx) /
                               len(restored_labeled_list))
                    temp_start_cursor = StartCursor(
                        pos=(start_x * self.width - 5, SlideBar().y_offset), color=self.all_connectionist_color_list[counter].rgb)
                    self.add_widget(temp_start_cursor)
                    self.all_cursor_list.append(temp_start_cursor)
                    # end cursor
                    end_x = (float(temp_end_idx) / len(restored_labeled_list))
                    temp_end_cursor = EndCursor(
                        pos=(end_x * self.width - 5, SlideBar().y_offset), color=self.all_connectionist_color_list[counter].rgb)
                    self.add_widget(temp_end_cursor)
                    self.all_cursor_list.append(temp_end_cursor)
                    # line between cursors
                    temp_line_pos_list = [
                        start_x * self.width, SlideBar().y_offset + 5, end_x * self.width, SlideBar().y_offset + 5]
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
        self.all_cursor_list = []
        self.all_connectionist_color_list = []
        self.all_cursor_lines_list = []
        # cursor_range = 1 / (2 * voc_length - 1)
        # cursor-i: range(2 * i * cursor_range, (2 * i + 1) * cursor_range)
        cursor_range = 1.0 / (2 * self.voc_length - 1)
        for i in range(self.voc_length):
            self.all_connectionist_color_list.append(self.get_color())
            # start cursor
            start_x = 2 * i * cursor_range
            temp_start_cursor = StartCursor(
                pos=(start_x * self.width - 5, SlideBar().y_offset), color=self.all_connectionist_color_list[i].rgb)
            self.add_widget(temp_start_cursor)
            self.all_cursor_list.append(temp_start_cursor)
            # end cursor
            end_x = (2 * i + 1) * cursor_range
            temp_end_cursor = EndCursor(
                pos=(end_x * self.width - 5, SlideBar().y_offset), color=self.all_connectionist_color_list[i].rgb)
            self.add_widget(temp_end_cursor)
            self.all_cursor_list.append(temp_end_cursor)
            # line between cursors
            temp_line_pos_list = [
                start_x * self.width, SlideBar().y_offset + 5, end_x * self.width, SlideBar().y_offset + 5]
            temp_line = Line(points=temp_line_pos_list, width=5)
            self.all_cursor_lines_list.append(temp_line)
            self.canvas.add(self.all_connectionist_color_list[i])  # add Color
            self.canvas.add(temp_line)  # add Line
            # selected points
            start_point_idx = int(len(self.points) / 2 * start_x)
            end_point_idx = int(len(self.points) / 2 * end_x)
            temp_selected_points = self.points[start_point_idx *
                                               2: end_point_idx * 2]
            self.all_selected_points_list.append(temp_selected_points)

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
            self.touch_action(touch, mode='on_touch_move')

    def on_touch_down(self, touch):
        super(DrawingBoard, self).on_touch_down(touch)
        if self.isInit:
            self.touch_action(touch, mode='on_touch_down')

    def get_cursor_matched_point_idx(self, cursor):
        if cursor.center_x > self.width - 5:
            cursor.center_x = self.width
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

    def touch_action(self, touch, mode):
        if mode == 'on_touch_down':
            # select the cloest cursor to modify its center_x
            self.closest_cursor = None
            self.closest_cursor_id = -1
            if touch.x > self.all_cursor_list[-1].center_x and touch.x < self.width:
                self.closest_cursor = self.all_cursor_list[-1]
                self.closest_cursor_id = len(self.all_cursor_list) - 1
            elif touch.x < self.all_cursor_list[0].center_x and touch.x > 0:
                self.closest_cursor = self.all_cursor_list[0]
                self.closest_cursor_id = 0
            else:
                for i in range(len(self.all_cursor_list) - 1):
                    if touch.x > self.all_cursor_list[i].center_x and touch.x < self.all_cursor_list[i + 1].center_x:
                        # select closest start cursor
                        if touch.button == 'left':
                            if i % 2 == 0:
                                self.closest_cursor = self.all_cursor_list[i]
                                self.closest_cursor_id = i
                            else:
                                self.closest_cursor = self.all_cursor_list[i + 1]
                                self.closest_cursor_id = i + 1
                        # select closest start cursor
                        elif touch.button == 'right':
                            if i % 2 == 1:
                                self.closest_cursor = self.all_cursor_list[i]
                                self.closest_cursor_id = i
                            else:
                                self.closest_cursor = self.all_cursor_list[i + 1]
                                self.closest_cursor_id = i + 1

        if self.closest_cursor is not None:
            if touch.x < self.width and touch.y < self.height:
                self.closest_cursor.center_x = touch.x

        # make sure the cursor's center_x value won't exceed neighbors
        # '+-5' is for foolproof
        if self.closest_cursor is not None:
            if self.closest_cursor_id > 0 and self.closest_cursor_id < len(self.all_cursor_list) - 1:
                for any_i, any_cursor in enumerate(self.all_cursor_list):
                    if any_i > self.closest_cursor_id and any_cursor.center_x < self.closest_cursor.center_x + 5:
                        any_cursor.center_x = self.closest_cursor.center_x + 5 * (any_i-self.closest_cursor_id)
                    if any_i < self.closest_cursor_id and any_cursor.center_x > self.closest_cursor.center_x - 5:
                        any_cursor.center_x = self.closest_cursor.center_x - 5 * (self.closest_cursor_id - any_i)

        # update all lins bwtween cursors
        for any_i, _ in enumerate(self.all_cursor_list):
            if any_i % 2 == 0:
                temp = self.all_cursor_lines_list[int(any_i/2)].points
                temp[0] = self.all_cursor_list[any_i].center_x
                temp[2] = self.all_cursor_list[any_i + 1].center_x
                self.all_cursor_lines_list[int(any_i/2)].points = temp
                

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
        self.vocs_amount = None
        self.result_dict = None
        self.user_id = None
        self.vocs_idx_counter = None
        self.normalized_dirpath = None
        self.result_dirpath = None
        self.words_list = None

        # create content and add to the popup
        self.create_userid_textinput(title="User ID")

    def create_userid_textinput(self, title):
        content = UserIDTextInput(on_enter=self.on_enter)
        self.popupUserID = Popup(title=title,
                                 title_size='40sp',
                                 title_align='center',
                                 title_color=[1, 1, 1, 1],
                                 content=content,
                                 auto_dismiss=False,
                                 size_hint=(.15, .25))
        # open the popup
        self.popupUserID.open()

    def on_enter(self, user_id):
        self.popupUserID.dismiss()
        self.init(user_id)

    def init(self, user_id):
        self.lastButton.bind(on_release=self.lastButtonCallback)
        self.nextButton.bind(on_release=self.nextButtonCallback)
        self.user_id = user_id
        self.vocs_idx_counter = -1

        if self.user_id != '':
            self.normalized_dirpath = os.path.join(
                NORMALIZED_DATA_DIR_PATH, user_id)
            self.result_dirpath = os.path.join(
                LABELED_DATA_DIR_PATH, self.user_id)
            data_dirpath = os.path.join(DATA_DIR_PATH, self.user_id)

            if not os.path.exists(self.normalized_dirpath):
                os.makedirs(self.normalized_dirpath)
            if not os.path.exists(self.result_dirpath):
                os.makedirs(self.result_dirpath)
            if not os.path.exists(data_dirpath):
                os.makedirs(data_dirpath)

            if os.listdir(self.result_dirpath):  # first, check if labeled
                pass
            elif os.listdir(self.normalized_dirpath):  # second, check if had normalized
                pass
            elif os.listdir(data_dirpath):  # check, origin voc if exist
                # forth, create it and check if successfull
                if fit_sphere(data_dirpath, self.normalized_dirpath):
                    pass
            else:
                self.create_userid_textinput(title="Try Again")
                return
        else:
            self.create_userid_textinput(title="Try Again")
            return

        for _, _, files in os.walk(self.normalized_dirpath):
            self.words_list = sorted(files)
            self.vocs_amount = len(files)

        self.move_next_voc()

    def lastButtonCallback(self, instance):
        self.update_final_dict()
        # move to last word
        print ('!!!! Move to <Last> Word !!!!')
        self.move_last_voc()

    def nextButtonCallback(self, instance):
        self.update_final_dict()
        # move to next word
        print ('!!!! Move to <Next> Word !!!!')
        self.move_next_voc()

    def get_current_target_filename(self):
        if self.is_idx_valid(self.vocs_idx_counter):
            filename = self.words_list[self.vocs_idx_counter]
            result_filename = os.path.join(self.result_dirpath, filename)
            normalized_filename = os.path.join(
                self.normalized_dirpath, filename)
            target_filename = None
            if os.path.isfile(result_filename):  # first, check if labeled
                target_filename = result_filename
            # second, check if had normalized
            elif os.path.isfile(normalized_filename):
                target_filename = normalized_filename
            return target_filename
        else:
            return None

    def update_final_dict(self):
        target_filename = self.get_current_target_filename()
        if target_filename is not None:
            with codecs.open(target_filename, 'r', 'utf-8') as f:
                raw_data = json.load(f)
            self.result_dict = raw_data
            for _, timestep_dict in enumerate(self.result_dict['data']):
                # default value with False: not labeled
                timestep_dict['isL'] = False
            for labeled_idx in self.board.all_selected_points_idx_list:
                # selected timestep idx with True: labeled
                self.result_dict['data'][labeled_idx]['isL'] = True
        else:
            self.result_dict = None
            return None

    def move_last_voc(self):
        if self.is_idx_valid(self.vocs_idx_counter):
            # save immediately before move next/ last word
            result_filename = os.path.join(self.result_dirpath, str(
                self.words_list[self.vocs_idx_counter]))
            with codecs.open(result_filename, 'w', 'utf-8') as out:
                json.dump(self.result_dict, out,
                          encoding="utf-8", ensure_ascii=False)
            print ("Saved to file path::", result_filename)

        if self.is_idx_valid(self.vocs_idx_counter - 1):
            self.vocs_idx_counter = self.vocs_idx_counter - 1

            # restore data if the word had been label
            restored_labeled_list = self.restore_labeled_index()

            points, voc_length = self.read_voc_from_json(
                self.words_list[self.vocs_idx_counter])
            self.board.init_board(points, voc_length, restored_labeled_list)
        else:
            # end
            pass

    def move_next_voc(self):
        if self.is_idx_valid(self.vocs_idx_counter):
            # save immediately before move next/ last word
            result_filename = os.path.join(self.result_dirpath, str(
                self.words_list[self.vocs_idx_counter]))
            with codecs.open(result_filename, 'w', 'utf-8') as out:
                json.dump(self.result_dict, out,
                          encoding="utf-8", ensure_ascii=False)

        if self.is_idx_valid(self.vocs_idx_counter + 1):
            self.vocs_idx_counter = self.vocs_idx_counter + 1

            # restore data if the word had been label
            restored_labeled_list = self.restore_labeled_index()

            points, voc_length = self.read_voc_from_json(
                self.words_list[self.vocs_idx_counter])
            self.board.init_board(points, voc_length, restored_labeled_list)
        else:
            # end
            # create content and add to the popup
            content = ContentWithButton(
                content_text="Many Thanks!\nAll Labeled data have saved to following path:\n" + self.result_dirpath, button_text='Close App')
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

    def restore_labeled_index(self):
        target_filename = self.get_current_target_filename()

        if target_filename is not None:
            with codecs.open(target_filename, 'r', 'utf-8') as f:
                raw_data = json.load(f)
            restored_labeled_list = []
            if 'isL' in raw_data['data'][0]:  # check key 'isL' exist
                for timestep_dict in raw_data['data']:
                    restored_labeled_list.append(timestep_dict['isL'])
                return restored_labeled_list  # init board by init_restore
            else:
                return None  # init board init_default
        else:
            return None  # init board init_default

    def read_voc_from_json(self, voc_filename):
        """
        params: voc_filename: string, the filename of traget word
        return:
        """
        target_filename = self.get_current_target_filename()

        if target_filename is not None:
            with codecs.open(target_filename, 'r', 'utf-8') as f:
                raw_data = json.load(f)
            voc = raw_data['word']
            print ("voc::", voc)
            self.word = str(voc)
            self.word_idx = str(self.vocs_idx_counter)
            voc_length = len(str(voc))
            voc_pos_list = []
            for time_step_dict in raw_data['data']:
                # i: timestep in one voc as dict format
                voc_pos_list.append(time_step_dict['pos'])
            scaled_pos = np.array(voc_pos_list)
            # normalization
            x_amax = np.amax(scaled_pos[:, 0])
            x_amin = np.amin(scaled_pos[:, 0])
            y_amax = np.amax(scaled_pos[:, 1])
            y_amin = np.amin(scaled_pos[:, 1])
            x_range = x_amax - x_amin
            y_range = y_amax - y_amin
            if x_range > y_range:
                x_scale = 1.0 / x_range
                scaled_pos[:, 0] = scaled_pos[:, 0] * x_scale * \
                    self.board.width * 0.9 + self.board.width * 0.05
                scaled_pos[:, 1] = scaled_pos[:, 1] * x_scale * \
                    self.board.height + self.board.height * 0.1
            else:
                y_scale = 1.0 / y_range
                scaled_pos[:, 0] = scaled_pos[:, 0] * y_scale * \
                    self.board.width * 0.9 + self.board.width * 0.05
                scaled_pos[:, 1] = scaled_pos[:, 1] * y_scale * \
                    self.board.height * 0.8 + self.board.height * 0.1

            return scaled_pos.flatten().tolist(), voc_length
        else:
            return None


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
