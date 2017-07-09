from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
import math

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "../data/")
LABEL_DATA_PATH = os.path.join(DATA_PATH, "ascii/")
STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

ESCAPE_CHAR = '~!@#$%^&*()_+{}:"<>?`-=[];\',./|\n'


def find_textline_by_id(filename):
    """
    Inputs:
        filename: string, textline prefix, eg: 'a01-020w-01'
    Return:
        label: string, label of one whole textline, eg: 'No secret talks # - Macleod.'
    """
    dir_name_L1 = filename[:3]  # eg: 'a01'
    dir_name_L2 = filename[:7]  # eg: 'a01-020'
    file_name = filename[:-3] + ".txt"  # eg: 'a01-020w.txt' or 'a01-020.txt'
    line_id = int(filename[-2:])  # eg: 1
    filepath = os.path.join(
        LABEL_DATA_PATH, dir_name_L1, dir_name_L2, file_name)
    line_counter = -2  # because line start after 2 new lines from "CSR:\n"

    label = []
    flag = False
    for line in open(filepath, 'r'):
        if line.startswith('CSR'):
            flag = True
        if flag:
            line_counter += 1
        if line_counter == line_id:
            for char in ESCAPE_CHAR:
                line = line.replace(char, '')
            label = line
            break
    return label


def main():
    # parse STROKES (.xml)
    text_line_data_all = []
    label_text_line_all = []
    for path_1, _, files in os.walk(STROKES_DATA_PATH):
        files = sorted(files)
        for file_name in files:  # TextLine files
            ############# label data #############
            # split our .xml (eg: a01-020w-01.xml -> a01-020w-01)
            text_line_id = file_name[:-4]
            label_text_line = find_textline_by_id(text_line_id)
            if len(label_text_line) != 0:  # prevent missing data in ascii(label data)
                label_text_line_all.append(label_text_line)
                ############# trajectory data #############
                text_line_path = os.path.join(path_1, file_name)
                e_tree = ET.parse(text_line_path).getroot()
                x_list = []
                y_list = []
                time_stamp = []
                first_time = 0.0
                for atype in e_tree.findall('StrokeSet/Stroke/Point'):
                    x_list.append(int(atype.get('x')))
                    y_list.append(int(atype.get('y')))
                    if len(time_stamp) == 0:
                        first_time = float(atype.get('time'))
                        time_stamp.append(0.0)
                    else:
                        time_stamp.append(
                            float(atype.get('time')) - first_time)
                    # print("x:", atype.get('x'), "y:", atype.get('y'), "time:", float(atype.get('time')) - first_time)
                    # input()
                ####curvature and speed #######################################
                # x y coordinate
                x_list = np.asarray(x_list, dtype=np.float32)
                y_list = np.asarray(y_list, dtype=np.float32)
                x_cor = np.copy(x_list)
                y_cor = np.copy(y_list)
                cor_dial = e_tree.findall(
                    'WhiteboardDescription/DiagonallyOppositeCoords')[0]
                x_max = int(cor_dial.get('x'))
                y_max = int(cor_dial.get('y'))
                x_min = min(x_list)
                y_min = min(y_list)
                scale = 1.0 / (y_max - y_min)
                # normalize x_cor , y_cor
                x_cor = (x_cor - x_min) * scale
                y_cor = (y_cor - y_min) * scale

                sin_list = []
                cos_list = []
                x_sp_list = []
                y_sp_list = []
                pen_up_list = []
                writing_sin = []
                writing_cos = []
                
                for stroke in e_tree.findall('StrokeSet/Stroke'):
                    x_point, y_point, time_list = [], [], []
                    for point in stroke.findall('Point'):
                        x_point.append(int(point.get('x')))
                        y_point.append(int(point.get('y')))
                        if len(time_list) == 0:
                            first_time = float(point.get('time'))
                            time_list.append(0.0)
                        else:
                            time_list.append(
                                float(point.get('time')) - first_time)
                    # calculate cos and sin
                    x_point[:] = [ (point - x_min) * scale for point in x_point]
                    y_point[:] = [ (point - y_min) * scale for point in y_point]

                    angle_stroke = []
                    if len(x_point) < 3:
                        # print("Oh no",len(x_point))
                        for _ in range(len(x_point)):
                            sin_list += [0]
                            cos_list += [1]
                    else:
                        for idx in range(1, len(x_point) - 1):
                            x_prev = x_point[idx - 1]
                            y_prev = y_point[idx - 1]
                            x_next = x_point[idx + 1]
                            y_next = y_point[idx + 1]
                            x_now = x_point[idx]
                            y_now = y_point[idx]
                            p0 = [x_prev, y_prev]
                            p1 = [x_now, y_now]
                            p2 = [x_next, y_next]
                            v0 = np.array(p0) - np.array(p1)
                            v1 = np.array(p2) - np.array(p1)
                            angle = np.math.atan2(
                                np.linalg.det([v0, v1]), np.dot(v0, v1))
                            angle_stroke.append(angle)
                        new_angle_stroke = [0] + angle_stroke + [0]
                        sin_stroke = np.sin(new_angle_stroke).tolist()
                        cos_stroke = np.cos(new_angle_stroke).tolist()
                        sin_list += sin_stroke
                        cos_list += cos_stroke
                    # calculate speed
                    if len(x_point) < 2:
                        for _ in range(len(x_point)):
                            x_sp_list += [0]
                            y_sp_list += [0]
                            
                        if len(x_point) < 1:
                            print("Meet 0")
                            exit()
                        x_sp = [0]
                        y_sp = [0]

                    else:
                        time_list = np.asarray(time_list, dtype=np.float32)
                        time_list_moved = np.array(time_list)[1:]
                        time_diff = np.subtract(
                            time_list_moved, time_list[:-1])
                        for idx, v in enumerate(time_diff):
                            if v == 0:
                                time_diff[idx] = 0.001
                        x_point_moved = np.array(x_point)[1:]
                        y_point_moved = np.array(y_point)[1:]
                        x_diff = np.subtract(x_point_moved, x_point[:-1])
                        y_diff = np.subtract(y_point_moved, y_point[:-1])
                        x_sp = np.divide(x_diff, time_diff).tolist()
                        y_sp = np.divide(y_diff, time_diff).tolist()
                        x_sp = [0] + x_sp
                        y_sp = [0] + y_sp
                        x_sp_list += x_sp
                        y_sp_list += y_sp
                    # pen up and down
                    pen_up = [1] * (len(x_point) - 1) + [0]
                    pen_up_list += pen_up
                    # writing direction
                    w_sin_stroke = []
                    w_cos_stroke = []
                    for idx, x_v in enumerate(x_sp):
                        y_v = y_sp[idx]
                        slope = np.sqrt(x_v * x_v + y_v * y_v)
                        if slope != 0:
                            w_sin_stroke.append(y_v / slope)
                            w_cos_stroke.append(x_v / slope)
                        else:
                            w_sin_stroke.append(0)
                            w_cos_stroke.append(1)
                    writing_sin += w_sin_stroke
                    writing_cos += w_cos_stroke

                ####curvature done####################
                time_stamp = np.asarray(time_stamp, dtype=np.float32)
                sin_list = np.asarray(sin_list, dtype=np.float32)
                cos_list = np.asarray(cos_list, dtype=np.float32)
                x_sp_list = np.asarray(x_sp_list, dtype=np.float32)
                y_sp_list = np.asarray(y_sp_list, dtype=np.float32)
                pen_up_list = np.asarray(pen_up_list, dtype=np.float32)
                writing_cos = np.asarray(writing_cos, dtype=np.float32)
                writing_sin = np.asarray(writing_sin, dtype=np.float32)
                # x y coordinate

                # time_list = np.asarray(time_list, dtype=np.float32)
                # # time 1st order
                # time_list_moved = np.array(time_list)
                # time_list_moved = time_list_moved[1:]
                # time_list_new = np.subtract(
                #     time_list_moved, time_list[:-1])
                # for idx, v in enumerate(time_list_new):
                #     if v == 0:
                #         time_list_new[idx] = 0.001  # prevent divide by 0
                # x_list_moved = np.array(x_list)
                # x_list_moved = x_list_moved[1:]
                # # x_list throw away the last element
                # x_list_new = np.subtract(x_list_moved, x_list[:-1])
                # # x_list_new = np.divide(x_list_new, time_list_new)
                # # y 1st order
                # y_list_moved = np.array(y_list)
                # y_list_moved = y_list_moved[1:]
                # y_list_new = np.subtract(y_list_moved, y_list[:-1])
                # y_list_new = np.divide(y_list_new, time_list_new)
                # stack x', y', time
                # print(x_sp_list.shape)
                # print(y_sp_list.shape)
                # print(x_cor.shape)
                # print(y_cor.shape)
                # print(sin_list.shape)
                # print(cos_list.shape)
                # print(time_stamp.shape)
                # print(writing_sin.shape)
                # print(writing_cos.shape)
                # print(pen_up_list.shape)
                text_line_data = np.stack(
                    (x_sp_list, y_sp_list, x_cor, y_cor, sin_list, cos_list, writing_sin, writing_cos, pen_up_list, time_stamp), axis=1)
                # (x_cor, y_cor, sin_list, cos_list, time_stamp), axis=1)
                # text_line_data = np.stack(
                #     (x_list, y_list, time_list), axis=1)
                temp_length = text_line_data.shape[0]
                # subsampling
                # text_line_data = text_line_data[[
                #     i % 3 == 0 for i in range(temp_length)]]
                text_line_data_all.append(text_line_data)
        print("Finished a file ", files)
        # print(text_line_data)
        # print(text_line_data.shape)
        # print(text_line_path)
        # print(np.array(text_line_data_all).shape)
        # input()

    text_line_data_all = np.array(text_line_data_all)
    label_text_line_all = np.array(label_text_line_all)
    print(text_line_data_all.shape)
    print(label_text_line_all.shape)
    # save as .npy
    np.save("data", text_line_data_all)
    np.save("label", label_text_line_all)
    print("Successfully saved!")


def visual():
    # parse STROKES (.xml)
    x_all = []
    y_all = []
    for path_1, _, files in os.walk(STROKES_DATA_PATH):
        files = sorted(files)
        for file_name in files:  # TextLine files
            ############# label data #############
            # split our .xml (eg: a01-020w-01.xml -> a01-020w-01)
            text_line_id = file_name[:-4]
            label_text_line = find_textline_by_id(text_line_id)
            if len(label_text_line) != 0:  # prevent missing data in ascii(label data)
                # label_text_line_all.append(label_text_line)
                ############# trajectory data #############
                text_line_path = os.path.join(path_1, file_name)
                e_tree = ET.parse(text_line_path).getroot()
                x_list = []
                y_list = []
                time_stamp = []
                first_time = 0.0
                for atype in e_tree.findall('StrokeSet/Stroke/Point'):
                    x_list.append(int(atype.get('x')))
                    y_list.append(int(atype.get('y')))
                    if len(time_stamp) == 0:
                        first_time = float(atype.get('time'))
                        time_stamp.append(0.0)
                    else:
                        time_stamp.append(
                            float(atype.get('time')) - first_time)
                    # print("x:", atype.get('x'), "y:", atype.get('y'), "time:", float(atype.get('time')) - first_time)
                    # input()
                ####curvature and speed #######################################
                # x y coordinate
                x_list = np.asarray(x_list, dtype=np.float32)
                y_list = np.asarray(y_list, dtype=np.float32)
                x_cor = np.copy(x_list)
                y_cor = np.copy(y_list)
                cor_dial = e_tree.findall(
                    'WhiteboardDescription/DiagonallyOppositeCoords')[0]
                x_max = int(cor_dial.get('x'))
                y_max = int(cor_dial.get('y'))
                x_min = min(x_list)
                y_min = min(y_list)
                scale = 1.0 / (y_max - y_min)
                # normalize x_cor , y_cor
                x_cor = (x_cor - x_min) * scale
                y_cor = (y_cor - y_min) * scale
                x_cor = x_cor.tolist()
                y_cor = y_cor.tolist()
                x_all += x_cor
                y_all += y_cor
    see_x = np.array(x_all)
    see_y = np.array(y_all)
    sns.distplot(see_y)
    plt.show()



if __name__ == "__main__":
    main()
    #visual()
