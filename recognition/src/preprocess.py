from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree as ET
import re
import numpy as np

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
                time_list = []
                first_time = 0.0
                for atype in e_tree.findall('StrokeSet/Stroke/Point'):
                    x_list.append(atype.get('x'))
                    y_list.append(atype.get('y'))
                    if len(time_list) == 0:
                        first_time = float(atype.get('time'))
                        time_list.append(0.0)
                    else:
                        time_list.append(
                            float(atype.get('time')) - first_time)
                    # print("x:", atype.get('x'), "y:", atype.get('y'), "time:", float(atype.get('time')) - first_time)
                    # input()
                x_list = np.asarray(x_list, dtype=np.float32)
                y_list = np.asarray(y_list, dtype=np.float32)
                time_list = np.asarray(time_list, dtype=np.float32)
                # time 1st order
                time_list_moved = np.array(time_list)
                time_list_moved = time_list_moved[1:]
                time_list_new = np.subtract(
                    time_list_moved, time_list[:-1])
                for idx, v in enumerate(time_list_new):
                    if v == 0:
                        time_list_new[idx] = 0.001  # prevent divide by 0
                x_list_moved = np.array(x_list)
                x_list_moved = x_list_moved[1:]
                # x_list throw away the last element
                x_list_new = np.subtract(x_list_moved, x_list[:-1])
                # x_list_new = np.divide(x_list_new, time_list_new)
                # y 1st order
                y_list_moved = np.array(y_list)
                y_list_moved = y_list_moved[1:]
                y_list_new = np.subtract(y_list_moved, y_list[:-1])
                # y_list_new = np.divide(y_list_new, time_list_new)
                # stack x', y', time
                text_line_data = np.stack(
                    (x_list_new, y_list_new, time_list[:-1]), axis=1)
                # text_line_data = np.stack(
                #     (x_list, y_list, time_list), axis=1)
                temp_length = text_line_data.shape[0]
                # subsampling
                # text_line_data = text_line_data[[i % 3 == 0 for i in range(temp_length)]]
                text_line_data_all.append(text_line_data)
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


if __name__ == "__main__":
    main()
