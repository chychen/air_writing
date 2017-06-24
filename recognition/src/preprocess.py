from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import xml.etree.ElementTree as ET
import numpy as np


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "../data/")
TEXTLINE_DATA_PATH = os.path.join(DATA_PATH, "original/")
STROKES_DATA_PATH = os.path.join(DATA_PATH, "lineStrokes/")

# parse STROKES (.xml)
text_line_data_all = []
for path_1, _, _ in os.walk(STROKES_DATA_PATH):
    for path_2, _, _ in os.walk(path_1):
        for path_3, _, files in os.walk(path_2):
            files = sorted(files)
            for file_name in files:  # TextLine files
                # split our .xml (eg: z01-000-01.xml -> z01-000-01)
                text_line_id = file_name[:-3]
                text_line_path = os.path.join(path_3, file_name)
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
                        time_list.append(float(atype.get('time')) - first_time)
                    # print("x:", atype.get('x'), "y:", atype.get('y'), "time:", float(atype.get('time')) - first_time)
                    # input()
                x_list = np.asarray(x_list, dtype=np.float32)
                y_list = np.asarray(y_list, dtype=np.float32)
                time_list = np.asarray(time_list, dtype=np.float32)
                # time 1st order
                time_list_moved = np.array(time_list)
                time_list_moved = time_list_moved[1:]
                time_list_new = np.subtract(time_list_moved, time_list[:-1])
                x_list_moved = np.array(x_list)
                x_list_moved = x_list_moved[1:]
                # x_list throw away the last element
                x_list_new = np.subtract(x_list_moved, x_list[:-1])
                x_list_new = np.divide(x_list_new, time_list_new)
                # y 1st order
                y_list_moved = np.array(y_list)
                y_list_moved = y_list_moved[1:]
                y_list_new = np.subtract(y_list_moved, y_list[:-1])
                y_list_new = np.divide(y_list_new, time_list_new)
                # stack x', y', time
                text_line_data = np.stack(
                    (x_list_new, y_list_new, time_list[:-1]), axis=1)
                text_line_data_all.append(text_line_data)
                # print(text_line_data)
                # print(text_line_data.shape)
                # print(text_line_path)
                # print(np.array(text_line_data_all).shape)
                # input()
text_line_data_all = np.array(text_line_data_all)
print(text_line_data_all.shape)
np.save("test", text_line_data_all)
# use TextLine id to parse text in original file (.xml)


# # parse STROKES (.xml)
# for path_1, dirs_1, _ in os.walk(STROKES_DATA_PATH):
#     dirs_1 = sorted(dirs_1)
#     for dir_1 in dirs_1: # 1st subdir
#         temp_path_1 = os.path.join(path_1, dir_1)
#         for path_2, dirs_2, _ in os.walk(temp_path_1):
#             dirs_2 = sorted(dirs_2)
#             for dirs3 in dirs_2: # 2nd subdir
#                 temp_path_2 = os.path.join(path_2, dirs3)
#                 for _, _, files in os.walk(temp_path_2):
#                     files = sorted(files)
#                     for file_name in files: # TextLine files
#                         text_line_path = os.path.join(temp_path_2, file_name)
#                         e_tree = ET.parse(text_line_path).getroot()
#                         for atype in e_tree.findall('StrokeSet/Stroke/Point'):
#                             print("x:", atype.get('x'), "y:", atype.get('y'), "time:", atype.get('time'))
#                         input()
