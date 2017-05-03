from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import codecs
import random
import numpy as np

USER_ID = '3'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.join(DIR_PATH, 'preprocessing/voc')
NORMALIZED_DATA_DIR_PATH = os.path.join(
    DIR_PATH, 'preprocessing/normalized_voc')
LABELED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'labeled_voc')

target_folder = os.path.join(LABELED_DATA_DIR_PATH, USER_ID)

for _, _, files in os.walk(target_folder):
    print (files)
    for fi in files:
        filename = os.path.join(target_folder, fi)
        with codecs.open(filename, 'r', 'utf-8') as f:
            raw_data = json.load(f)
        result_dict = raw_data

        flag = False
        final_list_list = []
        temp_start = None
        temp_end = None
        for i, timestep_dict in enumerate(result_dict['data']):
            if flag is False:
                if timestep_dict['isL'] is True:
                    flag = True
                    temp_start = i
            if flag is True:
                if timestep_dict['isL'] is False or i == (len(result_dict['data']) - 1):
                    temp_end = i
                    flag = False
                temp_list = []
                if temp_start is not None and temp_end is not None:
                    for i in range(temp_start, temp_end, 1):
                        temp_list.append(i)
                if len(temp_list) > 0:
                    final_list_list.append(temp_list)

        if len(final_list_list) > 0:
            result_dict['labeled_idx_list'] = final_list_list

        with codecs.open(filename, 'w', 'utf-8') as out:
            json.dump(result_dict, out,
                      encoding="utf-8", ensure_ascii=False)
