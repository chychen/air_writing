""" data_synthesization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
import os.path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SYNTHESIZE_DATA_DIR_PATH = os.path.join(DIR_PATH, 'synthesized_data')
NORMALIZED_ALPHABET_FILE_PATH = os.path.join(DIR_PATH, 'normalized_data/User_1.json')
DICTIONARY_DIR_PATH = os.path.join(DIR_PATH, 'dictionary')
FLAG_IF_VISULIZZATION = True

INTERVAL_WIDTH = 0.5
# TODO: stddev for more natural interval



# # TODO: read normalized data for each char from one author
# with codecs.open('1/_a.json', 'r', 'utf-8-sig') as f:
#     normalized_data = json.load(f)

# exit('a')
# # TODO: read target voc list

# # TODO: link two char function with width and smooth control
# speed_threshold = 0.1
# stride = 1.0 + 0.5

# # TODO: generate one word

# # TODO: visualization

# fig = plt.figure()
# # scatter
# # line
# data1 = np.array(normalized_data['data'])
# data2 = np.array(normalized_data['data'])
# data2[:, 0] += stride
# plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='o')
# plt.scatter(data2[:, 0], data2[:, 1], c='r', marker='o')
# plt.plot(data1[:, 0], data1[:, 1], c='g')
# plt.plot(data2[:, 0], data2[:, 1], c='g')

# plt.show()





def synthesize_one_word(voc, interval):
    """
    parameters: 
        voc: string, the word as synthesize target
    """
    with codecs.open(NORMALIZED_ALPHABET_FILE_PATH, 'r', 'utf-8-sig') as f:
        alphabet_dict = json.load(f)
    for i, v in enumerate(voc):
        print (v)


def visulization_2D(new_pos):
    """ show proccessed data in 2d
    """
    plt.figure(1)
    plt.plot(new_pos[:, 0], new_pos[:, 1])
    plt.axis([0, 1, 0, 1])


def main():
    """a
    """
    voc_dict_path = os.path.join(DICTIONARY_DIR_PATH, 'testing_voc.json')
    with codecs.open(voc_dict_path, 'r', 'utf-8-sig') as f:
        voc_dict = json.load(f)
    for _, v in enumerate(voc_dict):
        synthesize_one_word(v, INTERVAL_WIDTH)


if __name__ == '__main__':
    if not os.path.exists(SYNTHESIZE_DATA_DIR_PATH):
        os.makedirs(SYNTHESIZE_DATA_DIR_PATH)
    assert os.path.exists(NORMALIZED_ALPHABET_FILE_PATH) is True
    assert os.path.exists(DICTIONARY_DIR_PATH) is True
    main()
