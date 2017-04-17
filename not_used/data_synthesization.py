""" data_synthesization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
import os.path
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SYNTHESIZE_DATA_DIR_PATH = os.path.join(DIR_PATH, 'synthesized_data')
NORMALIZED_ALPHABET_FILE_PATH = os.path.join(
    DIR_PATH, 'normalized_data/User_1.json')
DICTIONARY_DIR_PATH = os.path.join(DIR_PATH, 'dictionary')
FLAG_IF_VISULIZZATION = True

INTERVAL_WIDTH = 0.7
SPEED_THRESHOLD = 0.1
# TODO: stddev for more natural interval


def velocity_to_speed(velocity):
    """
    velocity: a list of three element
    return: scalar of velocity
    """
    return math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)


def synthesize_one_word(voc, interval):
    """
    parameters: 
        voc: string, the word as synthesize target
        interval: the width between two alphabets
    """
    global FLAG_IF_VISULIZZATION

    result = np.ndarray([0, 2], buffer=None)
    print (result)
    with codecs.open(NORMALIZED_ALPHABET_FILE_PATH, 'r', 'utf-8-sig') as f:
        alphabet_dict = json.load(f)
    for i, v in enumerate(voc):
        print (v)
        temp_list = []
        plt.figure(1)
        for idx, timestep_dict in enumerate(alphabet_dict[v]):
            velocity = timestep_dict['velocity']
            timestep_pos = timestep_dict['position']
            temp_list.append(
                [timestep_pos[0] + i * INTERVAL_WIDTH, timestep_pos[1]])
            speed = velocity_to_speed(velocity)
            if speed <= SPEED_THRESHOLD:
                # visualiza noise by scatter slow speed points
                plt.scatter(timestep_pos[0] + i * INTERVAL_WIDTH,
                            timestep_dict['position'][1], c='r', marker='o')
        print("################")
        temp_list = np.array(temp_list)
        result = np.concatenate((result, temp_list), axis=0)
    plt.plot(result[:, 0], result[:, 1])
    plt.show()

    # TODO: add sample points in connectionist

    # TODO: add noise to each alphabets in one word

    # TODO: FLAG_IF_VISULIZZATION
    # if FLAG_IF_VISULIZZATION:
    #     visulization_2D(result)
    #     FLAG_IF_VISULIZZATION = False

    # TODO: Save to json file
    print ("successfully synthesize the word:: ", voc)


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

    # save to file as json word by word
    for _, v in enumerate(voc_dict['data']):
        synthesize_one_word(v, INTERVAL_WIDTH)


if __name__ == '__main__':
    if not os.path.exists(SYNTHESIZE_DATA_DIR_PATH):
        os.makedirs(SYNTHESIZE_DATA_DIR_PATH)
    assert os.path.exists(NORMALIZED_ALPHABET_FILE_PATH) is True
    assert os.path.exists(DICTIONARY_DIR_PATH) is True
    main()
