""" data_normalization.py
read all alphabet one by one collected from unity and different users,
then generate into one formatted, normalized, 2d-pca json file for each user.
Also, visualize the result in 2d/ 3d space to compare original data and formatted data
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
DATA_DIR_PATH = os.path.join(DIR_PATH, 'data')
NORMALIZED_DIR_PATH = os.path.join(DIR_PATH, 'normalized_data')
DICTIONARY_DIR_PATH = os.path.join(DIR_PATH, 'dictionary')
FLAG_IF_VISULIZZATION = True


def normalize_alphabet(filename, char_alignment_dict):
    """ Read alphabet one by one to normalized, 2d-pca, and aligned with proper position(charAlignmentDict).
    return two variable as below:
    wordName: name of the alphabet
    result: list of position normalized, 2d-pca, and aligned
    """
    global FLAG_IF_VISULIZZATION
    result = []
    pos_list = []
    with codecs.open(filename, 'r', 'utf-8-sig') as f:
        raw_data = json.load(f)
    word_name = raw_data['word']
    for i in range(len(raw_data['data'])):
        pos_list.append(raw_data['data'][i]['position'])
    pos_list = np.array(pos_list)

    # Create a regular PCA and fit 3-D to 2-D
    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(pos_list)

    # normalize to 0-1
    # = (pca_data - pca_data_amin) / pca_data_value_range * 1.0
    pca_data_Amax = np.amax(pcaData, axis=0)
    pca_data_Amin = np.amin(pcaData, axis=0)
    pca_data_value_range = pca_data_Amax - pca_data_Amin

    char_alignment_type = char_alignment_dict['char'][word_name]
    alignment_type_dict = char_alignment_dict['type'][char_alignment_type]
    normalized_pca_data = np.zeros(pcaData.shape)
    normalized_pca_data[:, 0] = (pcaData[:, 0] - pca_data_Amin[0]) / \
        pca_data_value_range[0] * 1.0
    normalized_pca_data[:, 1] = (pcaData[:, 1] - pca_data_Amin[1]) / \
        pca_data_value_range[1] * 1.0

    # transpose direction
    x = normalized_pca_data[:, 0]
    y = normalized_pca_data[:, 1]
    f = x + y - 1
    for i, v in enumerate(f):
        a = 1
        b = 1
        dist = math.fabs(x[i] + y[i] - 1) / (a**2 + b**2) * 2
        if v < 0:
            x[i] = x[i] + dist
            y[i] = y[i] + dist
        else:
            x[i] = x[i] - dist
            y[i] = y[i] - dist
    normalized_pca_data[:, 0] = x
    normalized_pca_data[:, 1] = y

    # align to correct alphabet position according to 'charAlignmentDict'
    normalized_pca_data[:, 0] = normalized_pca_data[:,
                                                0] * alignment_type_dict['yrange']
    normalized_pca_data[:, 1] = normalized_pca_data[:, 1] * \
        alignment_type_dict['yrange'] + alignment_type_dict['ymin']

    # centerize x-axis
    ax_max = np.amax(normalized_pca_data[:, 0], axis=0)
    ax_min = np.amin(normalized_pca_data[:, 0], axis=0)
    ax_range = ax_max - ax_min
    normalized_pca_data[:, 0] = normalized_pca_data[:, 0] + 0.5 - ax_range / 2

    if FLAG_IF_VISULIZZATION:
        visulization_2D(normalized_pca_data)
        visulization_3Dto2D(pos_list, normalized_pca_data)
        plt.show(block=False)
        FLAG_IF_VISULIZZATION = False

    newPos = normalized_pca_data.tolist()
    for i, v in enumerate(newPos):
        temp_dict = {}
        temp_dict['position'] = v
        temp_dict['time'] = raw_data['data'][i]['time']
        temp_dict['direction'] = raw_data['data'][i]['direction']
        temp_dict['velocity'] = raw_data['data'][i]['velocity']
        result.append(temp_dict)

    return word_name, result


def visulization_3Dto2D(ori_pos, new_pos):
    """ show original data vs new proccessed data
    """
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    # scatter
    ax.scatter(ori_pos[:, 0], ori_pos[:, 1], ori_pos[:, 2], c='b', marker='o')
    ax.scatter(new_pos[:, 0], new_pos[:, 1], 0, c='r', marker='o')
    # line
    plt.plot(new_pos[:, 0], new_pos[:, 1], c='g')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def visulization_2D(new_pos):
    """ show proccessed data in 2d
    """
    plt.figure(1)
    plt.plot(new_pos[:, 0], new_pos[:, 1])
    plt.axis([0, 1, 0, 1])


def main():
    """ read alphebet one by one then normalize them and save as json in one file for each author
    """
    final_dict = {}

    # normalize to 0-1 according to 'charAlignmentDict'
    alphabet_dict = os.path.join(DICTIONARY_DIR_PATH, 'char_alignment_dict.json')
    with codecs.open(alphabet_dict, 'r', 'utf-8-sig') as f:
        charAlignmentDict = json.load(f)

    for root, users, _ in os.walk(DATA_DIR_PATH):
        for f in users:
            # get author info
            usersPath = os.path.join(root, f)
            firstFileName = os.path.join(usersPath, 'a.json')
            with codecs.open(firstFileName, 'r', 'utf-8-sig') as f:
                raw_data = json.load(f)
                final_dict['id'] = raw_data['id']
                final_dict['name'] = raw_data['name']
                final_dict['fps'] = raw_data['fps']
                final_dict['face'] = raw_data['data'][0]['face']

            for _, _, files in os.walk(usersPath):
                for f in files:
                    temp_list = []
                    filename = os.path.join(usersPath, f)
                    wordname, temp_list = normalize_alphabet(
                        filename, charAlignmentDict)
                    final_dict[wordname] = temp_list

            # save result as json format
            file_name = 'User_' + str(raw_data['id']) + '.json'
            filepath = os.path.join(NORMALIZED_DIR_PATH, file_name)
            with codecs.open(filepath, 'w', 'utf-8') as out:
                json.dump(final_dict, out, encoding="utf-8",
                          ensure_ascii=False)
                print ("data saved into path:", filepath)

    raw_input("Hit Enter To Close")
    plt.close('all')


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR_PATH):
        os.makedirs(DATA_DIR_PATH)
    if not os.path.exists(NORMALIZED_DIR_PATH):
        os.makedirs(NORMALIZED_DIR_PATH)
    main()
