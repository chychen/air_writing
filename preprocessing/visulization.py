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
DATA_DIR_PATH = os.path.join(DIR_PATH, 'voc/rich')


def visulization_ori_3D(ori_pos):
    """ show original data
    """
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    # scatter
    ax.scatter(ori_pos[:, 0], ori_pos[:, 1], ori_pos[:, 2], c='b', marker='o')
    # line
    plt.plot(ori_pos[:, 0], ori_pos[:, 1], ori_pos[:, 2], c='g')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def visulization_2D(new_pos):
    """ show proccessed data in 2d
    """
    
    plt.figure(1)
    plt.plot(new_pos[:, 0], new_pos[:, 1])
    # plt.axis([0, 1, 0, 1])


def main():
    """ main
    """
    for _, _, files in os.walk(DATA_DIR_PATH):
        for fi in files:
            filename = os.path.join(DATA_DIR_PATH, fi)
            print (filename)
            with codecs.open(filename, 'r', 'utf-8-sig') as f:
                pos_list = []
                raw_data = json.load(f)
                for i in range(len(raw_data['data'])):
                    pos_list.append(raw_data['data'][i]['position'])
                pos_list = np.array(pos_list)
                visulization_ori_3D(pos_list)

                pca = PCA(n_components=2)
                fit_pca = pca.fit(pos_list)
                pca_data = fit_pca.transform(pos_list)
                pca_conponents = fit_pca.components_
                print (pca_conponents)

                ax_max = np.amax(pca_data[:, 0], axis=0)
                ax_min = np.amin(pca_data[:, 0], axis=0)
                ay_max = np.amax(pca_data[:, 1], axis=0)
                ay_min = np.amin(pca_data[:, 1], axis=0)
                if pca_conponents[0][2] < 0:
                    pca_data[:, 0] = ax_max - pca_data[:, 0]
                if pca_conponents[1][1] < 0:
                    pca_data[:, 1] = ay_max - pca_data[:, 1]
                    
                visulization_2D(pca_data)
                plt.show()


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR_PATH):
        os.makedirs(DATA_DIR_PATH)
    main()
