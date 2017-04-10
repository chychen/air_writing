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

import scipy.linalg


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.join(DIR_PATH, 'voc/rich')


def curve_fitting(data):

    X, Y = np.meshgrid(np.arange(0.0, 1.0, 0.1), np.arange(0.0, 1.0, 0.1))
    XX = X.flatten()
    YY = Y.flatten()

    order = 2    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])    # coefficients

        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(
            data[:, :2], axis=1), data[:, :2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX *
                         YY, XX**2, YY**2], C).reshape(X.shape)

    # plot points and fitted surface
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)

    # temp1 = [X[0][0], Y[0][0], Z[0][0]]
    # temp2 = [X[0][1], Y[0][1], Z[0][1]]
    # temp3 = [X[1][0], Y[1][0], Z[1][0]]
    # ax.scatter(temp1[0], temp1[1], temp1[2], c='g', s=50)
    # ax.scatter(temp2[0], temp2[1], temp2[2], c='g', s=50)
    # ax.scatter(temp3[0], temp3[1], temp3[2], c='g', s=50)


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


def normalize_0to1(dim_vec):
    a_max = np.amax(dim_vec)
    a_min = np.amin(dim_vec)
    a_range = a_max - a_min
    dim_vec = (dim_vec - a_min) / a_range * 1.0
    return dim_vec

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
                pos_list[:, 0] = normalize_0to1(pos_list[:, 0])
                pos_list[:, 1] = normalize_0to1(pos_list[:, 1])
                pos_list[:, 2] = normalize_0to1(pos_list[:, 2])
                visulization_ori_3D(pos_list)
                curve_fitting(pos_list)
                plt.show()


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR_PATH):
        os.makedirs(DATA_DIR_PATH)
    main()
