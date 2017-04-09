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
FLAG_IF_VISULIZZATION = True

# TODO: get mean position of head and the mean height of all points
HEAD = [-1.37, 1.64, -0.285]


def visulization_ori_3D(fig_id, positions):
    """ show original data
    """
    fig = plt.figure(fig_id)
    ax = fig.add_subplot(111, projection='3d')
    # scatter x:x, y:z, z:y
    ax.scatter(positions[:, 0], positions[:, 2],
               positions[:, 1], c='b', marker='o')
    ax.scatter(HEAD[0], HEAD[2],
               HEAD[1], c='r', marker='o')
    # line
    plt.plot(positions[:, 0], positions[:, 2], positions[:, 1], c='g')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')


def visulization_2D(fig_id, new_pos):
    """ show proccessed data in 2d
    """

    plt.figure(fig_id)
    plt.plot(new_pos[:, 0], new_pos[:, 1])
    # plt.axis([0, 1, 0, 1])


def transformed_onto_ball_coordinates(positions):
    """
    params: 3d positions as numpy array
    return:
    """
    relative_pos = positions - HEAD

    phi = np.empty(relative_pos.shape[0])
    theta = np.empty(relative_pos.shape[0])
    for i, v in enumerate(relative_pos):
        x = v[0]
        y = v[1]
        z = v[2] 
        # we always write as clockwise
        if x < 0 and z < 0: # starting Quadrant
            theta[i] = math.atan(-z / x)
        elif x < 0 and z > 0:
            theta[i] = math.atan(-z / x)
        elif x > 0 and z > 0:
            theta[i] = math.atan(-z / x) + math.pi
        # TODO: not yet verified
        elif x > 0 and z < 0:
            theta[i] = -math.atan(-z / x) + math.pi * 3 / 2

        # updown y because acos distribution
        phi[i] = math.acos(-y / np.sqrt(x**2 + y**2 + z**2))

    # theta = np.arctan(-z / x)
    ball_coordinates = np.stack([theta, phi], axis=-1)

    return ball_coordinates


def project_onto_ball(positions, head_position, radius):
    """
    params:
    return:
    """
    new_positions = np.array(positions)
    for i, v in enumerate(positions):
        vec = [v[0] - head_position[0],
               v[1] - head_position[1],
               v[2] - head_position[2]]
        dist = math.sqrt(vec[0]**2 +
                         vec[1]**2 +
                         vec[2]**2)
        vec = np.array(vec)
        vec = vec / dist * radius
        new_positions[i] = [head_position[0] + vec[0],
                            head_position[1] + vec[1],
                            head_position[2] + vec[2]]
    return new_positions


def fit_radius(positions, head_position):
    """
    params: positions: 3d position numpy array
    return: r: fitted radius for 3d ball
    """
    n = positions.shape[0]
    ball_function = (positions[:, 0] - head_position[0])**2 + \
        (positions[:, 1] - head_position[1])**2 + \
        (positions[:, 2] - head_position[2])**2

    r = math.sqrt(np.sum(ball_function) / n)
    return r


def main():
    """main
    """
    global FLAG_IF_VISULIZZATION
    # read voc one by one as pos_list
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

                radius = fit_radius(pos_list, HEAD)

                pos_new = project_onto_ball(pos_list, HEAD, radius)

                ball_coordinates = transformed_onto_ball_coordinates(pos_new)

                if FLAG_IF_VISULIZZATION:
                    visulization_ori_3D(1, pos_list)
                    visulization_ori_3D(2, pos_new)
                    visulization_2D(3, ball_coordinates)
                    plt.show(block=False)
                    FLAG_IF_VISULIZZATION = False
    
    raw_input("Hit Enter To Close")
    plt.close('all')


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR_PATH):
        os.makedirs(DATA_DIR_PATH)
    main()
