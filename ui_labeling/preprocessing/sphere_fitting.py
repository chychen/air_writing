from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
import os.path
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.linalg

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.join(DIR_PATH, 'voc/4322')
NORMALIZED_DATA_DIR_PATH = os.path.join(DIR_PATH, 'normalized_voc/4322')
FLAG_IF_VISULIZZATION = False


def normalize(positions):
    """
    params: positions: 2-d numpy array
    return: result: 2-d numpy array, normalize its height from 0 to 1, width starts from 0
    """
    result = np.empty(positions.shape, dtype=np.float32)
    # x_amax = np.amax(positions[:, 0]) #unused
    x_amin = np.amin(positions[:, 0])
    y_amax = np.amax(positions[:, 1])
    y_amin = np.amin(positions[:, 1])
    y_range = y_amax - y_amin
    scale = 1.0 / y_range
    result[:, 0] = (positions[:, 0] - x_amin) * scale
    result[:, 1] = 1.0 - ((positions[:, 1] - y_amin) * scale)

    return result


def visulization_3D(fig_id, positions, head_position):
    """ 
    params: fig_id: positive integer, canvas figure id
    params: positions: 3d positions as numpy array
    params: head_position: 3-d numpy array, the center of fitting sphere
    visulize 3d positions on specific figure
    """
    fig = plt.figure(fig_id)
    ax = fig.add_subplot(111, projection='3d')
    # scatter x:x, y:z, z:y
    ax.scatter(positions[:, 0], positions[:, 2],
               positions[:, 1], c='b', marker='o')
    ax.scatter(head_position[:, 0], head_position[:, 2],
               head_position[:, 1], c='r', marker='o')
    # line
    plt.plot(positions[:, 0], positions[:, 2], positions[:, 1], c='g')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')


def visulization_2D(fig_id, new_pos):
    """ 
    params: fig_id: positive integer, canvas figure id
    params: positions: 3d positions as numpy array
    visulize 2d positions on specific figure
    """
    plt.figure(fig_id)
    plt.plot(new_pos[:, 0], new_pos[:, 1])


def transforme_onto_sphere_coordinates(positions, head_position):
    """
    params: positions: 3d positions as numpy array
    params: head_position: 3-d numpy array, the center of fitting sphere
    return: sphere_coordinates: 2-d numpy array as [theta, phi] in the sphere coordinates
    """
    relative_pos = positions - head_position

    phi = np.empty(relative_pos.shape[0])
    theta = np.empty(relative_pos.shape[0])
    for i, v in enumerate(relative_pos):
        x = v[0]
        y = v[1]
        z = v[2]
        # we always write as clockwise
        if x < 0 and z < 0:  # starting Quadrant
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
    sphere_coordinates = np.stack([theta, phi], axis=-1)

    return sphere_coordinates


def project_onto_ball(positions, head_position, radius):
    """
    params: positions: 3-d numpy array, the original positions collected from unity
    params: head_position: 3-d numpy array, the center of fitting sphere
    params: radius: float, radius of the sphere
    return:
    """
    new_positions = np.array(positions)
    for i, v in enumerate(positions):
        vec = [v[0] - head_position[i][0],
               v[1] - head_position[i][1],
               v[2] - head_position[i][2]]
        dist = math.sqrt(vec[0]**2 +
                         vec[1]**2 +
                         vec[2]**2)
        vec = np.array(vec)
        vec = vec / dist * radius
        new_positions[i] = [head_position[i][0] + vec[0],
                            head_position[i][1] + vec[1],
                            head_position[i][2] + vec[2]]
    return new_positions


def fit_radius(positions, head_position):
    """
    params: positions: 3d position numpy array
    return: r: fitted radius for 3d ball
    """
    n = positions.shape[0]
    ball_function = (positions[:, 0] - head_position[:, 0])**2 + \
        (positions[:, 1] - head_position[:, 1])**2 + \
        (positions[:, 2] - head_position[:, 2])**2

    r = math.sqrt(np.sum(ball_function) / n)
    return r


def vr_sphere_fitting(raw_data):
    """
    input: raw_data: json
    output: data_dict: json, normalized
    """
    data_dict = {}
    word_data_list = []

    pos_list = []
    head_pos_list = []
    for i in range(len(raw_data['data'])):
        pos_list.append(raw_data['data'][i]['position'])
        head_pos_list.append(raw_data['data'][i]['head'])
    pos_list = np.array(pos_list)
    head_pos_list = np.array(head_pos_list)

    radius = fit_radius(pos_list, head_pos_list)

    pos_new = project_onto_ball(pos_list, head_pos_list, radius)

    ball_coordinates = transforme_onto_sphere_coordinates(
        pos_new, head_pos_list)

    normalized_pos = normalize(ball_coordinates)

    for i, v in enumerate(normalized_pos):
        temp_dict = {}
        temp_dict['pos'] = v.tolist()
        temp_dict['face'] = raw_data['data'][i]['face']
        temp_dict['time'] = raw_data['data'][i]['time']
        temp_dict['dir'] = raw_data['data'][i]['direction']
        temp_dict['vel'] = raw_data['data'][i]['velocity']
        temp_dict['tag'] = raw_data['data'][i]['tag']
        word_data_list.append(temp_dict)

    data_dict['uid'] = raw_data['id']
    data_dict['name'] = raw_data['name']
    data_dict['fps'] = raw_data['fps']
    data_dict['word'] = raw_data['word']
    data_dict['data'] = word_data_list
    print ("Successfully normalized!")
    return data_dict


def fit_sphere(data_path, result_path):
    """
    params: data_path: location of the original data collected via VIVE(unity leap motion)
    params: result_path: location of the normalized data
    return: : boolean: if the normalization 
    saved json format:
        create folder: {uid}
        word.json
        --[name]: string
        --[uid]: integer
        --[fps]: integer
        --[word]: string
        --[data]: dict in list
        ----[pos]: 2d list
        ----[face]: 3d list
        ----[time]: float
        ----[dir]: float
        ----[vel]: float
        ----[tag]: int
    """
    if not os.path.isdir(data_path):
        print ("ERROR: Directory Not Found:", data_path)
        return False
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    global FLAG_IF_VISULIZZATION
    for _, _, files in os.walk(data_path):
        data_dict = {}
        # read voc one by one as pos_list
        for fi in files:
            word_data_list = []
            filename = os.path.join(data_path, fi)
            with codecs.open(filename, 'r', 'utf-8-sig') as f:
                pos_list = []
                head_pos_list = []
                raw_data = json.load(f)
                for i in range(len(raw_data['data'])):
                    pos_list.append(raw_data['data'][i]['position'])
                    head_pos_list.append(raw_data['data'][i]['head'])
                pos_list = np.array(pos_list)
                head_pos_list = np.array(head_pos_list)

                radius = fit_radius(pos_list, head_pos_list)

                pos_new = project_onto_ball(pos_list, head_pos_list, radius)

                ball_coordinates = transforme_onto_sphere_coordinates(
                    pos_new, head_pos_list)

                normalized_pos = normalize(ball_coordinates)

                # only visulize the first vocabulary
                if FLAG_IF_VISULIZZATION:
                    visulization_3D(1, pos_list, head_pos_list)
                    visulization_3D(2, pos_new, head_pos_list)
                    visulization_2D(3, normalized_pos)
                    plt.show()

                for i, v in enumerate(normalized_pos):
                    temp_dict = {}
                    temp_dict['pos'] = v.tolist()
                    temp_dict['face'] = raw_data['data'][i]['face']
                    temp_dict['time'] = raw_data['data'][i]['time']
                    temp_dict['dir'] = raw_data['data'][i]['direction']
                    temp_dict['vel'] = raw_data['data'][i]['velocity']
                    temp_dict['tag'] = raw_data['data'][i]['tag']
                    word_data_list.append(temp_dict)

                data_dict['uid'] = raw_data['id']
                data_dict['name'] = raw_data['name']
                data_dict['fps'] = raw_data['fps']
                data_dict['word'] = raw_data['word']
                data_dict['data'] = word_data_list
            # print ("Successfully normalize vocabulary::", fi)

            stored_filepath = os.path.join(
                result_path, str(data_dict['word']) + '.json')
            with codecs.open(stored_filepath, 'w', 'utf-8') as out:
                json.dump(data_dict, out, encoding="utf-8", ensure_ascii=False)
            # print ("Saved to file path::", stored_filepath)


    if FLAG_IF_VISULIZZATION:
        plt.close('all')

    return True


if __name__ == '__main__':
    if not fit_sphere(DATA_DIR_PATH, NORMALIZED_DATA_DIR_PATH):
        print ("!!!!!!!!!!!!!!!!!Failed!!!!!!!!!!!!!!!!!")
    else:
        print ("!!!Successfully normalize all vocs!!!")
        
