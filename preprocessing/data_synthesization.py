""" data_synthesization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SYNTHESIZE_DATA_DIR_PATH = os.path.join(DIR_PATH, 'synthesized_data')
RESULT_DIR_PATH = os.path.join(DIR_PATH, 'results')

# TODO: read normalized data for each char from one author
with codecs.open('1/_a.json', 'r', 'utf-8-sig') as f:
    normalized_data = json.load(f)

exit('a')
# TODO: read target voc list

# TODO: link two char function with width and smooth control
speed_threshold = 0.1
stride = 1.0 + 0.5

# TODO: generate one word

# TODO: visualization

fig = plt.figure()
# scatter
# line
data1 = np.array(normalized_data['data'])
data2 = np.array(normalized_data['data'])
data2[:,0] += stride
plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='o')
plt.scatter(data2[:, 0], data2[:, 1], c='r', marker='o')
plt.plot(data1[:, 0], data1[:, 1], c='g')
plt.plot(data2[:, 0], data2[:, 1], c='g')

plt.show()



if __name__ == '__main__':
    if not os.path.exists(DATA_DIR_PATH):
        os.makedirs(DATA_DIR_PATH)
    if not os.path.exists(RESULT_DIR_PATH):
        os.makedirs(RESULT_DIR_PATH)
    main()
