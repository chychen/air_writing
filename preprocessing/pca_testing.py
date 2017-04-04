#!/usr/bin/evn python

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
import codecs
import os.path
import math


result = []
pos_list = []
filename = "preprocessing/data/1/_a.json"
with codecs.open(filename, 'r', 'utf-8-sig') as f:
    raw_data = json.load(f)
word_name = raw_data['word']
for i in range(len(raw_data['data'])):
    pos_list.append(raw_data['data'][i]['position'])
pos_list = np.array(pos_list)


# some 3-dim points
mean = np.array([0.0, 0.0, 0.0])
cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
data = np.random.multivariate_normal(mean, cov, 50)

data = pos_list / 10
# regular grid covering the domain of the data
X, Y = np.meshgrid(np.arange(-1.0, 1.0, 0.5), np.arange(-1.0, 1.0, 0.5))
XX = X.flatten()
YY = Y.flatten()

order = 1    # 1: linear, 2: quadratic
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
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)

temp1 = [X[0][0], Y[0][0], Z[0][0]]
temp2 = [X[0][1], Y[0][1], Z[0][1]]
temp3 = [X[1][0], Y[1][0], Z[1][0]]
ax.scatter(temp1[0], temp1[1], temp1[2], c='g', s=50)
ax.scatter(temp2[0], temp2[1], temp2[2], c='g', s=50)
ax.scatter(temp3[0], temp3[1], temp3[2], c='g', s=50)

plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()
