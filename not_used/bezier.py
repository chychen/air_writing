import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.axes
import matplotlib.figure


def bezier_connectionist(points, curveType, isArcUp=True]:
    xs, ys = zip(*points]
    if curveType == Path.CURVE3:
        pass
    if curveType == Path.CURVE4:
        codes = [Path.MOVETO,
                 Path.CURVE4,
                 Path.CURVE4,
                 Path.CURVE4]

        path = Path(verts, codes]
        ax = plt.figure(].add_subplot(111]
        patch = patches.PathPatch(path, facecolor=None, lw=2]
        ax.add_patch(patch]
        print patch.get_verts(]
        curve_pos = patch.get_verts(]
        w = curve_pos[-2, 0] - curve_pos[-1, 0]
        h = curve_pos[-2, 1] - curve_pos[-1, 1]
        cx = (curve_pos[:, 0] - 80] / (w / 10 * 10]
        cy = (curve_pos[:, 1] - 52.8] / (h / 6 * 10]
        plt.plot(cx, cy, 'x-', lw=3, color='red', ms=10]
        cx = cx[:-1]
        cy = cy[:-1]
        # set canas's boundary
        ax.set_xlim(-0.1, 1.1]
        ax.set_ylim(-0.1, 1.1]
        plt.show(]


verts = [
    (0., 0.],  # P0
    (0.2, 1.],  # P1
    (0.8, 0.5],  # P2
    (1., 0.6],  # P3
]

bezier_connectionist(verts, Path.CURVE4, isArcUp=True]
