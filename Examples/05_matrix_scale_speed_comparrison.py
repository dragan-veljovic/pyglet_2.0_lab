import typing

from tools_old.graphics import *
from tools_old.definitions import time_it

LIMIT = 10000
REPEATS = 200
ax = 500
ay = 500
sx = 0.5
sy = 1.5

one_point = np.array([[300, 400], ])

# dummy point lists
points_list = list(zip(range(LIMIT), range(LIMIT)))
points_array = np.column_stack((np.arange(LIMIT), np.arange(LIMIT)))


def scale_points_matrix(points, anchor_x, anchor_y, scale_x, scale_y=None):
    """
    Scale passed points using numpy matrix calculation.
    Pass points and np.ndarray for best performance.
    """
    scale_y = scale_y or scale_x
    anchor_matrix = np.array([anchor_x, anchor_y])
    scale_matrix = np.array([[scale_x, 0],
                             [0, scale_y]])
    return np.dot(points - anchor_matrix, scale_matrix) + anchor_matrix


@time_it
def benchmark_scale_points_python():
    for _ in range(REPEATS):
        scale_points(points_list, ax, ay, sx, sy)


@time_it
def benchmark_scale_points_numpy():
    for _ in range(REPEATS):
        scale_points_numpy(points_array, ax, ay, sx, sy)

@time_it
def benchmark_scale_points_matrix():
    for _ in range(REPEATS):
        scale_points_matrix(points_array, ax, ay, sx, sy)

if __name__ == "__main__":
    benchmark_scale_points_python()  # relative speed 1
    benchmark_scale_points_numpy()  # 10x (@1000x60) ~ 50x (@10000 x 200)
    benchmark_scale_points_matrix()  # 10x faster (@1000x60) ~ 20x (@10000 x 200)

