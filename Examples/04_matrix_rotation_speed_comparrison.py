from tools_old.graphics import *
from tools_old.definitions import time_it


def rotate_with_matrix(points, theta_rad, anchor_x, anchor_y):
    # prepare anchor matrix
    anchor = np.array((anchor_x, anchor_y))
    # pre-calculate sin/cos terms
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    # form rotation matrix
    rotation_matrix = np.array([[c, s],
                                [-s,  c]])
    # main calculation
    rotated_points = np.dot(points - anchor, rotation_matrix) + anchor
    return rotated_points

@time_it
def benchmark_rotate_pyglet():
    for _ in range(REPEATS):
        pyglet.shapes._rotate(pts_list, 30, anchor_x, anchor_y)

@time_it
def benchmark_pure_python():
    for _ in range(REPEATS):
        rotate_points(pts_array, theta_rad, anchor_x, anchor_y)

@time_it
def benchmark_rotate_numpy():
    for _ in range(REPEATS):
        rotate_points_numpy(pts_array, theta_rad, anchor_x, anchor_y)

@time_it
def benchmark_rotate_matrix():
    for _ in range(REPEATS):
        rotate_with_matrix(pts_array, theta_rad, anchor_x, anchor_y)


LIMIT = 1000
REPEATS = 60

anchor_x = 500
anchor_y = 500
theta_rad = np.radians(30)
x = np.arange(0, LIMIT, 1)
y = np.arange(0, LIMIT, 1)
pts_array = np.column_stack((x, y))

pts_list = list(zip(range(LIMIT), range(LIMIT)))

benchmark_rotate_pyglet()  # pyglet standard: relative speed 1
benchmark_pure_python()  # my implementation: 2x slower as uses geometry
benchmark_rotate_numpy()  # numpy version: 10x faster
benchmark_rotate_matrix()  # numpy matrix: 20x faster
