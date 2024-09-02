import numpy as np
import timeit


def time_it(func_handle):
    def wrapper(*args, **kwargs):
        result = 0
        start_time = timeit.default_timer()
        for _ in range(10):
            for _ in range(100):
                result = func_handle(*args, **kwargs)
            print(f"{func_handle.__name__}() - Execution time: ", timeit.default_timer() - start_time)
            return result
    return wrapper


@time_it
def update_position(closed=True) -> list:
    # attributes
    segments = SEGMENTS
    radius = 50
    center_x = 500
    center_y = 500
    # variables
    theta = 0
    dtheta = 2 * np.pi / segments
    points = []

    # generating points
    for n in range(segments + 1):
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        theta += dtheta
        points.append((x, y))

    # generating vertices
    line_vertices = []
    for i in range(len(points) - 1):
        line_points = *points[i], *points[i + 1]
        line_vertices.extend(line_points)

    # manually combining last and first point to close a circle
    if closed:
        last_pair = *points[-1], *points[0]
        line_vertices.extend(last_pair)

    return line_vertices


@time_it
def update_position_numpy(closed=True) -> list:
    # attributes
    segments = SEGMENTS
    radius = 50
    center_x = 500
    center_y = 500

    theta = np.linspace(0, 2 * np.pi, segments + 1, dtype=np.float32)
    points = np.column_stack((center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)))

    start_points = points[:-1]
    end_points = points[1:]

    line_vertices = np.column_stack((start_points, end_points)).flatten()

    if closed:
        last_pair = np.column_stack((points[-1], points[0])).flatten()
        line_vertices = np.concatenate((line_vertices, last_pair))

    return line_vertices

# constants
SEGMENTS = 1000
a = np.ndarray([1, 2, 3])
a.fl
# execution
update_position()
update_position_numpy()

# CONCLUSIONS:
# time_it simulates repeated execution of circle update
# on each frame (100) for each circle (10)
# numpy has some kind of initiation time so
# for fewer segments <20 python calculation is equal or faster
# but for 200 segments numpy is faster 10x
# and goes up to 50x for very large number of segments

