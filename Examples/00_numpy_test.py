import numpy as np
import timeit
lst = list(range(1_000_000))


def time_it(func_handle):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func_handle(*args, **kwargs)
        print("Execution time: ", timeit.default_timer() - start_time)
        return result
    return wrapper

@time_it
def for_loop(iterations: int, lista: list):
    for _ in range(iterations):
        for j in range(len(lista)):
            lista[j] += 1

@time_it
def numpy_loop(iterations: int):
    my_array = np.array(lst)
    for _ in range(iterations):
        my_array += 1
    return my_array.tolist()

@time_it
def generate_points_for_loop(length, repeats, amplitude=100, wavelength=300, step=1, _x=0, _y=0):
    for n in range(repeats):
        points = [(x + _x, amplitude * np.sin(2 * np.pi / wavelength * x) + _y) for x in range(_x, length, step)]

generate_points_for_loop(1000, 600)


@time_it
def generate_points_numpy(length, repeats, amplitude=100, wavelength=300, step=1, _x=0, _y=0):
    for n in range(repeats):
        x = np.arange(_x, length, step)
        y = amplitude * np.sin(2 * np.pi / wavelength * x)
        points = np.column_stack((x, y))

generate_points_numpy(1000, 600)
