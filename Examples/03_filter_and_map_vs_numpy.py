import numpy as np
from tools_old.definitions import time_it

# data
RANGE = 10000
a = list(range(RANGE))
a_np = np.arange(RANGE)


@time_it
def even_with_filter():
    b = list(filter(lambda x: x % 2 == 0, a))  # if function is None returns values that are True
    return b


@time_it
def even_with_for_loop():
    evens = []
    for num in a:
        if num % 2 == 0:
            evens.append(num)
    return evens


@time_it
def even_with_comprehension():
    return [num for num in a if num % 2 == 0]


@time_it
def even_with_numpy():
    return a_np[a_np % 2 == 0]

r1 = even_with_filter()
r2 = even_with_for_loop()
r3 = even_with_comprehension()
r4 = even_with_numpy()

"""
Conclusion: "filter" is just convenience method. It's even slower than for loop. 
"""