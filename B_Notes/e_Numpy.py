# 01/18/2019 Reopen cheatsheet

import numpy as np


def generate_ndarray():

    # TODO: generate a ndarray, from 1 to 10, step 1, shape as 5 row, 2 col
    arr = np.arange(1,11,1).reshape((5,2))

    # TODO: what's the difference between arrange & linspace?
    arr = np.linspace(1,10,10).reshape((5,2))

    # TODO: generate an all-zero matrix, all-one matrix
    arr = np.zeros((5,2))
    arr = np.ones((5,2))

    # TODO: can we print(arr) directly?

    # TODO: generate an diagonal matrix
    arr = np.eye(5, 2, 0)
    print(arr)
    arr = np.eye(5, 5, 1)
    print(arr)
    arr = np.identity(5, dtype="float")
    print(arr)

    return


generate_ndarray()

