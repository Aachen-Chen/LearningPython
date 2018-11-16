from math import *
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# from scipy.stats import norm


def BasicType():
    # using map on 2d array
    print("using map on 2d array")
    l = [
        ['2.182155676107039977e+00', '-3.305712673473705232e-01', '-2.600624101152366682e-02',
         '1.257961513453821345e+00', '1.036243827074764701e+00'],
        ['8.596631816560478256e-01', '-1.032116809507654720e+00', '-6.989320237768547051e-01',
         '-1.082778968644111939e-01', '-6.594311032523109128e-01']
    ]
    newl = list(map(float, l[0]))
    print(newl)
    # print(  list(map(float, l))    )
    # TypeError: float() argument must be a string or a number, not 'list'

    print(np.array(newl))

