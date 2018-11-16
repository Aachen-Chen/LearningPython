from math import *
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# from scipy.stats import norm


def dictionaryExample():


    dictionary = {'a': 1, 'b': 2}
    for key in dictionary:
        print(dictionary[key])

dictionaryExample()


def listExample():
    l = [1,2,3,4,5]
    l = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    print(l[0:1])

    zip_l = zip(*l)
    print(type(zip_l))
    for i in list(zip_l):
        print(i)

    zip_l = zip(l)
    for i in zip_l:
        print(i)

def listAndArray():
    # if ndarray change 2d list accordingly
    print('if ndarray change 2d list accordingly')
    l = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(l)
    # np.array() ->list, np.ndarray cannot.
    # a = np.ndarray(l)
    # print(a)
    a = np.array(l)
    print(a)

    # iterate ndarray
    print('iterate ndarray')
    for row in l:
        print(row)
    for row in a:
        print(row)


print('abc')
class Any(object):
    def run(self):
        print('abc')



