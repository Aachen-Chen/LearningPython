from math import *
import csv
import copy
import numpy as np
import pandas as pd


def dictionaryExample():

    dictionary = {'a': 1, 'b': 2}
    for key in dictionary:
        print(dictionary[key])

# dictionaryExample()


def listExample():
    l = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    print(l[0:1])

    zip_l = zip(*l)
    print(type(zip_l))
    for i in list(zip_l):
        print(i)

    zip_l = zip(l)
    for i in zip_l:
        print(i)

    print('Best practice:')
    zip_l = zip(*l)
    for i in zip_l:
        print(list(i))

    print("Sum each column:")
    print([
        sum(column)
        for column
        in zip(*l)
    ])

# listExample()

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

# listAndArray()

def MapAndLambda():
    # we want to standardize each row.
    l = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    # Example of map:
    a = [x for x in map(lambda x:x+1, l[0])]
    print(a)

    print("Example on matrix")
    a = [[round(r,4) for r in map(lambda a: a/sum(row), row)]
         for row
         in l]
    print(a)

MapAndLambda()

def others():
    print('abc')

    class Any(object):
        def run(self):
            print('abc')



