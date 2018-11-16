from math import *
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# from scipy.stats import norm

"""
AndrewID: kaichenc; name: Kaichen Chen

Run the main() to learn, predict, 
print data including prior / parameters / 
accuracy, and plot the relation between
number-of-sample and accuracy.

Please read main() first to see the 
order of primary functions. 

data files dir: ../../data/C_ML

@ line
344: def main()
412: calling main()
"""


def call(func):
    # Only a function for testing. Is ignored.
    def wrapper(*args, **kw):
        # print('call %s():' % func.__name__)
        return func(*args, **kw)

    return wrapper


def norm_pdf(x:float=0.0, mean:float= 0.0, std:float=1.0, log_likelihood:bool=False) -> float:
    denorm = 1 / sqrt(2 * pi * pow(std, 2))
    power = - pow(x - mean, 2) / (2 * pow(std, 2))

    if log_likelihood:
        return log(denorm) + power
    else:
        return denorm * exp(power)

# def test():
#     x = -0.5
#     miu = 2.
#     std = 3.
#     print(norm_pdf(x, miu, std))
#     print(norm(miu, std).pdf(x))
#
# test()

@call
def percepton(x:np.ndarray, y:int, weight:np.ndarray, bias)->bool:
    return (y * (np.inner(weight, x) + bias)) > 0

def responsibility(x:list, u:list, s:list, pi:list)-> list:
    # x: n, k;  u: k, d;   s: k;    pi: k;
    denominator = 0.
    for i in range(0, len(x)):
        pass
    # z: n, k
    return list()

def new_mean(r:list, x:list)->list:
    # r: n, k;  x: n, k
    # nk: k;
    nk = []

    # u: k, d
    return list()


@call
def getInput(filename:str) -> np.ndarray:
    input = np.array([
        list(map(float, row[0:5]))
        for row
        in list(csv.reader(open(filename, 'r'), delimiter =' '))
    ])
    # print(input)
    n, p = np.shape(input)
    # print('Input size:{0}, {1}'.format(n, p))
    return input

@call
def main():
    dataFolder  = '../../data/C_ML/GaussianMixture/'
    XTrain = getInput(dataFolder + 'X.txt')
    gmm = GaussianMixture(n_components=3, covariance_type='spherical', init_params='random', max_iter=1000)
    gmm.fit(XTrain)

    # Question 5.1
    for i in range(0, 3):
        print('Component {0}:\n'
              '  mean:  {1}\n'
              '  stds:  {2}\n'
              '  weig:  {3}'
              ''.format(
            i, gmm.means_[i].round(4),
            round(sqrt(gmm.covariances_[i]), 4),
            gmm.weights_[i].round(4)
        ))

    # Question 5.2
    label = gmm.predict(XTrain)
    plt.scatter(XTrain[:, 0], XTrain[:, 1], c=label, cmap='viridis')
    plt.xlabel('xi1')
    plt.ylabel('xi2')
    plt.show()
    plt.scatter(XTrain[:, 2], XTrain[:, 3], c=label, cmap='viridis')
    plt.xlabel('xi3')
    plt.ylabel('xi4')
    plt.show()
    plt.scatter(XTrain[:, 3], XTrain[:, 4], c=label, cmap='viridis')
    plt.xlabel('xi4')
    plt.ylabel('xi5')
    plt.show()

    # Question 5.3




# main()

def draft():
    pass

