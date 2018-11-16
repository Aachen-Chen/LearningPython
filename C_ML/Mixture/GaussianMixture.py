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

def responsibility(x:list, u:list, s:list, pi:list)-> tuple:
    # x: n, d;  u: k, d;   s: k;    pi: k;
    N = len(x)
    K = len(u)
    D = len(x[0])
    r = []
    for k in range(0, K):
        r.append([])
        for n in range(0, N):
            pdf = 1.0
            for d in range(0, D):
                pdf = pdf * norm_pdf(x[n][d], u[k][d], s[k])
            r[k].append(pi[k] * pdf)

    # turn column to row
    r = [list(row) for row in zip(*r)]

    # calculate current log likelihood of X:
    # log(product of instance likelihood)
    #     = sum(log(instance likelihood))
    # instance likelihood = sum( p(zi=k) * p(xi|zk) ) = sum(row in r)
    log_likelihood = sum([ log(sum(row)) for row in r ])

    # standardize each row
    r = [[r for r in map(lambda a: a/sum(row), row)]
         for row
         in r]
    # r: n, k;
    return r, log_likelihood

def new_mean(r:list, x:list)->list:
    # r: n, k;  x: n, d
    N = len(x)
    K = len(r[0])
    D = len(x[0])

    # nk: k;
    nk = [sum(column) for column in zip(*r)]

    u = []
    for k in range(0, K):
        u.append([])
        for d in range(0, D):
            s = 0.0
            for n in range(0, N):
                s = s + r[n][k] * x[n][d]
            u[k].append(s / nk[k])

    # u: k, d
    return u

@call
def main():
    dataFolder  = '../../data/C_ML/GaussianMixture/'
    x = getInput(dataFolder + 'X.txt')
    N = len(x)
    K = 3

    gmm = GaussianMixture(n_components=K, covariance_type='spherical', init_params='random', max_iter=1000)
    gmm.fit(x)

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

    # # Question 5.2
    label = gmm.predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=label, cmap='viridis')
    plt.xlabel('xi1')
    plt.ylabel('xi2')
    plt.show()
    plt.scatter(x[:, 2], x[:, 3], c=label, cmap='viridis')
    plt.xlabel('xi3')
    plt.ylabel('xi4')
    plt.show()
    plt.scatter(x[:, 3], x[:, 4], c=label, cmap='viridis')
    plt.xlabel('xi4')
    plt.ylabel('xi5')
    plt.show()

    # Question 5.3
    u = new_mean(
        # As per "use random initialization for this part"
        list(np.random.random(size=(N, K))),
        list(x)
    )
    r, log_likelihood = responsibility(
        list(x),
        u,
        [sqrt(cov) for cov in gmm.covariances_],
        gmm.weights_
    )

    while(True):
        u = new_mean(r, list(x))
        r, new_likelihood =responsibility(
            list(x),
            u,
            [sqrt(cov) for cov in gmm.covariances_],
            gmm.weights_
        )
        print(round(log_likelihood, 4), round(new_likelihood, 4))
        if abs(new_likelihood - log_likelihood) < 0.001:
            break
        log_likelihood = new_likelihood

    for i in range(0, 3):
        print('Component {0}:\n'
              '  sklearn:  {1}\n'
              '  manual :  {2}'
              ''.format(
            i,
            list(gmm.means_[i].round(4)),
            [round(mean, 4) for mean in u[i]]
        ))

main()

def draft():
    pass

