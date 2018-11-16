

from math import *
import numpy as np


# Only a function for testing. Is ignored.
def call(func):
    def wrapper(*args, **kw):
        # print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper





#
# E = 1.
# S = 2.
# xijk = 3.
#
# l =(1/sqrt(2 * pi * (S ** 2 + 10 ** (-9) ))) * exp(-(xijk - E) ** 2 / 2 * (S ** 2 + 10 ** (-9)))
# print(l)
# # print((1 / sqrt(2 * pi * (S ** 2 + 10 ** (-9)))) * exp( -(float(xijk) - E) ** 2 / 2 * (S ** 2 + 10 ** (-9))))
# print(log(l))
# print(log((1 / sqrt(2 * pi * (S ** 2 + 10 ** (-9))))) -(float(xijk) - E) ** 2 / 2 * (S ** 2 + 10 ** (-9)))
#
#
#

def eunknown():
    a = 1
    a += 3
    print(a)

    # from C_ML.NaiveBayes.census_income import norm_pdf
    def norm_pdf(x: float = 0.0, mean: float = 0.0, std: float = 1.0, log_likelihood: bool = False) -> float:
        denorm = 1 / sqrt(2 * pi * pow(std, 2))
        power = - pow(x - mean, 2) / (2 * pow(std, 2))
        # log((1 / sqrt(2 * pi * (S ** 2 + 10 ** (-9))))) - (float(xik[j]) - E) ** 2 / 2 * (S ** 2 + 10 ** (-9))
        if log_likelihood:
            return log(denorm * exp(power))
        else:
            return denorm * exp(power)

    p = norm_pdf(x=38.0, mean=37.03133803, std=13.75829498, log_likelihood=True)
    print("norm：", p)

    print("%.4f%%" % (100. / 3))

# eunknown()


#
#

# z = np.zeros((5,2))
# o = np.ones((5,2)).T
# # print(z)
# # print(o)
# # print(z+o)
#
# eta, lbda = 1, 1
# w0 = np.zeros(5)
# yit = -1
# Xit = np.ones(5)
# w0 = (1-lbda*eta)*w0 + eta * yit * Xit
#
# print(w0)




def e1836():
    l = [1,2,3]
    print(l*3)
    print([ll * 3 for ll in l])

# e1836()

def e1025():
    d = {}
    d['a'] = 1
    # print(d['a'])
    print(d.keys().__contains__('a'))



# e1025()


def solution(D, A) -> dict:
    # write your code in Python 3.6
    # 1 make a dict
    tree = {}
    for i in range(len(A)):
        if tree.keys().__contains__(A[i]):
            tree[A[i]].append(i)
        else:
            tree[A[i]] = [i]

    # apply dfs on dict



    return tree

# A = [-1, 1,2,3]
# print(solution(1, A))

def some():
    a = 1
    class inner:
        def __init__(self, a):
            self.a = a
        b = 1
        def innerfunc(self):
            self.a += 1
            self.b += 1
            if self.a != 1:
                pass
    # a = innerfunc(a)
    c = inner(a)
    c.a = a
    c.innerfunc()
    a = c.a
    print(a)

    # print(len(l))
    l = [3,2,1]
    l = sorted(l)
    print(l)
    return a

# some()

def e0953():
    l = list(range(1,4))


    print(l)

# e0953()

import time
current_milli_second = lambda: int(round(time.time() * 1000))

@call
def e0853():
    print(current_milli_second())

e0853()