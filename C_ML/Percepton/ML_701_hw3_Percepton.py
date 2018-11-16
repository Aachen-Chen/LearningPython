from math import *
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Only a function for testing. Is ignored.
def call(func):
    def wrapper(*args, **kw):
        # print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


@call
def getInput(filename:str) -> np.ndarray:
    fileContent = csv.reader(open(filename, 'r'), delimiter =' ')
    input = np.array(
        [
            list(map(float, row)) for row in list(fileContent)
        ]
    )
    n, p = np.shape(input)
    print('Input size:{0}, {1}'.format(n, p))
    return input

# @call
def percepton(x:np.ndarray, y:int, weight:np.ndarray, bias)->bool:
    return (y * (np.inner(weight, x) + bias)) > 0

# @call
def predict(x:np.ndarray, weight:np.ndarray, bias)->int:
    if ((np.inner(weight, x) + bias))>0:
        return 1
    else:
        return -1

@call
def accuracy(X, y, weight, bias) ->float:
    n, p = np.shape(X)
    num_mistakes = 0.
    for i in range(n):
        yhat = predict(x=X[i], weight=weight, bias=bias)
        if (yhat >= 0 and y[i]< 0) or (yhat < 0 and y[i]>0):
            num_mistakes += 1.0
    return 1- num_mistakes / n

@call
def train(X: np.ndarray, y:np.ndarray, eta:float, numIter: int):
    n, p = np.shape(X)
    w, b = np.zeros(p), 0
    numIter, round = numIter, 0
    isClassified = False

    while(not isClassified and round<numIter+1):
        # print("Round {0} accurarcy: {1}".format(
        #     round, accuracy(X=X, y=y, weight=w, bias=b)
        #     )
        # )
        print(w)

        mistakes = 0
        for i in range(n):
            if not percepton(x=X[i], y=y[i], weight=w, bias=b):
                mistakes += 1.
                w = w + eta * y[i] * X[i]
                b = b + eta * y[i]
        round +=1
        print("Round {0} accurarcy: {1}".format(round, 1 - mistakes / n))

        for i in range(n):
            if not percepton(x=X[i], y=y[i], weight=w, bias=b):
                break
            if i==n-1:
                isClassified = True

    return w, b, round, isClassified



@call
def run():
    dataFolder  = '../../data/C_ML/Percepton/ML701_HW3/'

    XTrain = getInput(dataFolder + 'Xtrain.txt')
    yTrain = getInput(dataFolder + 'ytrain.txt')
    XTest  = getInput(dataFolder + 'Xtest.txt')
    yTest  = getInput(dataFolder + 'ytest.txt')

    w, b, round, isClassified = train(X=XTrain, y=yTrain, eta=1, numIter=100)

    print("Training set distribution")
    unique, counts = np.unique(yTrain, return_counts=True)
    print(dict(zip(unique, counts)))
    print("Testing set distribution")
    unique, counts = np.unique(yTest, return_counts=True)
    print(dict(zip(unique, counts)))

    print('Train Accuracy: {0}, Test Accuracy: {1}'
          .format(accuracy(XTrain, yTrain, w, b),
                  accuracy(XTest, yTest, w, b)
                  )
          )

run()




# @call
# def isAllClassified(X: np.ndarray, y:np.ndarray, weight:np.ndarray, bias)->bool:
#     n, p = np.shape(X)
#     for i in range(n):
#         if not percepton(x=X[i], y=y[i], weight=weight, bias=bias):
#             return False
#     return True