import numpy as np
import pandas as pd
import scipy as sp
from scipy import io
import matplotlib.pyplot as plt
from PyCMU2018.C_ML.SVM.ML701_HW3 import svm

def pred(w, X):
    yhat = np.inner(w, X)
    if yhat >= 0:
        return 1
    else:
        return -1


# computes the accuracy on a test dataset
def accuracy(X, y, w):
    n, p = np.shape(X)
    n_mistakes = 0
    for j in range(n):
        yhat = pred(w, X[j, :])
        if (y[j] > 0 and yhat < 0) or (y[j] < 0 and yhat >= 0):
            n_mistakes = n_mistakes + 1.0
    return 1 - n_mistakes / n


def main():
    # Get training and testing data
    filePath = '../../../data/C_ML/svm/ML701_HW3/mnist2.mat'
    data = sp.io.loadmat(filePath)

    # xval, yval
    xtrain = data['xtrain']
    ytrain = data['ytrain']
    xtest = data['xtest']
    ytest = data['ytest']

    # training
    # n, p = np.shape(data['xtrain'])
    n, p = np.shape(xtrain)
    w0 = np.zeros(p)
    T = 200 * n

    # candidates = [1000, 100, 10, 1, 0.1]
    # for lbda in candidates:
    #     print("Current lambda:", lbda)
    #     w = svm.train(w0, xtrain, ytrain, T, lbda)
    #     print('Train Accuracy: {0}, Test Accuracy: {1}'
    #           .format(accuracy(xtrain, ytrain, w),
    #                   accuracy(xtest, ytest, w))    )

    epochs = [1]*200
    T = [t * n for t in epochs]
    train_accuracy, test_accuracy = [], []
    lbda = 0.1
    w = w0
    for t in T:
        print(t)
        w = svm.train(w, xtrain, ytrain, t, lbda)
        train_accuracy.append(accuracy(xtrain, ytrain, w))
        test_accuracy.append(accuracy(xtest, ytest, w))

    accuracy_plot = pd.DataFrame({
        'Epochs': range(1, 201),
        'train': train_accuracy,
        'test': test_accuracy,
    })
    accuracy_plot.plot(x="Epochs",
                       y=['train', 'test'],
                       grid=True, marker="o"
                       )
    print(accuracy_plot)
    plt.show()

    # lbda = 100.0
    # w = svm.train(w0, data['xtrain'], data['ytrain'], T, lbda)

    # # evaluation
    # print('Train Accuracy: {0}, Test Accuracy: {1}'.format(accuracy(data['xtrain'], data['ytrain'], w),
    #                                                        accuracy(data['xtest'], data['ytest'], w)))


if __name__ == '__main__':
    main()
