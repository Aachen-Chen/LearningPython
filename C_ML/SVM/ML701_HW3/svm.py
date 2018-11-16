import numpy as np
import random

# Runs the SSGD on SVM objective taking in an initial weight vector
# w0, matrix of covariates Xtrain, a vector of labels ytrain.
# 'T' is the number of passes to be made through the data.
# lbda is the regularization parameter.
# Outputs the learned weight vector w.
def train(w0: np.ndarray, Xtrain, ytrain, T: int, lbda)->np.ndarray:
    # your code
    n, p = np.shape(Xtrain)
    for t in range(1, T+1):
        i = random.randint(0, n-1)

        eta = 1/(lbda * t)
        if ytrain[i] * np.inner(w0, Xtrain[i])<1:
            w0 = (1-lbda*eta)*w0 + eta * ytrain[i] * Xtrain[i]
        else:
            w0 = (1-lbda*eta)*w0

    return w0





