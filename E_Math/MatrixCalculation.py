
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# X = np.diag((1,2,3))
#
# arr = \
#     [
#         [0, 1, 0],
#         [0, 0, 1],
#         [-1, -3, -3],
#     ]
# Y = np.array(arr, dtype=float)
# print(Y)
#
# evalue, evector = np.linalg.eig(Y)
#
# # print(np.array(evalue, dtype=int))
# # print(np.array(evector, dtype=int))
# print(evector)
#
# print(5**5)



# def calculation(X):
#     return 5 * X**3 + 2 * X**2 - 8*X + 3
#
# X = np.arange(0.605, 0.615, 0.00005)
# print(X)
# Y = np.array([calculation(i) for i in X.tolist()])
# print(Y)

# def root(X):
#     return math.sqrt(6+X)
#
# X = np.arange(0, 10, 1)
# # Y = X.tolist()
# # print(Y)
# Y = X.tolist()
# # Y[0] = 6
# for i in range(1, len(Y)):
#     Y[i] = root(Y[i-1])
#
# print(Y)
#
# functions = pd.DataFrame(
#     {
#     'X': X,
#     'Y': Y,
#     }
# )
#
# functions.plot(X = 'X')
# plt.show()
#
# # print(math.sqrt(6))

#
# X =np.array(
#     [
#         [0.25, 4, 1],
#         [0, 8, 4],
#         [0.125, 3, 2],
#     ], dtype=float
# )
# d1 = np.linalg.det(X)
# print(d1)
# X.transpose()
# d2 = np.linalg.det(X)
# print(d1 * d2)




# X = 1
# print(math.factorial(0))

def combination(m, n):
    return math.factorial(m) / (math.factorial(n) * math.factorial(abs(m-n)))

# print(combination(2,1))

def prob(p, n, x):
    return combination(n, x) * (p ** x) * ((1-p) ** (n-x))

print(prob(0.5, 2, 2))

X = np.arange(0, 100, 1)
Y = [prob(0.5, 100, i) for i in X]
# print(Y)
print(sum(Y[0:60]))



