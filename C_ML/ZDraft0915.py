
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readtxt(url="D:/1.txt"):
    df = pd.read_csv(url, header=0, dtype=float)
    # df.loc[df.iloc[:, 0].size] = df.columns.tolist()[0]
    l = df.iloc[:, 0].values.tolist()
    l.insert(0, float(df.columns.tolist()[0]))
    return l

def plotMLE_all(X, theta, input="DataFrame"):

    x1 = X[0:1000]
    x2 = X[0:10000]
    x3 = X

    l_1000 = []
    l_10000 = []
    l_100000 = []

    for i, t in enumerate(theta):
        l_1000.append(math.log(t)*len(x1) + math.log(1-t) * np.array(x1).sum())
        l_10000.append(math.log(t)*len(x2) + math.log(1-t) * np.array(x2).sum())
        l_100000.append(math.log(t)*len(x3) + math.log(1-t) * np.array(x3).sum())

    likelihood_df = pd.DataFrame({
        'theta': theta,
        'l_1000': l_1000,
        'l_10000': l_10000,
        'l_100000': l_100000
    })

    likelihood_df.plot(x="theta", grid=True, markevery=[4,10, 50])

def plotMLE(X, theta, input="DataFrame", plot_line=True):
    l = []
    for i, t in enumerate(theta):
        l.append(
            math.log(t) +
            math.log(1-t) * (np.array(X).sum() / len(X))
        )

    curmax, indmax = l[0], 0
    for i, t in enumerate(l):
        if t>curmax:
            curmax=t
            indmax=i
    likelihood_df = pd.DataFrame({
        'theta': theta,
        'likelihood': l
    })

    if plot_line:
        likelihood_df.plot(x="theta", y="likelihood", grid=True, marker="o", markevery=[indmax])
    return l, indmax

def plotMAP(X: list, theta, alpha: float=1.0, beta: float=2.0, plot_line=True):
    l = []
    for i, t in enumerate(theta):
        l.append(
            math.log(t) +
            math.log(1-t) * (beta-1+ np.array(X).sum() )/(alpha-1+len(X)) +
            0.5
             # math.call(B(alpha, beta))
        )

    curmax, indmax = l[0], 0
    for i, t in enumerate(l):
        if t>curmax:
            curmax=t
            indmax=i
    likelihood_df = pd.DataFrame({
        'theta': theta,
        'likelihood': l
    })

    if plot_line:
        likelihood_df.plot(x="theta", y="likelihood", grid=True, marker="o", markevery=[indmax])
    return l, indmax


X = readtxt("D:/10701 ML/hw1/hw1_dataset.txt")
theta = np.arange(0.01, 1., 0.01, dtype=float).tolist()

tags = ['MLE1k', 'MLE10k', 'MLE100k', 'MAP1k', 'MAP10k', 'MAP100k']
maximum = [0,0,0,0,0,0]
summary = pd.DataFrame(columns=tags)

summary["theta"]= theta
summary[tags[0]], maximum[0]= plotMLE(X[0:1000], theta  , plot_line=False)
summary[tags[1]], maximum[1]= plotMLE(X[0:10000], theta , plot_line=False)
summary[tags[2]], maximum[2]= plotMLE(X, theta          , plot_line=False)
summary[tags[3]], maximum[3]= plotMAP(X[0:1000], theta  , plot_line=False)
summary[tags[4]], maximum[4]= plotMAP(X[0:10000], theta , plot_line=False)
summary[tags[5]], maximum[5]= plotMAP(X, theta          , plot_line=False)

# summary.plot(x="theta", grid=True, marker='o', markers=maximum)
# plt.plot(summary['theta'], summary[summary.columns.tolist()[1:-1]])


# summary = summary
summary = summary[:20]
# summary = summary[5:15]
for i in range(6):
    plt.plot(summary['theta'], summary[tags[i]], marker='o', markevery=[maximum[i]])
plt.legend(tags, loc='lower center', shadow=True)
plt.show()



# def returntwo():
#     return 1, 2
# a, b = returntwo()
# print(a, b)

# df = pd.DataFrame({'a':[1,2,3], 'b': [6,7,8]})
# df['a'].plot(grid=True)

# print(np.arange(0.01, 1., 0.01, dtype=float).tolist())

# l = df.iloc[:, 0].value.tolist()
# l.insert(0, df.columns.tolist()[0])
# print(l)

# a =pd.DataFrame([1,2,3])
# print(a)
# print("--------")
# b = a.value.tolist()
# print(b)
# print("--------")
# c = a.iloc[:,0].value.tolist()
# print(c)

# a = [1,2,3]
# b = np.array(a)
# print(b.mean())

# l = [1,2,3]
# for i, val in enumerate(l):
#     print(i, val)

# df = pd.read_csv("D:/1.txt", header=0)
# print(df)
# print("--------")
# df.loc[df.iloc[:,0].size] = df.columns.tolist()[0]
# print(df)

# for i in range(3):
#     print(df.iloc[i,0])

# a = (1,2)
# print(a[1])