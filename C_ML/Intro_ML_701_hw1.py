import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readtxt(url="D:/1.txt"):
    df = pd.read_csv(url, header=0, dtype=float)
    l = df.iloc[:, 0].values.tolist()
    l.insert(0, float(df.columns.tolist()[0]))
    return l


""" 
6.1 (a) 
"""
def plotMLE(X: list,
            thetas: list = np.arange(0.01, 1., 0.01, dtype=float).tolist(),
            plot_line: bool =True
            ) -> tuple:

    """
    :param X: samples
    :param thetas: candidate θ, default at [0.01, 0.02, ... , 0.99]
    :param plot_line: plot line or not
    :return: tuple (list of l(w), l(w) maximizer)
    """

    """ Calculate l(w) """
    likelihood = []
    for i, theta in enumerate(thetas):
        # The original log-likelihood function is:
        #   l(w) = ln(θ)*n + ln(1-t)*sum(X)

        # For better comparison between result, standardize the
        # log-likelihood by dividing the l(w) by n, the number of sample,
        #   l(w)/n = ln(θ) + ln(1-t)*[sum(X)/n]

        # so that standardized l(w) doesn't grow with the sample size.
        likelihood.append(
            math.log(theta) +
            math.log(1-theta) * (np.array(X).sum() / len(X))
        )

    """ Find l(w) maximizer """
    curmax, indmax = likelihood[0], 0
    for i, theta in enumerate(likelihood):
        if theta>curmax:
            curmax=theta
            indmax=i
    likelihood_df = pd.DataFrame({
        'theta': thetas,
        'likelihood': likelihood
    })

    """ Plot l(w) """
    if plot_line:
        likelihood_df.plot(x="theta", y="likelihood", grid=True, marker="o", markevery=[indmax])

    return likelihood, indmax

""" 
6.2 (a) 
"""
def plotMAP(X: list,
            thetas: list = np.arange(0.01, 1., 0.01, dtype=float).tolist(),
            alpha: float=1.0,
            beta: float=2.0,
            plot_line=True
            )-> tuple:
    """
    :param X: samples
    :param thetas: candidate θ, default at [0.01, 0.02, ... , 0.99]
    :param alpha: α of Beta(α, β)
    :param beta: β of Beta(α, β)
    :param plot_line: plot line or not
    :return: tuple (list of l(w), l(w) maximizer)
    """

    """ Calculate l(w) """
    posteriors = []
    for i, theta in enumerate(thetas):
        posteriors.append(
            # The original log-posterior function is:
            #   l(w) = ln(θ)*(n+α-1) + ln(1-θ)*[sum(x)+β-1] - ln[Beta(α, β)]

            # For better comparison between result, standardize l(w)
            # by dividing by (n+α-1):
            #   l(w)/n = ln(θ) + ln(1-θ)*[sum(X)/n]/(n+α-1)

            # so that standardized l(w) doesn't grow with the sample size.
            math.log(theta) +
            math.log(1-theta) * (beta-1+ np.array(X).sum() )/(alpha-1+len(X)) +
            math.log(0.5) / (alpha-1+len(X))
        )

    """ Find l(w) maximizer """
    curmax, indmax = posteriors[0], 0
    for i, theta in enumerate(posteriors):
        if theta>curmax:
            curmax=theta
            indmax=i
    likelihood_df = pd.DataFrame({
        'theta': thetas,
        'likelihood': posteriors
    })

    """ Plot l(w) """
    if plot_line:
        likelihood_df.plot(x="theta", y="likelihood", grid=True, marker="o", markevery=[indmax])

    return posteriors, indmax


""" 
6.1 (b), 6.2 (b) 
"""
X = readtxt("D:/10701 ML/hw1/hw1_dataset.txt")

tags = ['MLE1k', 'MLE10k', 'MLE100k', 'MAP1k', 'MAP10k', 'MAP100k']
maximum = [0,0,0,0,0,0]
summary = pd.DataFrame(columns=tags)

plot_seperatly = True
summary["theta"]= np.arange(0.01, 1., 0.01, dtype=float).tolist()
summary[tags[0]], maximum[0]= plotMLE(X[0:1000] , plot_line=plot_seperatly)
summary[tags[1]], maximum[1]= plotMLE(X[0:10000], plot_line=plot_seperatly)
summary[tags[2]], maximum[2]= plotMLE(X,          plot_line=plot_seperatly)
summary[tags[3]], maximum[3]= plotMAP(X[0:1000],  plot_line=plot_seperatly)
summary[tags[4]], maximum[4]= plotMAP(X[0:10000], plot_line=plot_seperatly)
summary[tags[5]], maximum[5]= plotMAP(X,          plot_line=plot_seperatly)

plt.draw()

print(maximum)

""" Plotting together for better comparison """
plt.figure()
# Plot only the first 20 θ candidates for better comparison
summary = summary[:20]
for i in range(6):
    plt.plot(summary['theta'], summary[tags[i]], marker='o', markevery=[maximum[i]])
plt.legend(tags, loc='lower center', shadow=True)
plt.draw()


"""
6.2 (c)
"""
""" Setting α, β to 1000, 2000"""
plt.figure()

plot_seperatly = False
summary = pd.DataFrame(columns=tags)
a, b = 2200, 20000
summary["theta"]= np.arange(0.01, 1., 0.01, dtype=float).tolist()
summary[tags[0]], maximum[0]= plotMLE(X[0:1000] , plot_line=plot_seperatly)
summary[tags[1]], maximum[1]= plotMLE(X[0:10000], plot_line=plot_seperatly)
summary[tags[2]], maximum[2]= plotMLE(X,          plot_line=plot_seperatly)
summary[tags[3]], maximum[3]= plotMAP(X[0:1000],  alpha=a, beta=b, plot_line=plot_seperatly)
summary[tags[4]], maximum[4]= plotMAP(X[0:10000], alpha=a, beta=b, plot_line=plot_seperatly)
summary[tags[5]], maximum[5]= plotMAP(X,          alpha=a, beta=b, plot_line=plot_seperatly)

summary = summary[:20]
for i in range(6):
    plt.plot(summary['theta'], summary[tags[i]], marker='o', markevery=[maximum[i]])
plt.legend(tags, loc='lower center', shadow=True)
plt.draw()

plt.show()

