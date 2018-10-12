

from math import *

print('abc')


class Any(object):
    def run(self):
        print('abc')

dictionary = {'a': 1, 'b': 2}
for key in dictionary:
    print(dictionary[key])

# l = [1,2,3,4,5]
l = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# print(l[0:1])

# zip_l = zip(*l)
# print(type(zip_l))
# for i in list(zip_l):
#     print(i)
#
# zip_l = zip(l)
# for i in zip_l:
#     print(i)
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

a = 1
a+=3
print(a)

# from C_ML.NaiveBayes.census_income import norm_pdf
def norm_pdf(x: float=0.0, mean: float=0.0, std: float=1.0, log_likelihood:bool=False)-> float:
    denorm = 1/ sqrt(2 * pi * pow(std, 2))
    power = - pow(x-mean, 2) / (2 * pow(std, 2))
    # log((1 / sqrt(2 * pi * (S ** 2 + 10 ** (-9))))) - (float(xik[j]) - E) ** 2 / 2 * (S ** 2 + 10 ** (-9))
    if log_likelihood:
        return log(denorm * exp(power))
    else:
        return denorm * exp(power)

p = norm_pdf(x = 38.0, mean= 37.03133803, std =13.75829498, log_likelihood=True)
print("norm：", p)

print("%.4f%%" %(100./3))













