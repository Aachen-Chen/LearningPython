from math import *
import csv
import copy
import numpy
import pandas as pd
import matplotlib.pyplot as plt

def call(func):
    def wrapper(*args, **kw):
        # print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


class NaiveBayes(object):

    discrete_attr = {
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                      'Without-pay', 'Never-worked'],
        'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                      '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                           'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                       'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                       'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        'sex': ['Female', 'Male'],
        'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                           'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                           'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                           'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                           'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                           'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
    }

    continue_attr = {
        'age': ['continuous'],
        'fnlwgt': ['continuous'],
        'education-num': ['continuous'],
        'capital-gain': ['continuous'],
        'capital-loss': ['continuous'],
        'hours-per-week': ['continuous'],
    }

    attr = [
        'age', 'workclass', 'fnlwgt', 'education',
        'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country',
    ]

    valid_input = []
    X = {}
    prior = {}
    parameters = {}
    log_likelihood_record = {}
    accuracy = {}
    accuracy_total = 0.0
    input_length = 0

    @call
    def attribute_processing(self):
        for key in self.discrete_attr:
            values_count = {}
            for value in self.discrete_attr[key]:
                values_count[value.strip()] = 0
            self.discrete_attr[key] = values_count

        for key in self.continue_attr:
            self.continue_attr[key] = {}

    @call
    def getInput(self, filename:str, length: int=0):
        self.valid_input = []
        fileContent = csv.reader(open(filename, 'r'))

        if length !=0:
            input = list(fileContent)[0:length]
        else:
            input = list(fileContent)

        for line in input:
            trim_line = [x.strip().strip('.') for x in line]
            if len(line) != 15 or "?" in set(trim_line):
                continue
            self.valid_input.append(trim_line)



    @call
    def learn(self):
        self.X = {}
        for row in self.valid_input:
            if (row[-1] not in self.X):
                self.X[row[-1]] = []
            self.X[row[-1]].append(row[:-1])

        Y = list(self.X.keys())

        # i is index of sample (row of X)
        # j is index of attribute (column of X)
        # k is value of y

        # for each class 0 / 1
        for y in Y:
            self.prior[y] = float(len(self.X[y])) / len(self.valid_input)
            xk = list(zip(*self.X[y]))
            # for each attribute
            for j in range(len(self.attr)):

                if self.attr[j] in self.continue_attr.keys():
                    mean = sum([float(xijk) for xijk in xk[j]]) / float(len(xk[j]))
                    self.continue_attr[self.attr[j]] = {
                        "mean": mean,
                        "std": sqrt(sum([pow(float(xijk)-mean, 2) for xijk in xk[j]]) /float(len(xk[j]))),
                        # "count": len(xk[j]),
                    }
                    # print(self.attr[j],"\t", self.continue_attr[self.attr[j]])
                else:
                    # print(self.attr[j])
                    for xijk in xk[j]:
                        if xijk not in self.discrete_attr[self.attr[j]]:
                            self.discrete_attr[self.attr[j]][xijk] = 1
                        else:
                            self.discrete_attr[self.attr[j]][xijk] += 1
                    for key in self.discrete_attr[self.attr[j]]:
                        self.discrete_attr[self.attr[j]][key] /= len(xk[j])
                        # print(key, self.discrete_attr[self.attr[j]][key], end='\t')
                # print("\n", s, '\n')
            self.parameters[y] = {
                'discrete': copy.deepcopy(self.discrete_attr),
                'continuous': copy.deepcopy(self.continue_attr),
            }
        # self.discrete_attr = {}
        # self.continue_attr = {}
        self.X = {}

    @call
    def predict(self):
        self.X = {}
        for row in self.valid_input:
            if (row[-1] not in self.X):
                self.X[row[-1]] = []
            self.X[row[-1]].append(row[:-1])  # row[:] rather than row[:-1], retain original class.

        Y = list(self.X.keys())
        for y_actual in Y:
            # print(y_actual)
            cur = 0
            for xik in self.X[y_actual]:
                log_likelihood = {}
                for y in Y:
                    log_likelihood[y] = log(self.prior[y])
                    for j in range(len(xik)):
                        if self.attr[j] in self.parameters[y]['discrete']:
                            log_likelihood[y] += log(self.parameters[y]['discrete'][self.attr[j]][xik[j]])
                        else:
                            E = self.parameters[y]['continuous'][self.attr[j]]['mean']
                            S = self.parameters[y]['continuous'][self.attr[j]]['std']
                            # print(E, S, float(xik[j]))
                            log_likelihood[y] += log(
                                (1 / sqrt(2 * pi * (S ** 2 + 10 ** (-9))))
                            ) -(float(xik[j]) - E) ** 2 / 2 * (S ** 2 + 10 ** (-9))
                xik.append(Y[0] if log_likelihood[Y[0]]>log_likelihood[Y[1]] else Y[1])
                # print(xik)
                if cur<7:
                    self.log_likelihood_record[cur] = copy.deepcopy(log_likelihood)
                cur+=1

    @call
    def getAccuracy(self):
        self.accuracy = {}
        self.accuracy_total = 0.
        self.input_length = 0.

        Y = list(self.X.keys())

        for y in Y:
            self.accuracy[y] = 0.
            for xik in self.X[y]:
                if xik[-1] == y:
                    self.accuracy[y] +=1
            self.accuracy_total += self.accuracy[y]
            self.input_length += len(self.X[y])
            self.accuracy[y] /= len(self.X[y])
        self.accuracy_total /= self.input_length


    @call
    def report_prior(self):
        for y in self.prior:
            print(y, '\t', round(self.prior[y]),4)

    @call
    def report_attribute_parameter(self):
        for v in self.parameters:
            print('Class:\"', v, "\":")
            for attr in self.attr:
                sum = 0
                print('\t', attr, end=": ")
                if attr in self.parameters[v]['discrete']:
                    datatype = 'discrete'
                    # for value in self.discrete_attr[attr]:
                    #     print(value, "=", round(self.parameters[v][datatype][attr][value], 4), end=', ')
                    #     sum += self.parameters[v][datatype][attr][value]
                else:
                    datatype = 'continuous'
                    # for value in self.discrete_attr[attr]:
                    #     print(value, "=", round(self.parameters[v][datatype][attr][value], 4), end=', ')
                    #     sum += self.parameters[v][datatype][attr][value]
                # for attr in self.parameters[v][datatype]:


                for value in self.parameters[v][datatype][attr]:
                    print(value, "=", round(self.parameters[v][datatype][attr][value], 4), end=', ')
                    sum+= self.parameters[v][datatype][attr][value]
                # print('sum=', round(sum, 4))
                print()

    @call
    def report_likelihood_data(self):
        for record in self.log_likelihood_record:
            print("Log-likelihood of record", record+1, end=": \n")
            for y in self.log_likelihood_record[record]:
                print("\t", "Class", y, end=": ")
                # print(round(float(self.log_likelihood_record[record][y]), 4))
                print('%.4f' % round(float(self.log_likelihood_record[record][y]), 4))

    @call
    def report_accuracy(self):
        print("\nAccuracy")
        for y in self.accuracy:
            print(y, '\t', round(self.accuracy[y], 4),
                  # len(self.X[y])
                  )


@call
def run():
    nb = NaiveBayes()
    nb.attribute_processing()

    trainFile = 'data/C_ML/adult.data.csv'
    nb.getInput(trainFile)
    nb.learn()
    nb.report_prior()
    nb.report_attribute_parameter()

    testFile = 'data/C_ML/adult.test.csv'
    nb.getInput(testFile)
    nb.predict()
    nb.report_likelihood_data()

    nb.getInput(trainFile)
    nb.predict()
    nb.getAccuracy()
    nb.report_accuracy()

    nb.getInput(testFile)
    nb.predict()
    nb.getAccuracy()
    nb.report_accuracy()

    n, train_accuracy, test_accuracy = [], [], []
    for i in range(5, 14):
        n.append(int(pow(2,i)))
        nb.getInput(trainFile, int(pow(2,i)))
        nb.learn()
        nb.predict()
        nb.getAccuracy()
        train_accuracy.append(nb.accuracy_total)
        nb.getInput(testFile)
        nb.predict()
        nb.getAccuracy()
        test_accuracy.append(nb.accuracy_total)

    accuracy_plot = pd.DataFrame({
        'n':    n,
        'train':train_accuracy,
        'test': test_accuracy,
    })
    accuracy_plot.plot(x="n", y=['train', 'test'], grid=True, marker="o")
    plt.show()
    print("----end of run()----")

run()
print('Loaded data file {0} with {1} rows')

"""
安排训练集和测试集，横向清洗
内部定义逻辑，填充两个集合

把数据打竖，变成dict
分开 continuous 的和 discrete 的 

构建一个discrete的属性dict
discrete_attr = {}
for key in discrete_attr:
    attr_values = discrete_attr[key] 
    discrete_attr[key] = {}
    for value in attr_values:
        discrete_attr[key][value] = 0

对每一个continuous的dict obj
continuous: {'mean': 0, 'std':0}
"""


