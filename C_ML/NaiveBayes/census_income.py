from math import *
import csv
import copy
import numpy
import pandas as pd
import matplotlib.pyplot as plt


"""
AndrewID: kaichenc; name: Kaichen Chen
 
Run the run() to learn, predict, 
print data including prior / parameters / 
accuracy, and plot the relation between
number-of-sample and accuracy.

Please read run() first to see the 
order of primary functions. 

data files dir: ../../data/C_ML

@ line
344: def run()
412: calling run()
"""


# Only a function for testing. Is ignored.
def call(func):
    def wrapper(*args, **kw):
        # print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

def norm_pdf(x: float=0.0, mean: float=0.0, std: float=1.0, log_likelihood:bool=False)-> float:
    denorm = 1 / sqrt(2 * pi * pow(std, 2))
    power = - pow(x - mean, 2) / (2 * pow(std, 2))

    if log_likelihood:
        return log(denorm) + power
    else:
        return denorm * exp(power)



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

    """
    Dictionary is most frequently used.
    """
    valid_input = []
    X = {}
    prior = {}
    parameters = {}
    log_likelihood_record = {}
    accuracy = {}
    accuracy_total = 0.0
    input_length = 0
    decision = {}

    """
    Sort attributes and their value 
    in the order specified by the
    website.
    """
    @call
    def attribute_processing(self):
        for key in self.discrete_attr:
            values_count = {}
            for value in self.discrete_attr[key]:
                values_count[value.strip()] = 0
            self.discrete_attr[key] = values_count

        for key in self.continue_attr:
            self.continue_attr[key] = {}

    """
    Read data from data source,
    specify (optional) number of line to read,
    and clean data.
    
    Data store in self.valid_input = [].
    
    Each call will clear self.valid_input.
    """
    @call
    def getInput(self, filename:str, length: int=0):
        self.valid_input = []
        fileContent = csv.reader(open(filename, 'r'))

        input = list(fileContent)
        # print(length, filename)
        if length !=0 and length<=len(input):
            input = input[0:length]

        for line in input:
            trim_line = [x.strip().strip('.') for x in line]
            if len(line) != 15 or "?" in set(trim_line):
                continue
            self.valid_input.append(trim_line)
        # print(self.valid_input[0:10])


    """
    Estimate parameters.
    
    Store in 
        self.parameters 
        = {
            '<=50K': {
                'continuous':{attributes: { value: {likelihood} }}
                'discrete':  {attributes: { value: {{'mean': ll, 'std': ll}} }}
            },
            '>=50K': {
                'continuous':{attributes: { value: {likelihood} }}
                'discrete':  {attributes: { value: {{'mean': ll, 'std': ll}} }}
            },
        }
    """
    @call
    def learn(self):
        self.X = {}
        for row in self.valid_input:
            if (row[-1] not in self.X):
                self.X[row[-1]] = []
            self.X[row[-1]].append(row[:-1])

        Y = list(self.X.keys())

        """
        # i is index of sample (row of X)
        # j is index of attribute (column of X)
        # k is value of y
        """
        # for each class 0 / 1
        for y in Y:
            self.prior[y] = float(len(self.X[y])) / len(self.valid_input)
            xk = list(zip(*self.X[y]))
            # for each attribute
            for j in range(len(self.attr)):
                """
                Check whether attribute is continuous or discrete,
                and apply different estimation accordingly. 
                """
                if self.attr[j] in self.continue_attr.keys():
                    mean = sum([float(xijk) for xijk in xk[j]]) / float(len(xk[j]))
                    self.continue_attr[self.attr[j]] = {
                        "mean": mean,
                        "std": sqrt(
                            sum([pow(float(xijk)-mean, 2) for xijk in xk[j]]) /float(len(xk[j])-1)
                        ),
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


    """
    Calculate likelihood of each new sample,
    classify sample using decision boundary
    """
    @call
    def predict(self):
        self.X = {}
        for row in self.valid_input:
            if (row[-1] not in self.X):
                self.X[row[-1]] = []
            self.X[row[-1]].append(row[:-1])

        Y = list(self.X.keys())

        log_record={}



        for y_actual in Y:
            cur = 0
            for xik in self.X[y_actual]:
                """"""
                # print('\n xik: ', xik)
                log_likelihood = {}
                for y in Y:
                    log_record[y] = {}
                    # print(y)
                    log_likelihood[y] = log(self.prior[y])
                    for j in range(len(xik)):
                        if self.attr[j] in self.parameters[y]['discrete']:
                            # print(self.parameters[y]['discrete'][self.attr[j]][xik[j]])
                            likelihood = self.parameters[y]['discrete'][self.attr[j]][xik[j]]
                            """
                                No smooth, just make log(0) negative infinity
                            """
                            if likelihood == 0:
                                log_likelihood[y] += float('-inf')
                                log_record[y][self.attr[j]] = float('-inf')
                            else:
                                log_likelihood[y] += log(likelihood)
                                log_record[y][self.attr[j]] = log(likelihood)
                        else:
                            # print('xik parameters:', xik)
                            # for key in self.parameters[y]['continuous'][self.attr[j]]:
                            #     print(round(self.parameters[y]['continuous'][self.attr[j]][key],4), end=' ')

                            # print(self.attr[j])
                            E = self.parameters[y]['continuous'][self.attr[j]]['mean']
                            S = self.parameters[y]['continuous'][self.attr[j]]['std']
                            temp = norm_pdf(x = float(xik[j]), mean = E, std=(S+10.**(-9.)), log_likelihood=True)
                            log_likelihood[y] += temp
                            log_record[y][self.attr[j]] = temp
                xik.append(Y[0] if log_likelihood[Y[0]]>log_likelihood[Y[1]] else Y[1])
                """"""
                # print("\nlog_record", log_record)
                if cur<7:
                    self.log_likelihood_record[cur] = copy.deepcopy(log_likelihood)
                    self.decision[cur] = xik[-1]
                    # print("Decision:", xik[-1])
                    # print('\t', log_likelihood)
                cur+=1

        return log_record

    """
    Calculate accuracy
    """
    @call
    def getAccuracy(self):
        self.accuracy = {}
        self.accuracy_total = 0.
        self.input_length = 0.

        for y in self.X.keys():
            self.accuracy[y] = 0.
            for xik in self.X[y]:
                if xik[-1] == y:
                    self.accuracy[y] +=1
            self.accuracy_total += self.accuracy[y]
            self.input_length += len(self.X[y])
            self.accuracy[y] /= len(self.X[y])
        self.accuracy_total /= self.input_length


    """
    All function start with 'report_'
    will print required information.
    """
    @call
    def report_prior(self):
        for y in self.prior:
            print(y, '\t', round(self.prior[y],4))

    @call
    def report_attribute_parameter(self):
        for v in self.parameters:
            print('Class:\"', v, "\":")
            for attr in self.attr:
                sum = 0
                print('\t', attr, end=": ")
                if attr in self.parameters[v]['discrete']:
                    datatype = 'discrete'
                else:
                    datatype = 'continuous'

                for value in self.parameters[v][datatype][attr]:
                    if value=='std':
                        print("var", "=", round((self.parameters[v][datatype][attr][value])**2, 4), end=', ')
                    else:
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
                print('%.4f' % round(float(self.log_likelihood_record[record][y]), 4))
            print(self.decision[record])

    @call
    def report_accuracy(self):
        print("\nAccuracy")
        for y in self.accuracy:
            print(y, '\t', round(self.accuracy[y], 4),
                  )
        print(self.accuracy_total)


@call
def run():
    nb = NaiveBayes()
    nb.attribute_processing()

    trainFile = 'data/C_ML/adult.data.csv'
    testFile = 'data/C_ML/adult.test.csv'

    """ 5.1 """
    nb.getInput(trainFile)
    nb.learn()
    ''' (a) '''
    nb.report_prior()
    ''' (b) '''
    nb.report_attribute_parameter()

    ''' (c) '''
    nb.getInput(testFile)
    nb.predict()
    nb.report_likelihood_data()

    """ 5.2 """
    nb.getInput(trainFile)
    nb.predict()
    nb.getAccuracy()
    ''' (a) '''
    nb.report_accuracy()

    ''' (b) '''
    nb.getInput(testFile)
    nb.predict()
    nb.getAccuracy()
    nb.report_accuracy()

    ''' (c) '''
    n, train_accuracy, test_accuracy = [], [], []
    base = 2.0
    for i in range(5, 14):
        n.append(int(pow(base, i)))
        # n.append(int(i))
        nb.getInput(trainFile, int(pow(base, i)))
        nb.learn()
        nb.predict()
        nb.getAccuracy()
        train_accuracy.append(nb.accuracy_total)
        nb.getInput(testFile)
        nb.predict()
        nb.getAccuracy()
        test_accuracy.append(nb.accuracy_total)

    accuracy_plot = pd.DataFrame({
        'n': n,
        'train': train_accuracy,
        'test': test_accuracy,
    })
    accuracy_plot.plot(x="n",
                       y=['train', 'test'],
                       # y='test',
                       grid=True, marker="o"
                       )
    plt.show()
    for i in range(5, 14):
        print(i, "%.4f%%" %train_accuracy[i - 5])
    for i in range(5, 14):
        print(i, "%.4f%%" %test_accuracy[i - 5])

    print("----end of run()----")

""" Program start """
run()





""" Ignore below draft """

@call
def draft():
    nb = NaiveBayes()
    nb.attribute_processing()

    trainFile = 'data/C_ML/adult.data.csv'
    testFile = 'data/C_ML/adult.test.csv'
    # nb.getInput(trainFile, int(pow(2,7)))
    nb.getInput(trainFile)
    nb.learn()

    nb.getInput(testFile)
    log_record = nb.predict()
    nb.report_likelihood_data()
    nb.getAccuracy()
    nb.report_prior()
    nb.report_accuracy()
    print(nb.accuracy_total)
    # print("log record: \n", log_record)

def draft2():
    nb = NaiveBayes()
    nb.attribute_processing()
    trainFile = 'data/C_ML/adult.data.csv'
    testFile = 'data/C_ML/adult.test.csv'
    n, train_accuracy, test_accuracy = [], [], []
    base = 1.3
    for i in range(9, 40):
    # for i in range(6, 13):
        n.append(int(pow(base, i)))
        # n.append(int(i))
        nb.getInput(trainFile, int(pow(base, i)))
        nb.learn()
        nb.predict()
        nb.getAccuracy()
        train_accuracy.append(nb.accuracy_total)
        nb.getInput(testFile)
        nb.predict()
        nb.getAccuracy()
        test_accuracy.append(nb.accuracy_total)

    accuracy_plot = pd.DataFrame({
        'n': n,
        'train': train_accuracy,
        'test': test_accuracy,
    })
    accuracy_plot.plot(x="n",
                       y=['train', 'test'],
                       # y='test',
                       grid=True, marker="o"
                       )
    plt.show()
    print("----end of run()----")

# draft2()

# print('Loaded data file {0} with {1} rows')

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


