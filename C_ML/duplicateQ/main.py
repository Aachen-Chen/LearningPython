from math import *
import csv
import copy
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import WordPunctTokenizer
import nltk
import re


def call(func):
    def wrapper(*args, **kw):
        # print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


@call
def getInput(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df

@call
def getInputArray(filename:str) -> np.ndarray:
    """
    Watch for delimiter.
    This method is deprecated.
    :param filename:
    :return:
    """
    fileContent = csv.reader(open(filename, 'r', encoding="utf8"), delimiter =',')
    # input = np.array([list(map(float, row)) for row in list(fileContent)])
    fileContent = list(fileContent)
    # for items in fileContent[:3]:
    #     print(items,"\n")
    input = np.array([list(row) for row in fileContent])
    # for items in input[:3]:
    #     print(items,"\n")
    # print(type(input))
    n, p = np.shape(input)
    print('Input size:{0}, {1}'.format(n, p))
    return input

@call
def tokenizeRowSimple(s:str)->list:
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    subbed = pat_letter.sub(' ', s).strip().lower()
    result = WordPunctTokenizer().tokenize(subbed)

    return result

@call
def tokenizeRow(s:str, remove_stopwords=True)->str:
    # pat_letter = re.compile(r'[^a-zA-Z \']+')
    # subbed = pat_letter.sub(' ', s).strip().lower()
    # result = WordPunctTokenizer().tokenize(subbed)

    words = s.strip().lower().split()
    if remove_stopwords:
        stops = set(nltk.corpus.stopwords.words("english"))
        words = [w for w in words if not w in stops]

    s = "".join(words)

    s = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", s)
    s = re.sub(r"\'s", " 's ", s)
    s = re.sub(r"\'ve", " 've ", s)
    s = re.sub(r"n\'t", " 't ", s)
    s = re.sub(r"\'re", " 're ", s)
    s = re.sub(r"\'d", " 'd ", s)
    s = re.sub(r"\'ll", " 'll ", s)
    s = re.sub(r",", " ", s)
    s = re.sub(r"\.", " ", s)
    s = re.sub(r"!", " ", s)
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    s = re.sub(r"\?", " ", s)
    s = re.sub(r"\s{2,}", " ", s)

    words = s.split()
    stemmer = nltk.stem.SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    s = "".join(words)

    return s


def processDataFrame(target:list, source:pd.DataFrame, targetName:str):
    for row in source:
        target.append(tokenizeRow(row))
        if len(target)%10000 == 0:
            progress = len(target)/len(source) * 100
            print("Processing {0}: {1}%".format(targetName, round(progress, 1)))


@call
def write_WordCount(wordCount:dict):
    resultFile = 'results.txt'
    f = open(resultFile, 'w')
    i=0
    for key in wordCount:
        i = i+1
        # f.write(str(key) +': ' + str(wordCount[key]))
        f.write(str(i) +'\t'+ str(key))
        f.write('\n')
    return


@call
def run():
    dataFolder = '../../data/C_ML/duplicateQ/'
    # df = getInput(dataFolder + 'questions_sample.csv')
    df = getInput(dataFolder + 'questions.csv')
    df = df.dropna()

    # print(df.head())
    # print(df.question1)

    nonEnglish = []
    for row in df.question1:
        # print(str(row))

        # patternBase = re.compile(r"[^A-Za-z0-9()]")
        puntuation = "(),!.?\'\s\-:`’\"&/$…%\[\]^#@{}+-=_\<\>—~*|"
        math = "°"
        chinesePun = "“”？‘’"
        patternBase = r"[^A-Za-z0-9"+\
                      puntuation+\
                      math+\
                      chinesePun+\
                      "]"

        pos = re.search(patternBase, row)
        if pos != None:
            # print(pos.span())
            nonEnglish.append([str(row), row[pos.span()[0]]])

    for i in nonEnglish[:1000]:
        print(nonEnglish.index(i), i)
    print(len(nonEnglish))
    print(len(nonEnglish) / len(df.question1))

    return


@call
def draft():

    # s = "What is the \"step by step guide \" to invest in share market in india?"

    # pat_letter = re.compile(r'[^a-zA-Z \']+')
    # subbed = pat_letter.sub(' ', s).strip().lower()
    # print(subbed)
    #
    # result = WordPunctTokenizer().tokenize(subbed)
    # print(result)
    # # result2 = s.split()
    # # print(result2)
    #
    # freq1 = collections.Counter(result)
    # print(freq1)
    # # freq2 = collections.Counter(result2)
    # # print(freq2)

    # resultFile = 'results.txt'
    # f = open(resultFile, 'w')
    # for key in freq1:
    #     f.write(str(key)+': '+str(freq1[key]))
    #     f.write('\n')

    dataFolder  = '../../data/C_ML/duplicateQ/'
    # XTrain = getInputArray(dataFolder + 'questions_sample.csv')
    XTrain = getInputArray(dataFolder + 'questions.csv')
    #
    # XTrain = getInputArray(dataFolder + 'Xtrain.txt')
    # yTrain = getInputArray(dataFolder + 'ytrain.txt')
    # XTest  = getInputArray(dataFolder + 'Xtest.txt')
    # yTest  = getInputArray(dataFolder + 'ytest.txt')
    #
    # w, b, round, isClassified = train(X=XTrain, y=yTrain, eta=1, numIter=100)
    #
    # print("Training set distribution")
    # unique, counts = np.unique(yTrain, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print("Testing set distribution")
    # unique, counts = np.unique(yTest, return_counts=True)
    # print(dict(zip(unique, counts)))
    #
    # print('Train Accuracy: {0}, Test Accuracy: {1}'
    #       .format(accuracy(XTrain, yTrain, w, b),
    #               accuracy(XTest, yTest, w, b)
    #               )
    #       )
    wordCount = {}
    for i in range(len(XTrain)):
        # print(XTrain[i])
        for j in range(3, 5):
            # print(XTrain[i][j])
            words = tokenizeRowSimple(XTrain[i][j])
            count = collections.Counter(words)
            for word in count:
                if word in wordCount:
                    wordCount[word] += count[word]
                else:
                    wordCount[word] = count[word]
    wordCount = sorted(wordCount.items(), key=lambda pair: pair[1], reverse=True)
    write_WordCount(dict(wordCount))

# main()
draft()



