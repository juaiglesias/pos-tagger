import json
import math
import numpy as np
import kenlm
import matplotlib.pyplot as plt

def countSet (dataset):
    sentencesCount = len(dataset)
    wordsCount = 0
    differentwords = len(getWords(dataset))
    for words, _ in dataset:
        for _ in words:
            wordsCount+=1
    return (sentencesCount, wordsCount, differentwords)       

def getWords (dataset):
    return set(word for words, labels in dataset for word in words)
        
def outOfVocabulary (test_set, train_set):
    testWords = getWords(test_set)
    trainWords = getWords(train_set)
    resultWords = testWords.difference(trainWords)
    return len(resultWords) / len(testWords)

def get3grams (word):
    grams = []
    for x in range(0,len(word)-2):
        grams.append(word[x]+word[x+1]+word[x+2])
    return grams

def getAll3grams (dataset):
    words = getWords(dataset)
    dict3gram = {}
    for word in words:
        grams = get3grams(word)
        for gram in grams:
            if gram in dict3gram:
                dict3gram[gram]+=1
            else:
                dict3gram[gram]=1
    return dict3gram

def merge (dict1, dict2):
    mergDict = {}
    for (key, value) in dict1.items():
        mergDict[key] = value + dict2.get(key,0)
    for (key, value) in dict2.items():
        if key not in mergDict:
            mergDict[key]= value
    return mergDict

def probability (set_grams, gram, v, n):
    number3grams = sum(set_grams.values())
    return (set_grams.get(gram, 0) + 1)/(n + v * number3grams) 
 
def klDivergence (train_set, test_set):
    train3grams = getAll3grams(train_set)
    test3grams = getAll3grams(test_set)
    all3grams = merge(train3grams, test3grams)
    v = len(all3grams)
    n = sum(all3grams.values())
    kl = 0
    for gram in all3grams:
        ptrain = probability (train3grams, gram, v, n)
        ptest = probability (test3grams, gram, v, n)
        kl+= ptest * math.log(ptest/ptrain)
    return kl

def drawGraph(xvalues, yvalues, xlabel, ylabel, graphtitle):
    plt.bar(xvalues, yvalues)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(xvalues, fontsize=10, rotation=30)
    plt.title(graphtitle)
    plt.show()
"""
model = kenlm.Model('model.arpa')
print(model.score('this is a sentence .', bos = True, eos = True))
"""

train_sets = ["ewt", "gum", "lines", "partut"]
test_sets = ["ewt", "foot", "gum", "lines", "partut", "pud"]

"""
#SHOW COUNT OF WORDS AND SENTENCES OF EACH
for train_set in train_sets:
    fileset = json.load(open("data/en/en."+train_set+".train.json"))
    print(train_set,"train:",countSet(fileset))

for test_set in test_sets:
    fileset = json.load(open("data/en/en."+test_set+".test.json"))
    print(test_set,"test:",countSet(fileset))
"""


labels = []
percentages = []
divergences = []
for trainsetname in train_sets:
    train_set = json.load(open("data/en/en."+trainsetname+".train.json"))
    for testsetname in test_sets:
        test_set = json.load(open("data/en/en."+testsetname+".test.json"))
        labels.append("("+trainsetname+","+testsetname+")")
        percentages.append(round(outOfVocabulary(test_set, train_set) * 100, 2))
        divergences.append(klDivergence(train_set, test_set))
"""
#Draw OOV percentages
drawGraph(labels, percentages, '(Train set, test set)', 'Percentage of OOV words (%)', 'Percentage of Out of Vocabulary words for each combination of Train and Test sets')
"""

#Draw divergences
drawGraph(labels, divergences, '(Train set, test set)', 'kl-Divergence', 'kl-Divergences for each combination of Train and Test sets')