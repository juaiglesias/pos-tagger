import json
import math
import numpy as np
import kenlm

def countSet (dataset):
    sentencesCount = len(dataset)
    wordsCount = 0
    for words, _ in dataset:
        for _ in words:
            wordsCount+=1
    return (sentencesCount, wordsCount)       

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

def merge (dict3gram1, dict3gram2):
    mergDict = {}
    for (key, value) in dict3gram1.items():
        mergDict[key] = value + dict3gram2.get(key,0)
    for (key, value) in dict3gram2.items():
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
        kl+= ptest * math.log10(ptest/ptrain)
    return kl

"""
model = kenlm.Model('model.arpa')
print(model.score('this is a sentence .', bos = True, eos = True))
"""

train_sets = ["ewt", "gum", "lines", "partut"]
test_sets = ["ewt", "foot", "gum", "lines", "natdis", "partut", "pud"]

"""
#SHOW COUNT OF WORDS AND SENTENCES OF EACH
for train_set in train_sets:
    fileset = json.load(open("data/en/en."+train_set+".train.json"))
    print(train_set,"train:",countSet(fileset))

for test_set in test_sets:
    fileset = json.load(open("data/en/en."+test_set+".test.json"))
    print(test_set,"test:",countSet(fileset))
"""

for trainsetname in train_sets:
    train_set = json.load(open("data/en/en."+trainsetname+".train.json"))
    for testsetname in test_sets:
        test_set = json.load(open("data/en/en."+testsetname+".test.json"))
        print("For", trainsetname,"train set and", testsetname, "test set")
        print("Out for vocabulary:", outOfVocabulary(test_set, train_set))
        #print("Kl-divergence:", klDivergence(train_set, test_set))
