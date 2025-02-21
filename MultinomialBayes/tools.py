from utils import *
import math as M
epsilon  = 1e-6


def ncount(posTrain, negTrain):

    poswords = {}
    negwords = {}

    for doc in posTrain:
        for word in doc:
            if word in poswords:
                poswords[word] += 1
            else:
                poswords[word] = 1

    for doc in negTrain:
        for word in doc:
            if word in negwords:
                negwords[word] += 1
            else:
                negwords[word] = 1

    return (poswords, negwords)

def naive_bayes(posTrain, negTrain, dic, doc):

    (poswords, negwords) = dic

    prpos = len(posTrain)/(len(posTrain)+len(negTrain))
    prneg = len(negTrain)/(len(posTrain)+len(negTrain))


    possum = sum(poswords.values())
    negsum = sum(negwords.values())
    for word in doc:
        if word not in poswords:
            prpos *= 0
        else:
            prpos *= poswords[word]/ possum

        if word not in negwords:
            prneg *= 0
        else:
            prneg *= negwords[word]/ negsum

    if (prpos == prneg):
        flip = random.randint(0, 1) #toincoss for equal probability
        if (flip == 1):
            return 1
        else:
            return 0
    if (prpos > prneg):
        return 1 
    else:
        return 0

def naive_log(posTrain, negTrain, dic, doc):

    (poswords, negwords) = dic

    prpos = M.log(len(posTrain)/(len(posTrain)+len(negTrain)))
    prneg = M.log(len(negTrain)/(len(posTrain)+len(negTrain)))
    possum = sum(poswords.values())
    negsum = sum(negwords.values())
    for word in doc:
        if word not in poswords:
            prpos += M.log(epsilon)
        else:
            prpos +=  M.log(poswords[word]/ (possum))

        if word not in negwords:
            prneg += M.log(epsilon)
        else:
            prneg += M.log(negwords[word]/ (negsum))

    if (prpos == prneg):
        flip = random.randint(0, 1) #toincoss for equal probability
        if (flip == 1):
            return 1
        else:
            return 0
    if (prpos > prneg):
        return 1 
    else:
        return 0
    
def naive_smooth(posTrain, negTrain, dic, doc, alpha, vocab):

    (poswords, negwords) = dic

    prpos = M.log(len(posTrain)/(len(posTrain)+len(negTrain)))
    prneg = M.log(len(negTrain)/(len(posTrain)+len(negTrain)))

    possum = sum(poswords.values())
    negsum = sum(negwords.values())

    prposnumlist = []
    prnegnumlist = []
    posAlpha = False
    negAlpha = False

    for word in doc:
        if word in poswords:
            prposnumlist.append(poswords[word])
        else:
            posAlpha = True
            prposnumlist.append(0)

        if word in negwords:
            prnegnumlist.append(negwords[word])
        else:
            negAlpha = True
            prnegnumlist.append(0)

    
    for num in prposnumlist:
            if posAlpha == True:
                prpos += M.log((num + alpha) / (possum + (alpha * len(vocab))))
            else : 
                prpos += M.log(num / possum)\
                
    for num in prnegnumlist:
            if negAlpha == True:
                prneg += M.log((num + alpha) / (negsum + (alpha * len(vocab))))
            else : 
                prneg += M.log(num / negsum)

    return (1 if prpos > prneg else 0)

    
def confusion_matrix(tp, tn, pos_test, neg_test):
    fn = len(pos_test) - tp
    fp = len(neg_test) - tn

    print(f"tp: {tp} fn: {fn}")
    print(f"fp: {fp} tn: {tn}")











    
