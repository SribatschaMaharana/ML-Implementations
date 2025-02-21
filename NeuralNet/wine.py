from neuralnetwork import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_csv('hw3_wine.csv', sep='\t')
[class1, class2, class3] = [shuffle(df[df["# class "]==1], random_state=43),shuffle(df[df["# class "]==2], random_state=44),shuffle(df[df["# class "]==3], random_state=45)]
#seperate instances based on class


structures = [[13, 2, 3], [13, 4, 3], [13, 8, 3], [13, 2, 2, 3], [13, 4, 4, 3], [13, 2, 4, 3]]
lamdas=[0.1, 0.5, 1.0]
alpha=1
folds = kfoldNum([class1, class2, class3])
curSet=folds.copy()
for i in range(len(curSet)):
    curSet[i] = shuffle(curSet[i])
    curSet[i] = (curSet[i]).to_numpy()



for l in range(len(lamdas)):
    for s in range(len(structures)):
        structAcc=[]
        structF1=[]
        for i in range(10):
            trainingSet = curSet.copy()
            validationSet=trainingSet.pop(i)

            nn=NeuralNetwork(structures[s], lamdas[l], alpha)
            testX=validationSet[:, 1:]
            testY=validationSet[:, 0]
            testX = normalize(testX) #normalize attributes
            testY = vectorize(testY,3) #convert y into a list of vectors

            for k in range(30):

                for fold in trainingSet: #for each fold in the training set,
                    x = fold[:, 1:] #ATTRIBUTES=rest all columns
                    y = fold[:, 0]  #LABELS=first column
                    trainX = normalize(x) #normalize attributes
                    trainY = vectorize(y,3) #convert y into a list of vectors
                    
                    grads = nn.backProp(trainX,trainY)
                    dec = nn.descent(grads)
            [cost,predictions] = nn.costFunction(testX, testY)

            [acc,f1] = confusion_matrix_num(testY,predictions)
            structAcc.append(acc)
            structF1.append(f1)

        ModelAccuracy = np.sum(structAcc)/10
        f1score=np.sum(structF1)/10

        print("[structure=",structures[s],"lamda=",lamdas[l], "] Model Accuracy= ", ModelAccuracy," Model F1 Score= ",f1score)

        


