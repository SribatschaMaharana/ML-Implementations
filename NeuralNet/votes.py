from neuralnetwork import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


df = pd.read_csv('hw3_house_votes_84.csv')
cols = df.columns
transformer = make_column_transformer(
(OneHotEncoder(), cols[0:16]),remainder='passthrough')
transformed = transformer.fit_transform(df)

dft = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

[class1, class2] = [shuffle(df[df["class"]==0]),shuffle(df[df["class"]==1])]
# # • seperate instances based on classes


structures = [[16, 2, 2], [16, 4, 2], [16, 8, 2], [16, 2, 2, 2], [16, 4, 4, 2], [16, 2, 4, 2]]
lamdas=[0.1, 0.5, 1.0]
alpha=1
folds = kfold([class1, class2])

curSet=folds.copy()
for i in range(len(curSet)):
    curSet[i] = shuffle(curSet[i])
    curSet[i] = (curSet[i]).to_numpy()

structAcc=[]
structF1=[]
for l in range(len(lamdas)):
    for s in range(len(structures)):
        structAcc=[]
        structF1=[]
        for i in range(10):
            trainingSet = curSet.copy()
            validationSet=trainingSet.pop(i)

            nn=NeuralNetwork(structures[s], lamdas[l], alpha)
            testX = validationSet[:, :-1]  # All columns except the last one
            testY = validationSet[:, -1]   
            testX = normalize(testX)        
            testY = vectorize(testY, 2)

            for k in range(30):

                for fold in trainingSet: #for each fold in the training set,
                    x = fold[:, :-1]       #ATTRIBUTES=rest all columns
                    y = fold[:, -1]        #LABELS=last column
                    trainX = normalize(x)  # normalize attributes
                    trainY = vectorize(y, 2)  # Convert y into a list of vectors

                    
                    grads = nn.backProp(trainX,trainY)
                    dec = nn.descent(grads)

            [cost,predictions] = nn.costFunction(testX, testY)
            [acc,f1] = confusion_matrix_cat(testY,predictions)

            structAcc.append(acc)
            structF1.append(f1)

        ModelAccuracy = np.sum(structAcc)/10
        f1score=np.sum(structF1)/10

        print("[structure=",structures[s],"lamda=",lamdas[l], "] Model Accuracy= ", ModelAccuracy," Model F1 Score= ",f1score)

                
            

            