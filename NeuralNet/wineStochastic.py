from neuralnetwork import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_csv('hw3_wine.csv', sep='\t')
df=shuffle(df)
df=df.to_numpy()
trainSet, testSet = train_test_split(df, test_size=0.2, random_state=3, shuffle=True)
#seperate instances based on class

structure = [13, 8, 3]
lamda=0.0
alpha=0.5
 
nn=NeuralNetwork(structure, lamda, alpha)
testX=testSet[:, 1:]
testY=testSet[:, 0]
testX = normalize(testX) #normalize attributes
testY = vectorize(testY,3) #convert y into a list of vectors

x = trainSet[:, 1:] #ATTRIBUTES=rest all columns
y = trainSet[:, 0]  #LABELS=first column
trainX = normalize(x) #normalize attributes
trainY = vectorize(y,3) #convert y into a list of vectors

m=len(trainX)
costs = []
for i in range(len(trainSet)):
    grads = nn.backProp([trainX[i]],[trainY[i]])
    dec = nn.descent(grads)

    predict=nn.forward_propagation(trainX[i])
    #cost_i= np.sum(-np.array(trainY[i]) * np.log(predict) - (1-np.array(trainY[i]))*np.log(1-predict))
    [cost,predictions]=nn.costFunction([trainX[i]],[trainY[i]])
    costs.append(cost)

plt.plot(range(1, len(costs) + 1), costs)
plt.xlabel('Number of instances processed')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.show()