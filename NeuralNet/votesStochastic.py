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

dft=shuffle(dft)
dft=dft.to_numpy()
trainSet, testSet = train_test_split(dft, test_size=0.2, random_state=3, shuffle=True)
#seperate instances based on class

structure = [16, 4, 2]
lamda=0.0
alpha=0.5
 
nn=NeuralNetwork(structure, lamda, alpha)
testX = testSet[:, :-1]  # All columns except the last one
testY = testSet[:, -1]   
testX = normalize(testX)        
testY = vectorize(testY, 2)


x = trainSet[:, :-1]       #ATTRIBUTES=rest all columns
y = trainSet[:, -1]        #LABELS=last column
trainX = normalize(x)  # normalize attributes
trainY = vectorize(y, 2)  # Convert y into a list of vectors

m=len(trainX)
costs = []
for i in range(len(trainSet)):

    grads = nn.backProp(trainX[i],trainY[i])
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