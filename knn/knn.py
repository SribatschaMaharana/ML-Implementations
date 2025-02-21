import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def computeDistance(x, x_i):
    return np.sqrt(np.sum((x[:-1] - x_i[:-1]) ** 2))

def knn(x, dataset, k):
    distanceLabel = []
    for i in range(len(dataset)):
        distanceLabel.append((dataset[i], computeDistance(x, dataset[i])))

    distanceLabel.sort(key=lambda x: x[1])  # sort based on distance
    neighbors = [x[0] for x in distanceLabel[:k]]  # pick top k neighbors

    labels = {}
    for neighbor in neighbors:
        label = neighbor[-1]  # last column is label
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    majority = max(labels, key=labels.get)
    return majority


df = pd.read_csv('iris.csv', header=None)

#to convert to numpy array with same datatypes
df.replace(to_replace='Iris-setosa', value=0.0, inplace=True)
df.replace(to_replace='Iris-versicolor', value=1.0, inplace=True)
df.replace(to_replace='Iris-virginica', value=2.0, inplace=True)


trainedAccuracy = []
testedAccuracy = []
trainedStdDev = []
testedStdDev = []


for k in range(1, 52, 2):
  trainedAccs = []
  testedAccs = []

  for _ in range(20):

      df = shuffle(df)
      trainSet, testSet = train_test_split(df, test_size=0.2, random_state=3, shuffle=True)

      trainSetNormalized = trainSet.copy()
      testSetNormalized = testSet.copy()
      for idx in range(4):
          trainSetNormalized.iloc[:, idx] = (trainSet.iloc[:, idx] - trainSet.iloc[:, idx].min()) / (trainSet.iloc[:, idx].max() - trainSet.iloc[:, idx].min())
          testSetNormalized.iloc[:, idx] = (testSet.iloc[:, idx] - trainSet.iloc[:, idx].min()) / (trainSet.iloc[:, idx].max() - trainSet.iloc[:, idx].min())

      trainArr = trainSetNormalized.to_numpy()
      testArr = testSetNormalized.to_numpy()

      # computing accuracies
      trainCorrect = 0
      for instance in trainArr:
          prediction = knn(instance, trainArr, k)
          if prediction == instance[-1]:
              trainCorrect += 1
      curTrainAcc = trainCorrect / len(trainArr)
      trainedAccs.append(curTrainAcc)

      testCorrect = 0
      for instance in testArr:
          prediction = knn(instance, trainArr, k)
          if prediction == instance[-1]:
              testCorrect += 1
      curTestAcc = testCorrect / len(testArr)
      testedAccs.append(curTestAcc)

  # getting average and standard deviation
  trainAccAvg = sum(trainedAccs) / len(trainedAccs)
  testAccAvg = sum(testedAccs) / len(testedAccs)

  trainAccStd = np.std(trainedAccs)
  testAccStd = np.std(testedAccs)

  trainedAccuracy.append(trainAccAvg)
  testedAccuracy.append(testAccAvg)
  trainedStdDev.append(trainAccStd)
  testedStdDev.append(testAccStd)


#plotting for Q1.1 and Q1.2
kVals = list(range(1, 52, 2))

plt.errorbar(kVals, trainedAccuracy, yerr=trainedStdDev, label='Average Training Accuracy', fmt='-o', capsize=5)
plt.title('Average Accuracy of k-NN Model on Training Set')
plt.xlabel('k values')
plt.ylabel('Average Training Accuracy per k')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the results for Q1.2
plt.errorbar(kVals, testedAccuracy, yerr=testedStdDev, label='Average Testing Accuracy', fmt='-o', capsize=5)
plt.title('Average Accuracy of k-NN Model on Testing Set')
plt.xlabel('k')
plt.ylabel('Average Testing Accuracy per k')
plt.legend()
plt.grid(True)
plt.show()