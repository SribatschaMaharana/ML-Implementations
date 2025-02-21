import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
random.seed(60)


def kfoldNum(classes):
  class1, class2, class3 = classes[0], classes[1], classes[2]
  c1ratio = len(class1) // 10
  c2ratio = len(class2) // 10
  c3ratio = len(class3) // 10
  c1rem = len(class1) % 10
  c2rem = len(class2) % 10
  c3rem = len(class3) % 10

  karray = []

  for i in range(10):
      fold = pd.concat([class1.iloc[i * c1ratio: (i + 1) * c1ratio], class2.iloc[i * c2ratio: (i + 1) * c2ratio], class3.iloc[i * c3ratio: (i + 1) * c3ratio]])
      karray.append(fold)

  for j in range(c1rem):
      fold_idx = j * (10 // c1rem)
      karray[fold_idx] = pd.concat([karray[fold_idx], class1.iloc[(c1ratio * 10) + j: (c1ratio * 10) + j + 1]])

  for k in range(c2rem):
      fold_idx = k * (10 // c2rem)
      karray[fold_idx] = pd.concat([karray[fold_idx], class2.iloc[(c2ratio * 10) + k: (c2ratio * 10) + k + 1]])
  for l in range(c3rem):
      fold_idx = l * (10 // c3rem)
      karray[fold_idx] = pd.concat([karray[fold_idx], class3.iloc[(c3ratio * 10) + l: (c3ratio * 10) + l + 1]])

  return karray
def kfold(classes):
  class1, class2 = classes[0], classes[1]
  c1ratio = len(class1) // 10
  c2ratio = len(class2) // 10
  c1rem = len(class1) % 10
  c2rem = len(class2) % 10

  karray = []

  for i in range(10):
    fold = pd.concat([class1.iloc[i * c1ratio: (i + 1) * c1ratio], class2.iloc[i * c2ratio: (i + 1) * c2ratio]])
    karray.append(fold)

  for j in range(c1rem):
      fold_idx = j * (10 // c1rem)
      karray[fold_idx] = pd.concat([karray[fold_idx], class1.iloc[(c1ratio * 10) + j: (c1ratio * 10) + j + 1]])

  for k in range(c2rem):
      fold_idx = k * (10 // c2rem)
      karray[fold_idx] = pd.concat([karray[fold_idx], class2.iloc[(c2ratio * 10) + k: (c2ratio * 10) + k + 1]])

  return karray

def normalize(arr):
    # Find the maximum and minimum values of each feature (column)
    max_vals = np.amax(arr, axis=0)
    min_vals = np.amin(arr, axis=0)
    
    # Normalize each feature using min-max normalization
    normalized_array = (arr - min_vals) / (max_vals - min_vals)
    
    return normalized_array

def vectorize(labels, numClasses):
    vectoredLabels=[]
    for label in labels:
        vector = [0]*(numClasses)
        vector[int(label)-1]=1
        vectoredLabels.append(vector)
    return vectoredLabels

def argmax(arr):
    max_index = 0
    max_value = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_index = i
    return max_index


def confusion_matrix_num(trueVals, predictions):
    matrix = np.zeros((3,3))
    for i in range(len(predictions)) :
       
        matrix[np.argmax(trueVals[i])][np.argmax(predictions[i])] += 1

    acc = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / (len(predictions))

    rec1 = (matrix[0][0]) / (matrix[0][0] + matrix[0][1] + matrix[0][2])
    rec2 = (matrix[1][1]) / (matrix[1][1] + matrix[1][0] + matrix[1][2])
    rec3 = (matrix[2][2]) / (matrix[2][2] + matrix[2][0] + matrix[2][1])
    recall = (rec1 + rec2 + rec3) / 3

    if((matrix[0][0] + matrix[1][0] + matrix[2][0]) == 0 ):
       prec1 = 0
    else :
       prec1 = (matrix[0][0]) / (matrix[0][0] + matrix[1][0] + matrix[2][0])

    if((matrix[1][1] + matrix[0][1] + matrix[2][1]) == 0) :
       prec2 = 0
    else : 
       prec2 = (matrix[1][1]) / (matrix[1][1] + matrix[0][1] + matrix[2][1])
    
    if((matrix[2][2] + matrix[0][2] + matrix[1][2]) == 0) :
       prec3 = 0
    else :
       prec3 = (matrix[2][2]) / (matrix[2][2] + matrix[0][2] + matrix[1][2])
    
    precision = (prec1+prec2+prec3) / 3

    f1 = 2* ((precision * recall) / (precision + recall))
    return [acc, f1]


def confusion_matrix_cat(trueVals, preds):
    matrix = np.zeros((2, 2))
    for i in range(len(preds)):
        matrix[np.argmax(trueVals[i])][np.argmax(preds[i])] += 1

    acc = (matrix[0][0] + matrix[1][1]) / len(preds)

    if (matrix[0][0] + matrix[1][0]) == 0:
        prec = 0
    else:
        prec = matrix[0][0] / (matrix[0][0] + matrix[1][0])

    if (matrix[0][0] + matrix[0][1]) == 0:
        rec = 0
    else:
        rec = matrix[0][0] / (matrix[0][0] + matrix[0][1])

    if prec + rec == 0:
        f1 = 0
    else:
        f1 = 2 * ((prec * rec) / (prec + rec))

    return [acc, f1]