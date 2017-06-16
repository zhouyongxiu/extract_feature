# coding=utf-8
import sys 
import numpy as np
from matplotlib import pyplot as plt
#import cv2
import os
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
global PERSON_NUM
PERSON_NUM=1668


#Before training,the mean must be substract

def calculate_cosin(data0,data1,num):
    distance = []
    for i in range(num):
        cosV12 = np.dot(data0[i],data1[i])/(np.linalg.norm(data0[i])*np.linalg.norm(data1[i]))
        distance.append(cosV12)
    return distance
    
def Verify(A, G, x1, x2):

    x1.shape = (-1,1)
    x2.shape = (-1,1)
    ratio = np.dot(np.dot(np.transpose(x1),A),x1) + np.dot(np.dot(np.transpose(x2),A),x2) - 2*np.dot(np.dot(np.transpose(x1),G),x2)
    return float(ratio)

def calculate_JB(data0,data1,num):
    distance = []
    A =  np.load('.\\feature_data_train\\A.npy')
    G =  np.load('.\\feature_data_train\\G.npy')
    for i in range(num):
        ratio = Verify(A, G, data0[i], data1[i])
        distance.append(ratio)
    return distance

if __name__=="__main__":
    data0 = np.load('.\\feature_data_test\\data0.npy')
    data1 = np.load('.\\feature_data_test\\data1.npy')
    distance = calculate_JB(data0,data1,PERSON_NUM)
    
    x = range(PERSON_NUM)
    plt.figure()
    plt.plot(x, distance,'ro')
    
    data0 = np.load('.\\feature_data_test_match\\data0.npy')
    data1 = np.load('.\\feature_data_test_match\\data1.npy')
    distance = calculate_JB(data0,data1,PERSON_NUM)
    
    x = range(PERSON_NUM)
    plt.figure()
    plt.plot(x, distance,'bo')
    
    
    
    pass
    