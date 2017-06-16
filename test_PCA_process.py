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
PERSON_NUM=10


def data_pre(data):
    
    for i in range(1668):
        for j in range(4000):
            if (data[i][j] < 0):
                data[i][j] = 0
    data = np.sqrt(data)
    data = np.divide(data, np.repeat(np.sum(data, 1), data.shape[1]).reshape(data.shape[0], data.shape[1]))
    return data
    
if __name__=="__main__":
    
    features0 = np.load('.\\feature_data_test\\features0.npy')
    features0 = data_pre(features0)
    #np.save('.\\feature_data\\data_pro0.npy',features0)
    
    features1 = np.load('.\\feature_data_test\\features1.npy')
    features1 = data_pre(features1)
    #np.save('.\\feature_data\\data_pro1.npy',features1)
    #data_pca = PCA_pro(features)
    clt_pca = joblib.load(".\\feature_data_train\\pca_model.m")
    
    data0 = clt_pca.transform(features0)
    data1 = clt_pca.transform(features1)
    
    np.save('.\\feature_data_test\\data0.npy',data0)
    np.save('.\\feature_data_test\\data1.npy',data1)
    pass
    