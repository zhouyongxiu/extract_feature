# coding=utf-8
import sys 
import numpy as np
from matplotlib import pyplot as plt
#import cv2
import os
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
global PERSON_NUM
PERSON_NUM=10


def data_pre(data):
    
    for i in range(1531):
        for j in range(4000):
            if (data[i][j] < 0):
                data[i][j] = 0
    data = np.sqrt(data)
    data = np.divide(data, np.repeat(np.sum(data, 1), data.shape[1]).reshape(data.shape[0], data.shape[1]))
    np.save('.\\feature_data_train\\data_pre.npy',data)
    return data
    
def PCA_pro(features):
    pca = PCA(n_components=180)
    pca.fit(features)
    joblib.dump(pca, ".\\feature_data_train\\pca_model.m")
    print("PCA done.")
    data_pca = pca.transform(features)
    np.save('.\\feature_data_train\\data_pca.npy',data_pca)
    return data_pca
    
def LDA_pro(features,label):
    lda = LinearDiscriminantAnalysis(n_components=10)
    lda.fit(features,label)
    joblib.dump(lda, ".\\feature_data_train\\lda_model.m")
    print("lda done.")
    data_lda = lda.transform(features)
    np.save('.\\feature_data_train\\data_lda.npy',data_lda)
    return data_lda

if __name__=="__main__":
    features = np.load('.\\feature_data_train\\features.npy')
    label = np.load('.\\feature_data_train\\label.npy')
    features = data_pre(features)
    data_pca = PCA_pro(features)
    clt_pca = joblib.load(".\\feature_data_train\\pca_model.m")
    data1 = np.array([[1]])
    data1 = np.repeat(data1,4000,axis = 1)
    data = clt_pca.transform(data1)
    pass
    