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


#Before training,the mean must be substract
def JointBayesian_Train(trainingset, label, fold = "./"):
    if fold[-1] != '/':
        fold += '/'
    print trainingset.shape
    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim   = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # the total people num
    n_class = len(classes)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image)
    maxNumberInOneClass = 0
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels==i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]
        
        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print "prepare done, maxNumberInOneClass=", maxNumberInOneClass

    u  = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp=0
    for i in range(n_class):
        # the mean of cur[i]
        u[:,i] = np.mean(cur[i], 0)
        b = u[:,i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp+n_same_label] = cur[i].T-b
            nowp += n_same_label

    Su = np.cov(u.T,  rowvar=0)
    Sw = np.cov(ep.T, rowvar=0)
    oldSw = Sw
    SuFG  = {}
    SwG   = {}
    convergence = 1
    min_convergence = 1
    for l in range(500):
        F  = np.linalg.pinv(Sw)
        u  = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            #print 'mi=%d' % mi
            if numberBuff[mi] == 1:
		#G = −(mS μ + S ε )−1*Su*Sw−1
                G = -np.dot(np.dot(np.linalg.pinv(mi*Su+Sw), Su), F)
		#Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F+mi*G))
		#Sw*G for e
                SwG[mi]  = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
	    #formula 7 in suppl_760
            u[:,i] = np.sum(np.dot(SuFG[nn_class],cur[i].T), 1)
	    #formula 8 in suppl_760
            ep[:,nowp:nowp+nn_class] = cur[i].T+np.sum(np.dot(SwG[nn_class],cur[i].T),1).reshape(n_dim,1)
            nowp = nowp+nn_class

        Su = np.cov(u.T,  rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw-oldSw)/np.linalg.norm(Sw)
        print("Iterations-" + str(l) + ": "+ str(convergence))
        if convergence<1e-6:
            print "Convergence: ", l, convergence
            break;
        oldSw=Sw

        if convergence < min_convergence:
       	    min_convergence = convergence
            F = np.linalg.pinv(Sw)
            G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
            A = np.linalg.pinv(Su+Sw)-(F+G)
            np.save('.\\feature_data_train\\A.npy',A)
            np.save('.\\feature_data_train\\G.npy',G)
            #data_to_pkl(G, fold + "G.pkl")
            #data_to_pkl(A, fold + "A.pkl")

    F = np.linalg.pinv(Sw)
    G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
    A = np.linalg.pinv(Su+Sw) - (F+G)
    #data_to_pkl(G, fold + "G_con.pkl")
    #data_to_pkl(A, fold + "A_con.pkl")

    return A, G
if __name__=="__main__":
    data = np.load(".\\feature_data_train\\data_pca.npy")
    label = np.load('.\\feature_data_train\\label.npy')
    A,G = JointBayesian_Train(data, label, ".\\feature_data_train")
    np.save('.\\feature_data_train\\A.npy',A)
    np.save('.\\feature_data_train\\G.npy',G)
    pass
    