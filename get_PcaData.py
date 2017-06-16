# coding=utf-8
import numpy as np
global SAMPLE_NUM
SAMPLE_NUM=90000

def get_count(label):
    #label = np.load('..\\data_pca\\label.npy')
    count = np.zeros(10575)
    max_count = 0
    for n in range(10):   
        for i in range(10575):
            if i in label:
                count[i] = count[i] + 1
                index = np.argwhere(label == i)[0]
                label = np.delete(label,index)
                max_count = max_count + 1
                if max_count >= SAMPLE_NUM:
                    break
        if max_count >= SAMPLE_NUM:
                    break
    np.save('..\\data_pca\\count.npy',count)
    return count

def get_index(label,count):
    #label = np.load('..\\data_pca\\label.npy')
    #count = np.load('..\\data_pca\\count.npy')
    index = np.zeros(count.sum(),dtype=np.int)
    index_num = 0
    for i in range(10575):
        index_i = np.argwhere(label == i)
        for j in range(int(count[i])):
            index[index_num] = index_i[j]
            index_num = index_num + 1
    np.save('..\\data_pca\\index.npy',index)
    return index

def split_features():
    features = []
    features = np.load('..\\data_pca\\features\\feature_%03d.npy'%(0))
    for i in range(1,25):
        feature1 = np.load('..\\data_pca\\features\\feature_%03d.npy'%(i))
        features = np.hstack((features,feature1))
    np.save('..\\data_pca\\features.npy',features)
    return features

def extract_feature(features,index):
    #index = np.load('..\\data_pca\\index.npy')
    ex_features = np.zeros(features.shape,dtype = np.float64)
    for i in range(SAMPLE_NUM):
        ex_features[index[i],:] = features[index[i],:]
    np.save('..\\data_pca\\ex_features.npy',ex_features)
    return ex_features 
    
if __name__=="__main__":
    label = np.load('..\\data_pca\\label.npy')
    count = get_count(label)
    index = get_index(label,count)
    features = split_features()
    ex_features = extract_feature(features,index)
    
    