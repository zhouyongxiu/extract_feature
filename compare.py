# coding=utf-8
import sys
import numpy as np
import cv2
import os
from imgaug import augmenters as iaa

def calculate_cosin(data0,data1):

    cosV12 = np.dot(data0[0],data1[0])/(np.linalg.norm(data0[0])*np.linalg.norm(data1[0]))
    return cosV12

def calculate_T(scores, label):

    best_T = 0
    maxright = 0
    if len(scores) != len(label):
        return 0, 0
    index = np.argsort(scores)

    for i in range(len(index) - 1):
        if scores[index[i]] == scores[index[i + 1]]:
            continue
        T = (scores[index[i]] + scores[index[i + 1]]) * 0.5
        right = 0
        for i in range(len(scores)):
            if scores[i] > T:
                if label[i] == 1:
                    right += 1
            else:
                if label[i] == 0:
                    right += 1
        if right > maxright:
            maxright = right
            best_T = T

    accuracy = float(maxright) / float(len(label))
    return best_T, accuracy

def get_features():
    filename = 'data/lfw/paris.txt'
    datafolder = 'data/lfw/features'
    score = []
    label = []
    count = 0
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            parts = lines.split('\t')
            print (count)
            if len(parts) == 3:
                data1_path = '%s/%s/%s_%d_1.npy'%(datafolder, parts[0], parts[0], int(parts[1]) - 1)
                data2_path = '%s/%s/%s_%d_1.npy' % (datafolder, parts[0], parts[0], int (parts[2].split('\n')[0]) - 1)
                if os.path.exists(data1_path) and os.path.exists(data2_path):
                    data1 = np.load(data1_path)
                    data2 = np.load(data2_path)
                    score.append(calculate_cosin(data1, data2))
                    label.append(1)

            elif len(parts) == 4:
                data1_path = '%s/%s/%s_%d_1.npy'%(datafolder, parts[0], parts[0], int(parts[1]) - 1)
                data2_path = '%s/%s/%s_%d_1.npy' % (datafolder, parts[2], parts[2], int (parts[3].split('\n')[0]) - 1)

                if os.path.exists(data1_path) and os.path.exists(data2_path):
                    data1 = np.load(data1_path)
                    data2 = np.load(data2_path)
                    score.append(calculate_cosin(data1, data2))
                    label.append(0)

            else:
                pass

            count += 1

    np.save('data/lfw/score.npy', score)
    np.save('data/lfw/label.npy', label)
    return score,label

if __name__=="__main__":
    
    score = np.load('data/lfw/score.npy')
    label = np.load('data/lfw/label.npy')

    T, accuracy = calculate_T(score, label)
    print ('T = %f\taccuracy = %f'%(T, accuracy))