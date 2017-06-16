# coding=utf-8
import sys 
import numpy as np
from matplotlib import pyplot as plt
#import cv2
import caffe
import os
caffe.set_mode_gpu() #set calcu mode(cpu or gpu)
global PERSON_NUM
PERSON_NUM=10575
import copy
old_feature_index = [1,6,7,8,10,13,14,15,16,19,20,22,26,27,29,41,59,121,122,125,126,127,130,132,133]
def extract_feature(feature_num):
    within_feature=[]
    label_list = []
    # 加载模型
    model_def="F:\\MyCaffe\\data\\data_pca\\train_test_feature_common.prototxt"
    model_pretrained = "F:\\MyCaffe\\data\\out\\F%03d\\feature_%03d_iter_600000.caffemodel"%(feature_num,feature_num)  
    MEAN_PROTO_PATH ='F:\\MyCaffe\\data\\data\\data_mean\\mean_train_F%03d.binaryproto'%(feature_num)
    # 1-load mean file
    blob = caffe.proto.caffe_pb2.BlobProto() 
    data=open(MEAN_PROTO_PATH,'rb').read()
    blob.ParseFromString(data)  
    array = np.array(caffe.io.blobproto_to_array(blob))
    # 2-load classifier
    net = caffe.Net(model_def,      # 定义模型结构
			model_pretrained,  # 包含了模型的训练权值
			caffe.TEST)     # 使用测试模式(不执行dropout)
    mu=array[0]
    mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
    transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
    transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
    transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR
    # 设置输入图像大小
    net.blobs['data'].reshape(1,3,55,47)
    # 导入图片并输出特征值
    for person_index in np.arange(0,PERSON_NUM):
        #within_feature.append([])
        #label_list.append([])
        print 'this is the %d person'%(person_index)
        person_path='F:\\MyCaffe\\data\\face\\%06d\\%03d'%(person_index,old_feature_index[feature_num])
        for parent,dirnames,filenames in os.walk(person_path):
            for filename in filenames:
                if ".jpg" in filename:
                    img_path=os.path.join(parent,filename)
                    #print img_path
                    # 3-predict
                    input_image = caffe.io.load_image(img_path)    
                    net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
                    out = net.forward()
                    feature = net.blobs['fc160_2'].data
                    #print (feature[0]).shape
                    within_feature.append(copy.deepcopy(feature[0]))
                    label_list.append(person_index)
    np.save('F:\\MyCaffe\\data\\data_pca\\features\\feature_%03d.npy'%(feature_num),within_feature)
    np.save('F:\\MyCaffe\\data\\data_pca\\labels\\label_%03d.npy'%(feature_num),label_list)
    return within_feature
# 得到总的特征值
def split_features():
    features = []
    label = np.load('F:\\MyCaffe\\data\\data_pca\\labels\\label_%03d.npy'%(0))
    features = np.load('F:\\MyCaffe\\data\\data_pca\\features\\feature_%03d.npy'%(0))
    for i in range(1,25):
        feature1 = np.load('F:\\MyCaffe\\data\\data_pca\\features\\feature_%03d.npy'%(i))
        features = np.hstack((features,feature1))
    np.save('F:\\MyCaffe\\data\\data_pca\\features.npy',features)
    np.save('F:\\MyCaffe\\data\\data_pca\\label.npy',label)
    return features,label

def split_features_test():
    features = []
    label = np.load('F:\\MyCaffe\\data\\data_pca\\labels\\label_%03d.npy'%(0))
    features = np.load('F:\\MyCaffe\\data\\data_pca\\features\\feature_%03d.npy'%(0))
    for i in range(1,25):
        feature1 = np.load('F:\\MyCaffe\\data\\data_pca\\features\\feature_%03d.npy'%(0))
        features = np.hstack((features,feature1))
    np.save('F:\\MyCaffe\\data\\data_pca\\features.npy',features)
    np.save('F:\\MyCaffe\\data\\data_pca\\label.npy',label)
    return features,label
if __name__=="__main__":    
    for feature_num in np.arange(19,25,1):
        #存储特征值
        print 'this is the %d feature'%(feature_num)
        within_feature=extract_feature(feature_num)
    #features,label = split_features_test()