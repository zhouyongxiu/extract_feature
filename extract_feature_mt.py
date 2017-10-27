# coding=utf-8
import sys 
import numpy as np
from matplotlib import pyplot as plt
#import cv2
import caffe
import os
caffe.set_mode_gpu() #set calcu mode(cpu or gpu)
global PERSON_NUM
PERSON_NUM=1668
import copy 
def extract_feature(feature_num):
    compare0=[]
    compare1=[]
    # 加载模型
    model_def="F:\\MyCaffe\\face_10_feature\\deploy2.prototxt"
    model_pretrained = "F:\\MyCaffe\\face_10_feature\\face_10_model\\feature_%03d\\feature_%03d_iter_10000.caffemodel"%(feature_num,feature_num)
    MEAN_PROTO_PATH ='F:\\MyCaffe\\face_10_feature\\face_10_feature_mean\\mean_test_F%03d.binaryproto'%(feature_num)
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
    net.blobs['data'].reshape(1,3,55,55)
    # 导入图片并输出特征值
    for imag_num in np.arange(0,PERSON_NUM):
        #within_feature.append([])
        #label_list.append([])
        img_path='F:\\MyCaffe\\lfw\\lfw_mismatch_feature\\%04d\\0_%03d.jpg'%(imag_num,feature_num)
        #print img_path
        # 3-predict
        input_image = caffe.io.load_image(img_path)    
        net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
        out = net.forward()
        feature = net.blobs['fc160_2'].data
        #print (feature[0]).shape
        compare0.append(copy.deepcopy(feature[0]))
        
        img_path='F:\\MyCaffe\\lfw\\lfw_mismatch_feature\\%04d\\1_%03d.jpg'%(imag_num,feature_num)
        #print img_path
        # 3-predict
        input_image = caffe.io.load_image(img_path)    
        net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
        out = net.forward()
        feature = net.blobs['fc160_2'].data
        #print (feature[0]).shape
        compare1.append(copy.deepcopy(feature[0]))
        
    np.save('.\\feature_data_test\\features\\feature0_np%03d.npy'%(feature_num),compare0)
    np.save('.\\feature_data_test\\features\\feature1_np%03d.npy'%(feature_num),compare1)
    return compare0,compare1
# 得到总的特征值
def split_features():
    features0 = np.load('.\\feature_data_test\\features\\feature0_np%03d.npy'%(0))
    features1 = np.load('.\\feature_data_test\\features\\feature1_np%03d.npy'%(0))
    for i in range(1,25):
        feature = np.load('.\\feature_data_test\\features\\feature0_np%03d.npy'%(i))
        features0 = np.hstack((features0,feature))
        feature = np.load('.\\feature_data_test\\features\\feature1_np%03d.npy'%(i))
        features1 = np.hstack((features1,feature))
    np.save('.\\feature_data_test\\features0.npy',features0)
    np.save('.\\feature_data_test\\features1.npy',features1)
    return features0,features1
    
if __name__=="__main__":
    #features,label = split_features()
    for feature_num in np.arange(0,25,1):
        #存储特征值
        #print 'this is the %d feature'%(feature_num)
        features0,features1 = extract_feature(feature_num)
        
    split_features()
    