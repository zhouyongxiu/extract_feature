# coding=utf-8
import sys 
import numpy as np
from matplotlib import pyplot as plt
#import cv2
import caffe
import os
import copy

# caffe.set_mode_gpu()
# caffe.set_device(0)
caffe.set_mode_cpu()

def extract_feature():
    # 加载模型
    model_def = "models/VGG_FACE_deploy.prototxt"
    model_pretrained = "models/VGG_FACE.caffemodel"
    MEAN_PROTO_PATH = 'models/face_mean.binaryproto'
    floder_path = 'data/lfw/lfw_crop'
    result_path = 'data/lfw/features'
    # 1-load mean file
    blob = caffe.proto.caffe_pb2.BlobProto() 
    data = open(MEAN_PROTO_PATH,'rb').read()
    blob.ParseFromString(data)  
    array = np.array(caffe.io.blobproto_to_array(blob))
    # 2-load classifier
    net = caffe.Net(model_def,      # 定义模型结构
			model_pretrained,  # 包含了模型的训练权值
			caffe.TEST)     # 使用测试模式(不执行dropout)
    mu = array[0]
    mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
    transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
    transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
    transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR
    # 设置输入图像大小
    net.blobs['data'].reshape(1,3,224,224)
    # 导入图片并输出特征值

    for name in os.listdir(floder_path):
        #within_feature.append([])
        #label_list.append([])
        print (name)
        name_path = os.path.join(floder_path, name)
        result_name_path = os.path.join(result_path, name)
        if not os.path.exists(result_name_path):
            os.mkdir(result_name_path)
        #print img_path
        # 3-predict
        features = []
        for parent, dirnames, filenames in os.walk(name_path):
            for img in filenames:
                if ".jpg" in img:
                    img_path = os.path.join(name_path, img)
                    input_image = caffe.io.load_image(img_path)
                    net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
                    out = net.forward()
                    feature = net.blobs['fc7'].data
                    #print (feature[0]).shape
                    # features.append(copy.deepcopy(feature[0]))
                    np.save('%s/%s.npy' % (result_name_path, img.split('.')[0]), feature)
        # if (len(features)):
        #     np.save('%s/%s.npy' % (result_path,name))

# 得到总的特征值
    
if __name__=="__main__":

    extract_feature()
    