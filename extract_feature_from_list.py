# coding=utf-8
import sys 
import numpy as np
from matplotlib import pyplot as plt
#import cv2
import caffe
import os
import copy
import cv2
import time

caffe.set_mode_gpu()
caffe.set_device(3)
#caffe.set_mode_cpu()

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def extract_feature(pairs):
    # 加载模型
    model_def = "models/VGG_asian.prototxt"
    model_pretrained = "models/_iter_110005.caffemodel"
    output_lapyer = 'fc7'
    img_height = 224
    img_width = 224
    img_scale = 255
    mu = np.array([103.75, 117.00, 147.64], dtype=np.float64)
    # mu = np.array([127.5 * 0.0078125, 127.5 * 0.0078125, 127.5 * 0.0078125], dtype=np.float64)
    # mu = np.array([127.5, 127.5, 127.5], dtype=np.float64)
    # MEAN_PROTO_PATH = 'models/face_mean.binaryproto'
    floder_path = 'data/asian_test/crop'
    flip = False

    # 1-load mean file
    blob = caffe.proto.caffe_pb2.BlobProto() 
    # data = open(MEAN_PROTO_PATH,'rb').read()
    # blob.ParseFromString(data)
    # array = np.array(caffe.io.blobproto_to_array(blob))
    # 2-load classifier
    net = caffe.Net(model_def,      # 定义模型结构
			model_pretrained,  # 包含了模型的训练权值
			caffe.TEST)     # 使用测试模式(不执行dropout)
    # mu = array[0]
    # mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
    transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
    # transformer.set_raw_scale('data', img_scale)      #将像素值从[0,255]变换到[0,1]之间
    #transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR
    # 设置输入图像大小
    net.blobs['data'].reshape(1,3,img_height,img_width)
    # 导入图片并输出特征值

    label_list = np.zeros((pairs.shape[0],1), dtype=np.int)
    features1 = []
    features2 = []
    for i in range(pairs.shape[0]):
        #within_feature.append([])
        #label_list.append([])

        print ('%d\n'%i)

        if len(pairs[i]) == 3:

            label_list[i] = 1
            img1_path = os.path.join(floder_path, pairs[i][0], pairs[i][0] + '_%04d'%(int(pairs[i][1])) + '_0.png')
            img2_path = os.path.join(floder_path, pairs[i][0], pairs[i][0] + '_%04d'%(int(pairs[i][2])) + '_0.png')


            if not os.path.exists(img1_path):
                print img1_path
                assert (0)
            if not os.path.exists(img2_path):
                print img2_path
                assert (0)
            #input_image = caffe.io.load_image(img1_path)
            input_image = cv2.imread(img1_path).astype(np.float64)
            start = time.clock()
            net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
            out = net.forward()
            feature = net.blobs[output_lapyer].data
            feature_temp = copy.deepcopy(feature[0])
            if (flip):
                input_image = cv2.flip(input_image, 1)
                net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
                out = net.forward()
                feature = net.blobs[output_lapyer].data
                feature_temp = np.hstack((feature_temp, copy.deepcopy(feature[0])))
            features1.append(copy.deepcopy(feature_temp))
            #print (np.sum(feature[0]))
            end = time.clock()
            print ('time = %f ms'%((end - start) * 1000))

            #input_image = caffe.io.load_image(img2_path)
            input_image = cv2.imread(img2_path).astype(np.float64)
            net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
            out = net.forward()
            feature = net.blobs[output_lapyer].data
            feature_temp = copy.deepcopy(feature[0])
            if (flip):
                input_image = cv2.flip(input_image, 1)
                net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
                out = net.forward()
                feature = net.blobs[output_lapyer].data
                feature_temp = np.hstack((feature_temp, copy.deepcopy(feature[0])))
            features2.append(copy.deepcopy(feature_temp))
            #print (np.sum(feature[0]))

        elif len(pairs[i]) == 4:

            label_list[i] = 0
            img1_path = os.path.join(floder_path, pairs[i][0], pairs[i][0] + '_%04d'%(int(pairs[i][1])) + '_0.png')
            img2_path = os.path.join(floder_path, pairs[i][2], pairs[i][2] + '_%04d'%(int(pairs[i][3])) + '_0.png')

            if not os.path.exists(img1_path):
                print img1_path
                assert (0)
            if not os.path.exists(img2_path):
                print img2_path
                assert (0)

            #input_image = caffe.io.load_image(img1_path)
            input_image = cv2.imread(img1_path).astype(np.float64)
            start = time.clock()
            net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
            out = net.forward()
            feature = net.blobs[output_lapyer].data
            feature_temp = copy.deepcopy(feature[0])
            if (flip):
                input_image = cv2.flip(input_image, 1)
                net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
                out = net.forward()
                feature = net.blobs[output_lapyer].data
                feature_temp = np.hstack((feature_temp, copy.deepcopy(feature[0])))
            features1.append(copy.deepcopy(feature_temp))
            #print (np.sum(feature[0]))
            end = time.clock()
            print ('time = %f ms' % ((end - start) * 1000))

            #input_image = caffe.io.load_image(img2_path)
            input_image = cv2.imread(img2_path).astype(np.float64)
            net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
            out = net.forward()
            feature = net.blobs[output_lapyer].data
            feature_temp = copy.deepcopy(feature[0])
            if (flip):
                input_image = cv2.flip(input_image, 1)
                net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
                out = net.forward()
                feature = net.blobs[output_lapyer].data
                feature_temp = np.hstack((feature_temp, copy.deepcopy(feature[0])))
            features2.append(copy.deepcopy(feature_temp))
            #print (np.sum(feature[0]))
        else:

            print ('pairs read err: %d' % i)
            cv2.waitkey(0)

    f1 = np.array(features1)
    f2 = np.array(features2)

    return f1, f2, label_list

def calculate_cosin(data0,data1):

    cosV12 = np.dot(data0[0],data1[0])/(np.linalg.norm(data0[0])*np.linalg.norm(data1[0]))
    return cosV12

def calculate_score(features1, features2):

    score = np.zeros((features1.shape[0], 1), dtype=np.float64)
    for i in range(features1.shape[0]):
        score[i] = calculate_cosin(features1[i:], features2[i:])

    return score


def calculate_T(scores, label):

    best_T = 0
    maxright = 0
    if len(scores) != len(label):
        return 0, 0

    for T in np.arange(0, 1, 0.01):
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

if __name__=="__main__":


    pairs = read_pairs('data/asian_test/pairs.txt')
    [features1, features2, label_list] = extract_feature(pairs)

    #np.save('data/asian_test/features1.npy', features1)
    #np.save('data/asian_test/features2.npy', features2)
    #np.save('data/asian_test/label_list.npy', label_list)


    #features1 = np.load('data/asian_test/features1.npy')
    #features2 = np.load('data/asian_test/features2.npy')
    #label_list = np.load('data/asian_test/label_list.npy')

    score = calculate_score(features1, features2)

    best_T, accuracy = calculate_T(score, label_list)

    print (best_T, accuracy)
