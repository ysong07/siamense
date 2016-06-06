import matplotlib.pyplot as plt
import numpy as np
import pdb
import cPickle as pkl
from sklearn import preprocessing
import h5py
import copy
import math
import matplotlib
import caffe, h5py
import scipy.io as scipy_io
from pylab import *
from caffe import layers as L
import re
import pdb
caffe.set_mode_gpu()
net = caffe.Net('./VGG_model/deploy.prototxt','./VGG_model/VGG_CNN_S.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
mean_file =scipy_io.loadmat('./VGG_model/VGG_mean')['image_mean']

transformer.set_mean('data',mean_file.transpose([2,0,1]))

transformer.set_channel_swap('data', (2,1,0))
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1,3,224,224)

if __name__ == '__main__':
    # train 8737 test 2717
    train_num = 8737
    f1 = h5py.File('train.h5','w')
    f1.create_dataset('data',(train_num,4096),dtype='f8')
    f1.create_dataset('label',(train_num,),dtype="S100")
    f1.create_dataset('file_name',(train_num,),dtype = "S100")
    f1.create_dataset('scene_name',(train_num,),dtype = "S100")
    count = 0
    with open('./data/train.txt','rb') as txt_file:
        for line_info in txt_file.readlines():
            temp_split = re.split(' ',line_info)
            image_name = './data/image_train/'+ temp_split[0]
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_name))
            out = net.forward()
            f1['data'][count] = net.blobs['fc6'].data.flatten()
            f1['label'][count] = temp_split[1]
            print temp_split[1]
	    f1['scene_name'][count] = temp_split[2]
            file_temp = re.split('_',temp_split[0])
            f1['file_name'][count]= "_".join(file_temp[0:-1])
	    count = count+1
	    print count
	    if count >=train_num-1:
	        break
    f1.close()

    test_num = 2717
    f2 = h5py.File('test.h5','w')
    f2.create_dataset('data',(test_num,4096),dtype='f8')
    f2.create_dataset('label',(test_num,),dtype="S100")
    f2.create_dataset('file_name',(test_num,),dtype = "S100")
    f2.create_dataset('scene_name',(test_num,),dtype = "S100")
    count = 0
    with open('./data/test.txt','rb') as txt_file:
        for line_info in txt_file.readlines():
            temp_split = re.split(' ',line_info)
            image_name = './data/image_test/'+ temp_split[0]
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_name))
            out = net.forward()
            f2['data'][count] = net.blobs['fc6'].data.flatten()
            f2['label'][count] = temp_split[1]
	    print temp_split[1]
            f2['scene_name'][count] = temp_split[2]
            file_temp = re.split('_',temp_split[0])
            f2['file_name'][count]= "_".join(file_temp[0:-1])
            count = count+1
            print count
            if count >=test_num-1:
                break
    f2.close()

