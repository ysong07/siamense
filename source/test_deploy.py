import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
import pdb
import pickle as pkl
import h5py
from sklearn import preprocessing
 

#data = np.array(h5py.File('test_pair.h5','r')['data'])
ground_truth = np.array(h5py.File('train_pair.h5','r')['label'])
pdb.set_trace()
count= 0
""" only deploy piece that's needed """
caffe.set_mode_gpu()
sample_num = data.shape[0]
net = caffe.Net('./deploy.prototxt','./model/mlp_iter_30000.000000',caffe.TEST)
for i in range(sample_num):
    # input is 4096 *2 vector concatenated to 8192*1 vector,  each vector individually has norm 1
    # truth is 2*1 
    input = data[i,:][:]
    truth = ground_truth[i,:][:]
    net.blobs['data'].data[...] = input
    out = net.forward()
    if truth.argmax() == out['prob'].argmax():
	count = count+1
    print out['prob'].argmax()
print count
