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



if __name__ =='__main__':
    train_pair = 5000
    f1 = h5py.File('test.h5','r')
    f1_pair = h5py.File('test_pair.h5','w')
    f1_pair.create_dataset('data',(train_pair,8192),dtype='f8')
    f1_pair.create_dataset('label',(train_pair,2),dtype="f8")

    scene_list = np.unique(f1['scene_name'])
    scene_list = scene_list[1:scene_list.size]
    np.random.seed(100)
    i = 0
    
    while True:
        try:	
	    scene = scene_list[np.random.random_integers(0,6,1)][0]
 	    neg_or_pos = np.random.random_integers(0,1,1)
	    test_label = np.zeros(2)
	    test_label[neg_or_pos]=1
	    # find videos belong to the same scene
	    video_list = np.where(np.array(f1['scene_name'])==scene)[0]
	    # find unique file names
	    sub_file_list = np.unique(np.array(f1['file_name'])[video_list])		
	    if neg_or_pos ==1:#same label
	        np.random.shuffle(sub_file_list)
	    	video_1 = sub_file_list[0]
	    	temp_vector = np.where(np.array(f1['file_name']) ==video_1)[0]
	    	np.random.shuffle(temp_vector)
	    	frame_1 = temp_vector[0]
	    	label_1 = np.array(f1['label'])[frame_1]
	   
	   	candidate_list = np.intersect1d(video_list,np.where(np.array(f1['label'])==label_1)[0])
	    	candidate_list = np.intersect1d(candidate_list,np.where(np.array(f1['file_name'])!=video_1)[0])
	    	np.random.shuffle(candidate_list)
	    	frame_2 = candidate_list[0]
	  	feature = np.concatenate((f1['data'][frame_1],f1['data'][frame_2]),axis=0)
	    else :
	    	np.random.shuffle(sub_file_list)
            	video_1 = sub_file_list[0]
	    	temp_vector = np.where(np.array(f1['file_name']) ==video_1)[0]
	    	np.random.shuffle(temp_vector)
            	frame_1 = temp_vector[0]
            	label_1 = np.array(f1['label'])[frame_1]
	    	candidate_list = np.intersect1d(video_list,np.where(np.array(f1['label'])!=label_1)[0])
	    	np.random.shuffle(candidate_list)
            	frame_2 = candidate_list[0]
	    	feature = np.concatenate((f1['data'][frame_1],f1['data'][frame_2]),axis=0)
        #print test_label
	#print f1['file_name'][frame_1]
	#print f1['file_name'][frame_2]
	#print f1['label'][frame_1]
	#print f1['label'][frame_2]
	    f1_pair['data'][i] = feature
	    f1_pair['label'][i]  = test_label
	    i = i+1
	    print i 
	    if i >= train_pair:
	         break	
        except:
	    continue 
    f1.close()
    f1_pair.close()    
    
    	       			   
