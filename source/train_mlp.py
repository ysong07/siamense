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

def net(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=1024, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=1024, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)
    n.ip3 = L.InnerProduct(n.relu2, num_output=2, weight_filler=dict(type='xavier'))
    n.loss = L.SigmoidCrossEntropyLoss(n.ip3, n.label)
    return n.to_proto()



if __name__=='__main__':

    sample_num = 5000
    with open('./auto_train.prototxt', 'w') as f:
        f.write(str(net('train.h5list', 50)))
    with open('auto_test.prototxt', 'w') as f:
        f.write(str(net('test.h5list', 20)))

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('auto_solver.prototxt')

    solver.net.forward()
    solver.test_nets[0].forward()
    solver.step(1)

    niter = 100000
    test_interval = 1000
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
    print len(test_acc)
    output = zeros((niter, 2, 2))

    # The main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        train_loss[it] = solver.net.blobs['loss'].data
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
	    correct = 0
            data = solver.test_nets[0].blobs['ip3'].data
            label = solver.test_nets[0].blobs['label'].data
            for test_it in range(sample_num):
                solver.test_nets[0].forward()
                # Positive values map to label 1, while negative values map to label 0
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if data[i][j] > 0 and label[i][j] == 1:
                            correct += 1
                        elif data[i][j] <= 0 and label[i][j] == 0:
                            correct += 1
            test_acc[int(it / test_interval)] = correct * 1.0 / (len(data) * len(data[0]) * sample_num)
            solver.net.save('./model/mlp_iter_%f'%(it))
    scipy_io.savemat('curve.mat',{'train':train_loss,'test':test_acc})

