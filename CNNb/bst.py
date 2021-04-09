import numpy as np
import tensorflow as tf
import time
import copy
import os
import sys
import h5py
import random
import pandas as pd
import pickle
from CNN_1 import CNN_1
from CNN_2 import CNN_2
from CNN_3 import CNN_3
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as layers_lib

t1 = time.time()
LR = .001
epsilonADAM = 1e-8
time_lenght=80
num_nodes = 1
stock_num = 0
num_levels = 10
num_inputs = num_levels*2+4
num_stocks = 1
batches = 500
num_classes=3
cnn_filter_size=3
pooling_filter_size=2
num_filters_per_size=(64,128,256,512)
num_rep_block=(1,1,1,1) 
epoch_limit = 1
keep_prob_train=0.95
T = 1000*batches
T_eval = 1*batches
levels = 10


folder = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
stock_file_name = 'AMD_loadRNN_1.hdf5'
stock_name = 'AMD'
model_identifier = stock_name
HDF5_file = h5py.File(folder + stock_file_name, 'r')
X_np_eval = HDF5_file['X_test']
Y_np_eval = HDF5_file['y_test']


cond_acc = np.zeros(epoch_limit)

for i in range(epoch_limit):
    print('Epoch: %d' %(i+1))

    # EVALUATION 
    current_eval_error = 0.0
    current_eval_acc = 0.0
    random_index_list = []
    total_movements=0.0
    for k in range(batches):
        random_index = random.randint(0,len(Y_np_eval)-T_eval-1)
        random_index_list.append(random_index)
    counter = 0
    x_batch = np.float32(np.zeros((batches, num_inputs,time_lenght)))
    y_batch = np.int32(np.zeros((batches)))
    Movements_predicted = 0.0
    for i in range(0, T_eval, batches):
        for k in range(batches):
            kk =random_index_list[k]
            x_batch[k, :, :] = np.transpose(X_np_eval[i+kk :i+kk + time_lenght, :])
            y_batch[k] = Y_np_eval[i+kk + time_lenght-1]
        actual_out, probs_1 = CNN_1('AMD', x_batch, y_batch)
        actual_out, probs_2 = CNN_2('AMD', x_batch, y_batch)
        actual_out, probs_3 = CNN_3('AMD', x_batch, y_batch)
        probs_1[:, 1] = 0
        probs_2[:, 1] = 0
        probs_3[:, 1] = 0
        pd1 = [np.argmax(probs_1, 1)]
        pd1 = np.array(pd1).reshape(-1)
        pd2 = [np.argmax(probs_2, 1)]
        pd2 = np.array(pd2).reshape(-1)
        pd3 = [np.argmax(probs_3, 1)]
        pd3 = np.array(pd3).reshape(-1)
        pred_out = pd1[:]
        pred_out = np.array(pred_out).reshape(-1)
        actual_out_1 = np.array(actual_out).reshape(-1)
        tmp1 = np.vstack((pd1,pd2))
        temp2 = np.vstack((pd3,pred_out))
        temp = np.vstack((tmp1, temp2))
        temp = np.vstack((temp, actual_out_1))
        temp = np.transpose(temp)
        temp = pd.DataFrame(temp)
        temp.to_csv('temp.csv')
        for l in range(len(pd1)):
            if pd1[l]==pd2[l]:
                pred_out[l] = pd1[l]
            else:
                if pd3[l]==pd1[l]:
                    pred_out[l] = pd1[l]
                else:
                    pred_out[l] = pd2[l]

        pred_out = np.array(pred_out).reshape(-1)
        counter += 1
        Movements_predicted += sum([1 for p, j in zip(actual_out, pred_out) if p != 1 and (p == j)])
        total_movements += sum([1 for p, j in zip(actual_out, pred_out) if p!= 1])

    conditional_accuracy = np.array(Movements_predicted) / total_movements
    print(Movements_predicted, 'Correct Prediction')
    print(total_movements)
    print(conditional_accuracy)
cond_acc = pd.DataFrame(cond_acc)
cond_acc.to_csv('cond_acc.csv')
t2 = time.time()
print(t2-t1)



