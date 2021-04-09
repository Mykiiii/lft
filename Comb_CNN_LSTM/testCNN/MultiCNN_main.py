import numpy as np
import pandas as pd
import tensorflow as tf  
from MultiCNN_model import MultiCNN_Model
from load_data import _load_train_test_data, _trainAndtest_length

# from mpi4py import MPI
import os
import time

#*********************************************
# fixed variables    
time_interval_1 = 1
time_interval_01 = 0.1
time_interval_ratio = int(time_interval_1/time_interval_01)
num_classes = 3
num_levels = 10
num_inputs = int(num_levels*2+4)
num_per_day = 20000
# Hyperparamaters
num_layers = 2
BPTT_length = 20
batch_size = 1024
cnn_filter_size=3
num_filters_per_size=(64,128,256,512)
num_rep_block=(1,1,1,1) 
pooling_filter_size=2
LSTM_lasting_time = 20*BPTT_length
train_dropout_1 = 0.8
train_dropout_01 = 0.8
test_dropout = 1.0
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilonADAM = 1e-8
#*****************************************
epoch_limit = 100

stock_ticker = 'CBS'
folder_1 = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
folder_01 = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_01/'

start_time = time.time()

HDF5_file_LoadRNN_1 = _load_train_test_data(stock_ticker, time_interval_1, folder_1 = folder_1, folder_01 =folder_01)
X_train_1 = HDF5_file_LoadRNN_1['X_train']
X_test_1 = HDF5_file_LoadRNN_1['X_test']
y_train_1 = HDF5_file_LoadRNN_1['y_train']
y_test_1 = HDF5_file_LoadRNN_1['y_test']

HDF5_file_LoadRNN_01 = _load_train_test_data(stock_ticker, time_interval_01, folder_1 = folder_1, folder_01 =folder_01)
X_train_01 = HDF5_file_LoadRNN_01['X_train']
X_test_01 = HDF5_file_LoadRNN_01['X_test']
y_train_01 = HDF5_file_LoadRNN_01['y_train']
y_test_01 = HDF5_file_LoadRNN_01['y_test']

train_len_1, test_len_1 = _trainAndtest_length(stock_ticker, time_interval_1, folder_1 = folder_1, folder_01 =folder_01)
train_len_01, test_len_01 = _trainAndtest_length(stock_ticker, time_interval_01, folder_1 = folder_1, folder_01 =folder_01)
num_BPTT_iterations = int(LSTM_lasting_time/BPTT_length)
train_num_of_days = int(train_len_1/num_per_day)
test_num_of_days = int(test_len_1/num_per_day)


config = {    'stock_ticker':   stock_ticker,
              'num_inputs' :    num_inputs,               
              'filter_size' :   cnn_filter_size,         
              'BPTT_length' :    BPTT_length,
              'batch_size' :    batch_size,
              'num_filters_per_size': num_filters_per_size,
              'num_rep_block': num_rep_block, 
              'beta1'  :        beta1,  
              'beta2'  :        beta2,  
              'pooling_filter_size': pooling_filter_size,         
              'learning_rate' : learning_rate,
              'epsilonADAM' : epsilonADAM,          
              'num_classes':    num_classes}


model = MultiCNN_Model(config)

sess = tf.Session()
sess.run(model.init_op)
# sess.run(model.saver)
saver = tf.train.Saver()


#save path
header = '/home/leifan/Dropbox/Comb/testCNN/'
# os.system('mkdir'+ ' '+header+stock_name+'_01')
os.system('mkdir'+ ' '+header+stock_ticker)
path = header+stock_ticker+'/'
os.system('mkdir'+ ' '+path+'save')
save_path = path+'save/'
path_saver = save_path + "MultiCNN.ckpt"

saver.restore(sess, path_saver)


#define savable values
cond_acc = np.zeros(epoch_limit)


for ii in range(epoch_limit):
    print('epoch: %d' %(ii+1))
    #*****************************************************************************************************************               
    # Testing 
    counter = 0
    # test_state_1 = sess.run(model.reset_state_1)

    test_random_day = np.random.choice(test_num_of_days, batch_size)
    test_random_day_index = test_random_day * num_per_day
    test_random_start_time = np.random.choice(np.arange(0, num_per_day - LSTM_lasting_time), batch_size)
    test_random_start_time_index = test_random_day_index + test_random_start_time


    X_test_batch_1 = np.float32(np.zeros((batch_size,BPTT_length,num_inputs)))
    y_test_batch_1 = np.int32(np.zeros(batch_size))
    X_test_batch_01 = np.float32(np.zeros((batch_size,BPTT_length,num_inputs)))
    y_test_batch_01 = np.int32(np.zeros(batch_size))

    total_movements = 0
    Movements_predicted = 0.0
    
    for i in range(num_BPTT_iterations):
        for k in range(batch_size):    
            batch_k_start_index_1 = test_random_start_time_index[k]
            batch_k_end_index_1 = batch_k_start_index_1 + BPTT_length 
            
            X_test_batch_1[k,:,:] = X_test_1[batch_k_start_index_1: batch_k_end_index_1,:]
            y_test_batch_1[k] = y_test_1[batch_k_end_index_1-1]

            batch_k_end_index_01 = (batch_k_end_index_1-1) * time_interval_ratio + 1
            batch_k_start_index_01 = batch_k_end_index_01 - BPTT_length
            
            X_test_batch_01[k,:,:] = X_test_01[batch_k_start_index_01: batch_k_end_index_01,:]
            y_test_batch_01[k] = y_test_01[batch_k_end_index_01-1]
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
        # test_state_01 = sess.run(model.reset_state_01)
        actual_out,probs = sess.run([model.labels_01, model.u_prob],
                feed_dict = {model.input_1: X_test_batch_1, model.labels_1: y_test_batch_1,
                            model.input_01: X_test_batch_01, model.labels_01: y_test_batch_01,
                            model.keep_prob_1:test_dropout, model.keep_prob_01:test_dropout})
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        test_random_start_time_index = test_random_start_time_index + BPTT_length

        probs[:, 1] = 0
        pred_out = [np.argmax(probs, 1)]
        pred_out = np.array(pred_out).reshape(-1)
        counter += 1
        Movements_predicted += sum([1 for p, j in zip(actual_out, pred_out) if p != 1 and (p == j)])
        total_movements += sum([1 for p, j in zip(actual_out, pred_out) if p!= 1])

    conditional_accuracy = np.array(Movements_predicted) / total_movements
    print(Movements_predicted, 'Movements_predicted correctly')
    print(total_movements, 'Total Movements')
    print(conditional_accuracy, 'Conditional Accuracy')
    cond_acc[ii] = conditional_accuracy

cond_acc = pd.DataFrame(cond_acc)
cond_acc.to_csv(path+stock_ticker+'_CNN_cond.csv')
end_time = time.time()
print(end_time-start_time)
