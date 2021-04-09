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
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as layers_lib


def CNN_1(stock_name, x_batch, y_batch):
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
	epoch_limit = 40
	keep_prob_train=0.95
	T = 1000*batches
	T_eval = 100*batches
	levels = 10


	folder = '/home/leifan/Data/1Y/20Stocks_LoadRNNdata_1/'
	
	stock_file_name = stock_name + '_loadRNN_1.hdf5'
	model_identifier = stock_name
	HDF5_file = h5py.File(folder + stock_file_name, 'r')
	X_np_train = HDF5_file['X_train']
	Y_np_train = HDF5_file['y_train']
	X_np_eval = HDF5_file['X_test']
	Y_np_eval = HDF5_file['y_test']


	training_case = [LR, num_rep_block, time_lenght,pooling_filter_size,cnn_filter_size,num_filters_per_size, num_levels, epsilonADAM, keep_prob_train, T, T_eval, batches, stock_name]
	HP = '_' + 'num_rep_block' + str(num_rep_block) + '_' + 'batches'+ str(batches) + '_' + 'time_lenght' + str(time_lenght) +'_' + 'Dropout' + str(keep_prob_train)



	def resUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
	    with tf.variable_scope("res_unit_" + str(i) + "_" + str(j)):
	        part1 = slim.conv2d(input_layer, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
	        part2 = slim.batch_norm(part1, activation_fn=None)
	        part3 = tf.nn.relu(part2)
	        part4 = slim.conv2d(part3, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
	        part5 = slim.batch_norm(part4, activation_fn=None)
	        part6 = tf.nn.relu(part5)
	        output = part6
	        return output

	input_x1 =  tf.placeholder(tf.float32, shape=[None,num_inputs,time_lenght],name="input_x")
	input_x=tf.reshape(input_x1,[-1,num_inputs,time_lenght,1])
	input_y = tf.placeholder(tf.int64, shape=[None],name="input_y")
	actual_y = input_y[:]
	dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	h = slim.conv2d(input_x, num_filters_per_size[0], [num_inputs, cnn_filter_size], normalizer_fn=slim.batch_norm, scope = 'conv0', padding='VALID')


	for i in range(0,len(num_filters_per_size)):
	    for j in range(0,num_rep_block[i]):
	        h = resUnit(h, num_filters_per_size[i], cnn_filter_size, i, j)
	    h = slim.max_pool2d(h, [1,pooling_filter_size], scope='pool_%s' % i)
	# ================ Layer FC ================
	# Global avg max pooling
	h = math_ops.reduce_mean(h, [1, 2], name='pool5', keep_dims=True)

	# Conv
	h = slim.conv2d(h, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output')

	# FC & Dropout
	scores = slim.flatten(h)
	u_prob=tf.nn.softmax(scores)
	pred1D = tf.argmax(scores, 1, name="predictions")
	
	with tf.name_scope("evaluate"):
	    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,labels=input_y)
	    loss = tf.reduce_mean(losses)
	    correct_predictions = tf.equal(pred1D, input_y)
	    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	session = tf.Session()

	header = '  '
	path = header+ stock_name+'_1'
	save_path = path+'/'+'save'
	path = path+'/'
	save_path = save_path+'/'

	path_saver = save_path + "Test" + model_identifier + HP + ".ckpt"

	#save model
	saver = tf.train.Saver()
	saver.restore(session, path_saver)

	actual_out, probs = session.run([actual_y, u_prob],feed_dict={input_x1: x_batch, input_y: y_batch, dropout_keep_prob: np.float32(1.0)})


	return actual_out, probs
