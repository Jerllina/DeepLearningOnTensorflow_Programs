# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:42:31 2018

@author: lijie
"""
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #数据集被分成3部分：
    #55000张训练数据(minist.train),5000张验证数据(mnist.validation);10000张测试数据(mnist.test)
    #one-hot向量:
    #除了某一位的数字是1以外其余各维度数字都是0 
    # Define loss and optimizer
import numpy as np
    
#def train_model():
'''第一层卷积'''
    #输入尺寸：28*28
x = tf.placeholder (tf.float32, [None, 784])
y_ = tf.placeholder(tf.int32, [None,10])
    # tf.reshape函数校正张量的维度，-1表示自适应
x_image = tf.reshape (x, [-1, 28, 28, 1])
    #打破权重的对称性&避免0梯度
W_conv1 = tf.Variable (tf.truncated_normal ([5, 5, 1, 32], stddev = 0.1))#随机量填充
b_conv1 = tf.Variable (tf.constant (0.1, shape = [32]))
    #ReLU函数去线性化 一个像素一个像素地移动
h_conv1 = tf.nn.relu (tf.nn.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)
    
'''第一层池化'''
    #14*14
h_pool1 = tf.nn.max_pool (h_conv1, ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')#same：全0填充
    
'''第二层卷积'''
W_conv2 = tf.Variable (tf.truncated_normal ([5, 5, 32, 64], stddev = 0.1))
b_conv2 = tf.Variable (tf.constant(0.1, shape = [64]))
    #ReLU函数去线性化
h_conv2 = tf.nn.relu (tf.nn.conv2d(h_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv2)
    
'''第二层池化'''
    #7*7
h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')
    
'''全连接层'''
W_fc1 = tf.Variable (tf.truncated_normal ([7 * 7 * 64, 1024], stddev = 0.1))
b_fc1 = tf.Variable (tf.constant (0.1, shape = [1024]))
    
    #4维张量转换为2维
h_pool2_flat = tf.reshape (h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu (tf.matmul (h_pool2_flat, W_fc1) + b_fc1)
 
keep_prob = tf.placeholder (tf.float32)
h_fc1_drop = tf.nn.dropout (h_fc1, keep_prob)
    
'''输出层'''
    #全连接+Softmax 输出 1024→ 10个数字
W_fc2 = tf.Variable (tf.truncated_normal ([1024, 10], stddev = 0.1))
b_fc2 = tf.Variable (tf.constant(0.1, shape = [10]))
    
y_conv =tf.nn.softmax( tf.matmul (h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits (labels = y_, logits = y_conv))

        
'''模型评估'''
 
train_step = tf.train.AdamOptimizer(1e-04).minimize (cross_entropy)
 
correct_prediction = tf.equal(tf.argmax (y_conv, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))

 
sess = tf.InteractiveSession()
sess.run (tf.global_variables_initializer())
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if i % 100 == 0:    
        train_accuracy = accuracy.eval (feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        
    train_step.run (feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})
 
print ('test accuracy %g' % accuracy.eval (feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    

        
    


    

