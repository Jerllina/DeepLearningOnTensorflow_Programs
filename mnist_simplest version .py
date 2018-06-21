# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:06:52 2018

@author: lijie
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
sess = tf.InteractiveSession()


'''load data'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''model building'''
#input
x = tf.placeholder (tf.float32, [None, 784])
#weight of greyscale
W = tf.Variable (tf.zeros([784, 10]))
#weight of figure
b = tf.Variable (tf.zeros([10]))
#output
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None,10])

#cost function
#cross_entropy = -tf.reduce_sum (y_ * tf.log (y))
#y = tf.matmul (x, W) + b
cross_entropy = tf.reduce_mean (
        tf.nn.softmax_cross_entropy_with_logits (labels = y_, logits = y))

train_step = tf.train.GradientDescentOptimizer (0.5).minimize (cross_entropy)

#sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

'''model training'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run (train_step, feed_dict = {x: batch_xs, y_: batch_ys})
    
'''model evaluation'''
correct_prediction = tf.equal (tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean (tf.cast(correct_prediction, tf.float32))
print (sess.run (accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))  

    