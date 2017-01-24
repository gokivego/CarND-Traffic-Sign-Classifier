import pickle
#import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import math

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# input X: 32x32 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 42])
# variable learning rate
lr = tf.placeholder(tf.float32)

#  Two layers and their number of neurons (tha last layer has 42 softmax neurons)
L = 200
M = 100

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10

W1 = tf.Variable(tf.truncated_normal([1024, L], stddev=0.1))  # 1024 = 32 * 32
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W5 = tf.Variable(tf.truncated_normal([O, 42], stddev=0.1))
B5 = tf.Variable(tf.zeros([42]))

# The model

