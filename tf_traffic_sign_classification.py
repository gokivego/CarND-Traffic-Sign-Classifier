
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Step 0: Load The Data

# In[20]:

import tensorflow as tf
import tensorflowvisu
import DataSet
import numpy as np
import cv2
#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
tf.set_random_seed(0)
# Load pickled data
import pickle
import math


# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# Data preprocessing steps

# convert to B/W
X_train_bw = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_train])
X_test_bw = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_test])

# apply histogram equalization
X_train_hst_eq = np.array([cv2.equalizeHist(image) for image in X_train_bw])
X_test_hst_eq = np.array([cv2.equalizeHist(image) for image in X_test_bw])

# reshape for conv layer
X_train = X_train_hst_eq[..., np.newaxis]
X_test = X_test_hst_eq[..., np.newaxis]
print('Before shaping:', X_train_hst_eq.shape)
print('After shaping:', X_train.shape)

X_train_normalized = (X_train - np.mean(X_train)) / 128.0
X_test_normalized = (X_test - np.mean(X_test)) / 128.0
print('Mean before normalizing:', np.mean(X_train), np.mean(X_test))
print('Mean after normalizing:', np.mean(X_train_normalized), np.mean(X_test_normalized))

X_train = X_train_normalized

# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# In[21]:

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Create a validation set using sklearn.model_selections train_test_split and load shuffle to shuffle the data

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state= 5)
dataset = DataSet.DataSet(X_train, y_train, reshape=False)

# y_train = tf.one_hot(y_train,43)
# y_validation = training_file.one_hot(y_validation, 43)

print ("The size of the training set is:",len(y_train) ,"and the size of the validation set is:",len(y_validation))


# ## Visualize Data
# 
# View a sample from the dataset.
# 
# You do not need to modify this section.

# In[23]:

"""
import random
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

"""

# ## Preprocess Data
# 
# Shuffle the training data.
# 
# Then center the data. Find the mean of each pixel and subtract the mean from every image
# 
# You do not need to modify this section.

# In[31]:


# ## Setup TensorFlow

# The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
# 
# You do not need to modify this section.

# In[25]:

EPOCHS = 10
BATCH_SIZE = 100


# ## SOLUTION: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
# 
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
# 
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 14x14x6.
# 
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
# 
# **Activation.** Your choice of activation function.
# 
# **Pooling.** The output shape should be 5x5x16.
# 
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
# 
# **Layer 3: Fully Connected.** This should have 120 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 4: Fully Connected.** This should have 84 outputs.
# 
# **Activation.** Your choice of activation function.
# 
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
# 
# ### Output
# Return the result of the 2nd fully connected layer.

# In[26]:


# def LeNet(x):
# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer

# `X` is a placeholder for a batch of input images.
# `Y_` is a placeholder for a batch of output labels.

X = tf.placeholder(tf.float32, (None, 32, 32,1))
Y_ = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(Y_, 43)

# variable learning rate
# lr = tf.placeholder(tf.float32)

mu = 0
sigma = 0.1

# SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
conv1_b = tf.Variable(tf.zeros(6))
conv1 = tf.nn.conv2d(X, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

# SOLUTION: Activation.
conv1 = tf.nn.relu(conv1)

# SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
conv2_b = tf.Variable(tf.zeros(16))
conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

# SOLUTION: Activation.
conv2 = tf.nn.relu(conv2)

# SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# SOLUTION: Flatten. Input = 5x5x16. Output = 400.
fc0 = flatten(conv2)

# SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
fc1_b = tf.Variable(tf.zeros(120))
fc1 = tf.matmul(fc0, fc1_W) + fc1_b

# SOLUTION: Activation.
fc1 = tf.nn.relu(fc1)

# SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
fc2_b = tf.Variable(tf.zeros(84))
fc2 = tf.matmul(fc1, fc2_W) + fc2_b

# SOLUTION: Activation.
fc2 = tf.nn.relu(fc2)

# SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
fc3_b = tf.Variable(tf.zeros(43))
logits = tf.matmul(fc2, fc3_W) + fc3_b

# return logits

# ## Features and Labels
# Train LeNet to classify Traffic Sign Data
#
# You do not need to modify this section.

# In[27]:


# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
# 
# You do not need to modify this section.

# In[28]:

rate = 0.001

# logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
cross_entropy = tf.reduce_mean(cross_entropy)

#loss_operation = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#ation = optimizer.minimize(loss_operation)


# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
# 
# You do not need to modify this section.

# In[29]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

"""
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
"""

# matplotlib visualisation
allweights = tf.concat(0, [tf.reshape(conv1_W, [-1]), tf.reshape(conv2_W, [-1]), tf.reshape(fc1_W, [-1]), tf.reshape(fc2_W, [-1]), tf.reshape(fc3_W, [-1])])
allbiases  = tf.concat(0, [tf.reshape(conv1_b, [-1]), tf.reshape(conv2_b, [-1]), tf.reshape(fc1_b, [-1]), tf.reshape(fc2_b, [-1]), tf.reshape(fc3_b, [-1])])
conv_activations = tf.concat(0, [tf.reshape(tf.reduce_max(conv1, [0]), [-1]), tf.reshape(tf.reduce_max(conv2, [0]), [-1])])
dense_activations = tf.concat(0, [tf.reshape(tf.reduce_max(fc1, [0]), [-1]), tf.reshape(tf.reduce_max(fc2, [0]), [-1])])
#I = tensorflowvisu.tf_format_mnist_images(X, logits, one_hot_y)
#It = tensorflowvisu.tf_format_mnist_images(X, logits, one_hot_y, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()
#datavis = tensorflowvisu.MnistDataVis(title4="batch-max conv activations", title5="batch-max dense activations", histogram4colornum=2, histogram5colornum=2)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels

    #X_train, y_train = shuffle(X_train, y_train)
    batch_X, batch_Y = dataset.next_batch(BATCH_SIZE)


    """
    # learning rate decay
    #max_learning_rate = 0.003
    #min_learning_rate = 0.0001
    #decay_speed = 2000
    max_learning_rate = 0.02
    min_learning_rate = 0.00015
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    """

    # compute training values for visualisation
    if update_train_data:
        a, c, ca, da = sess.run([accuracy, cross_entropy, conv_activations, dense_activations],
                                {X: batch_X, Y_: batch_Y})
        # a, c,im, ca, da = sess.run([accuracy, cross_entropy, I, conv_activations, dense_activations],
        #                            {X: batch_X, Y_: batch_Y, tst: False, pkeep: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + ")")
        # print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        # datavis.update_image1(im)
        datavis.append_data_histograms(i, ca, da)

    # compute test values for visualisation
    if update_test_data:
        a, c= sess.run([accuracy, cross_entropy], {X: X_validation, Y_: y_validation})
        # a, c, im = sess.run([accuracy, cross_entropy, It],
        #                    {X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0})
        print(str(i) + ": ********* epoch " +
              str(i * BATCH_SIZE // X_train.shape[0] + 1) + " ********* test accuracy:" +
              str(a) + " test loss: " + str(c))
        # print(str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        # datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y})
    # sess.run(update_ema, {X: batch_X, Y_: batch_Y})

    # sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75})
    # sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i})


datavis.animate(training_step, 2001, train_data_update_freq=20, test_data_update_freq=100)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
#for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))



# ## Train the Model
# Run the training data through the training pipeline to train the model.
# 
# Before each epoch, shuffle the training set.
# 
# After each epoch, measure the loss and accuracy of the validation set.
# 
# Save the model after training.
# 
# You do not need to modify this section.

# In[32]:

"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
# 
# Be sure to only do this once!
# 
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
# 
# You do not need to modify this section.

# In[ ]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

"""