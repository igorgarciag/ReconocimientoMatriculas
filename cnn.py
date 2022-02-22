"""
http://robromijnders.github.io/tensorflow_basic/

LeNet-5 Yann Lecun

"""
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras import layers, models, losses
import matplotlib.pyplot as plt

import dataload as dataset

#definicion de valores predeterminados de datos de la red
def generate_weight(shape, name):
    initial = tf.compat.v1.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def generate_bias(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = name)

def convolution2d(x, W, stride = (1, 1), padding = 'SAME'):
    return tf.nn.conv2d(x, W, strides = [1, stride[0], stride[1], 1], padding = padding)

def maxpooling2x2(x, ksize = (2, 2), stride = (2, 2)):
    return tf.nn.max_pool(x, ksize = [1, ksize[0], ksize[1], 1], strides = [1, stride[0], stride[1], 1], padding = 'SAME')

#deshabilitar eager_execution y resetear grafo computacional
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

#declaracion de la sesion
session = tf.compat.v1.Session()

#creacion de placeholders para imagenes y clases
x = tf.compat.v1.placeholder("float", shape = [None, 1024])
y = tf.compat.v1.placeholder("float", shape = [None, 36])

with tf.name_scope("Reshaping") as scope:
    imagex = tf.reshape(x, [-1, 32, 32, 1])

with tf.name_scope("Convolution1") as socpe:
    w_conv1 = generate_weight([5, 5, 1, 64], "ConvLayer01")
    b_conv1 = generate_bias([64], "BiasConvLayer01")
    h_conv1 = tf.nn.relu(convolution2d(imagex, w_conv1) + b_conv1)
    pooling1 = maxpooling2x2(h_conv1)

with tf.name_scope("Convolution2") as socpe:
    w_conv2 = generate_weight([3, 3, 1, 64], "ConvLayer02")
    b_conv2 = generate_bias([64], "BiasConvLayer02")
    h_conv2 = tf.nn.relu(convolution2d(imagex, w_conv2) + b_conv2)
    pooling2 = maxpooling2x2(h_conv2)

with tf.name_scope("Dense1") as scope:
    w_dense1 = generate_weight([4 * 4 * 64, 1024], "DenseLayer01")
    b_dense1 = generate_bias([1064], "BiasDenseLayer01")
    pooling2flatten = tf.reshape(pooling2, [-1, 4 * 4 * 64])
    h_Dense1 = tf.nn.relu(tf.matmul(pooling2flatten, w_dense1) + b_dense1)

with tf.name_scope("Dense2") as scope:
    prob = tf.compat.v1.placeholder("float")

    #prevenir overfitting
    h_dense1drop = tf.nn.dropout(h_Dense1, prob)

    w_dense2 = generate_weight([4 * 4 * 64, 1024], "DenseLayer02")
    b_dense2 = generate_bias([1064], "BiasDenseLayer02")

with tf.name_scope("Softmax") as scope:
    conv = tf.nn.softmax(tf.matmul(h_dense1drop, w_dense2) + b_dense2)

with tf.name_scope("Entropy") as scope:
    crossentropy = -tf.reduce_sum(y*tf.compat.v1.log(conv))

with tf.name_scope("Training") as scope:
    trainingstep = tf.compat.v1.train.GradientDescentOptimizer(1e-4).minimize(crossentropy)

with tf.name_scope("Evaluating") as scope:
    correctpredict = tf.equal(tf.argmax(conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctpredict, "float"))

save = tf.compat.v1.train.Saver()

