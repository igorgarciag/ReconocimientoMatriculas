"""
http://robromijnders.github.io/tensorflow_basic/

https://programmerclick.com/article/517651159/

"""
from dataclasses import replace
from sqlalchemy import false
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

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
x = tf.compat.v1.placeholder("float", shape = [None, 1024], name='Input_data')
y = tf.compat.v1.placeholder("float", shape = [None, 36], name= 'Ground_truth')

with tf.name_scope("Reshaping") as scope:
    imagex = tf.reshape(x, [-1, 32, 32, 1])

with tf.name_scope("Convolution1") as socpe:
    w_conv1 = generate_weight([5, 5, 1, 64], "ConvLayer01")
    b_conv1 = generate_bias([64], "BiasConvLayer01")
    h_conv1 = tf.nn.relu(convolution2d(imagex, w_conv1) + b_conv1)
    pooling1 = maxpooling2x2(h_conv1)

with tf.name_scope("Convolution2") as socpe:
    w_conv2 = generate_weight([3, 3, 64, 64], "ConvLayer02")
    b_conv2 = generate_bias([64], "BiasConvLayer02")
    h_conv2 = tf.nn.relu(convolution2d(pooling1, w_conv2) + b_conv2)
    pooling2 = maxpooling2x2(h_conv2)

with tf.name_scope("Convolution3") as scope:
    w_conv3 = generate_weight([3, 3, 64, 64], "ConvLayer03")
    b_conv3 = generate_bias([64], "BiasConvLayer03")
    h_conv3 = tf.nn.relu(convolution2d(pooling2, w_conv3) + b_conv3)
    pooling3 = maxpooling2x2(h_conv3)

with tf.name_scope("FullyConnected1") as scope:
    w_dense1 = generate_weight([4 * 4 * 64, 1024], "DenseLayer01")
    b_dense1 = generate_bias([1024], "BiasDenseLayer01")
    pooling3flatten = tf.reshape(pooling3, [-1, 4 * 4 * 64])
    h_Dense1 = tf.nn.relu(tf.matmul(pooling3flatten, w_dense1) + b_dense1)

with tf.name_scope("FullyConnected2") as scope:
    prob = tf.compat.v1.placeholder("float")

    #prevenir overfitting
    h_dense1drop = tf.nn.dropout(h_Dense1, prob)

    w_dense2 = generate_weight([1024, 36], "DenseLayer02")
    b_dense2 = generate_bias([36], "BiasDenseLayer02")

with tf.name_scope("Softmax") as scope:
    conv = tf.nn.softmax(tf.matmul(h_dense1drop, w_dense2) + b_dense2)

with tf.name_scope("Entropy") as scope:
   crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=conv))

with tf.name_scope("Training") as scope:
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)
    trainingstep = optimizer.minimize(crossentropy)

with tf.name_scope("Evaluating") as scope:
    correctpredict = tf.equal(tf.argmax(conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctpredict, "float"))

save = tf.compat.v1.train.Saver()

session.run(tf.compat.v1.global_variables_initializer())

x_train, x_test, y_train, y_test = dataset.load()

N = x_train.shape[0]
epochs = 2000
batch = 200

for i in range(0, epochs):
    batch_ind = np.random.choice(N, batch, replace = false)
            
    if i%100 == 0:
        result = session.run([accuracy, crossentropy], feed_dict={x: x_train[batch_ind], y: y_train[batch_ind], prob: 1.0})
        acc = result[0]
        print("Step %s: Current training accuracy %s"%(i, acc))
            
    session.run(trainingstep, feed_dict={x: x_train[batch_ind], y: y_train[batch_ind], prob: 1.0})

session.close()
