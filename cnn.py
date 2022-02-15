"""
http://robromijnders.github.io/tensorflow_basic/

LeNet-5 Yann Lecun

"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras import layers, models
import matplotlib.pyplot as plt

import training_load as dataset

def weight_variable(shape,name):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

x = tf.compat.v1.placeholder("float", shape=[None, 1024])
y_ = tf.compat.v1.placeholder("float", shape=[None, 36])

with tf.name_scope("Reshaping_data") as scope:
    x_image = tf.reshape(x, [-1,32,32,1])

with tf.name_scope("Conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 64],"Conv_Layer_1")
    b_conv1 = bias_variable([64],"Bias_Conv_Layer_1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("Conv2") as scope:
    W_conv2 = weight_variable([3, 3, 64, 64],"Conv_Layer_2")
    b_conv2 = bias_variable([64],"Bias_Conv_Layer_2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("Conv3") as scope:
    W_conv3 = weight_variable([3, 3, 64, 64],"Conv_Layer_3")
    b_conv3 = bias_variable([64],"Bias_Conv_Layer_3")
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

with tf.name_scope("Fully_Connected1") as scope:
    W_fc1 = weight_variable([4 * 4 * 64, 1024],"Fully_Connected_layer_1")
    b_fc1 = bias_variable([1024],"Bias_Fully_Connected1")
    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

with tf.name_scope("Fully_Connected2") as scope:
    keep_prob = tf.compat.v1.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 36],"Fully_Connected_layer_2")
    b_fc2 = bias_variable([36],"Bias_Fully_Connected2")

with tf.name_scope("Final_Softmax") as scope:
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope("Entropy") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.math.log(y_conv))

with tf.name_scope("train") as scope:
    train_step = tf.compat.v1.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
saver = tf.compat.v1.train.Saver()

sess.run(tf.compat.v1.global_variables_initializer())
indice = 0
batch_size = 300
total_epoch = 450
resultados_accuracy_training = []
resultados_accuracy_validation = []
resultados_loss_train = []

ckpt = tf.train.get_checkpoint_state("CNN_log/")

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

else:
    print("no se ha encontrado checkpoint")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load()

    X_train = X_train - np.mean(X_train, axis=0)
    X_val = X_val - np.mean(X_val, axis=0)
    X_test = X_test - np.mean(X_test, axis=0)

 # Bucle de iteraciones de entrenamiento

    for i in range(total_epoch):
        batch_x = X_train[indice:indice + batch_size]
        batch_y = y_train[indice:indice + batch_size]
        indice = indice + batch_size + 1

        if indice > X_train.shape[0]:
            indice = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=0)

        if i%10 == 0:
            results_train = sess.run([accuracy,cross_entropy],feed_dict={x:batch_x,
            y_: batch_y, keep_prob: 1.0})
            train_validation = sess.run(accuracy,feed_dict={x:X_val, y_: y_val,
            keep_prob: 1.0})

            train_accuracy = results_train[0]
            train_loss = results_train[1]

            resultados_accuracy_training.append(train_accuracy)
            resultados_accuracy_validation.append(train_validation)
            resultados_loss_train.append(train_loss)

            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, validation accuracy %g"%(i, train_validation))
            print("step %d, loss %g"%(i, train_loss))

        saver.save(sess, 'CNN_log/model.ckpt', global_step=i+1)

        sess.run(train_step,feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

    print ("FINALIZADO training")
        