"""
https://medium.com/analytics-vidhya/lenet-with-tensorflow-a35da0d503df

"""
import tensorflow as tf
import dataload as dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from keras import layers, models, losses

x_train, x_test, y_train, y_test = dataset.load()

#Normalizacion de los datos
x_train = tf.pad(x_train, [[0, 0], [0,0], [0,0], [0,0]])/255
x_test = tf.pad(x_test, [[0, 0], [0,0], [0,0], [0,0]])/255

#Escogemos 25 imagenes de entrenamiento aleatorias
plt.figure(figsize=(12, 12))
start_index = 0
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    label = np.argmax(y_train[start_index+i])
    
    plt.xlabel('i={}, label={}'.format(start_index+i, label))
    plt.imshow(x_train[start_index+i], cmap='gray')
plt.show()

#Llevamos las ultimas 2000 imágenes a los datos de validación
x_val = x_train[-2000:,:,:,:] 
y_val = y_train[-2000:] 
x_train = x_train[:-2000,:,:,:] 
y_train = y_train[:-2000]

#Definición de la red
model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='relu', input_shape=x_train.shape[1:]))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(16, 5, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(120, 5, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(36, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])

#Ejecución del entrenamiento
history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))

preds = model.predict(x_test)

plt.figure(figsize=(12, 12))
start_index = random.randint(0, 2745)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = np.argmax(y_test[start_index+i])
    
    col = 'g'
    if pred != gt:
        col = 'r'
    
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color=col)
    plt.imshow(x_test[start_index+i], cmap='gray')
plt.show()

"""
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[:-2000,:,:,:]
y_train = y_train[:-2000]

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_train: ",x_val.shape)
print("y_train: ",y_val.shape)

history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))"""
