import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf

#Esta funcion obtiene las imagenes de la carpeta de datos y las introduce en la matriz X
def generate_dataset():

    dirname = os.path.join(os.getcwd(), 'English/Img')
    imgpath = dirname + os.sep

    cant = 0
    image = []
    X = []
    y = []

    print("leyendo imagenes de ", imgpath)

    for root, dirnames, filenames in os.walk(imgpath):
        dirnames.sort()
        
        for filename in filenames:
            clas =  filename[3:filename.index('-')]
            y.append(float(clas))
            
            filepath = os.path.join(root, filename)
            image = io.imread(filepath)

            #Reducir imagen a 32x32
            reduce_image = resize(image,(32,32,1))

            #Invertir imagen
            reduce_image = 1 - reduce_image
            X.append(reduce_image.reshape(1024,1))

            print(filename)
            cant=cant+1

    print(cant)

    X = np.array(X)
    X = X.reshape(X.shape[:2])
    print(X.shape)

    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    y = np.array(y, dtype=float)

    np.savetxt('datax.txt', X)
    np.savetxt('datay.txt', y)

def load():
    if os.path.isfile('datax.txt') and os.path.isfile('datay.txt'):
        X = np.loadtxt('datax.txt')
        y = np.loadtxt('datay.txt')
        print ("Data loaded, ", X.shape)

        X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test
    else: generate_dataset()