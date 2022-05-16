import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Esta funcion obtiene las imagenes de la carpeta de datos y las introduce en la matriz de datos x, matriz de clases y 
def generate_dataset():

    dirname = os.path.join(os.getcwd(), 'English/Img')
    imgpath = dirname + os.sep

    cant = 0
    image = []
    X = []
    y = []

    print("Reading images from ", imgpath)

    for root, dirnames, filenames in os.walk(imgpath):
        dirnames.sort()
        
        for filename in filenames:
            clas =  filename[3:6]
            y.append(float(clas))
            
            filepath = os.path.join(root, filename)
            image = io.imread(filepath)

            #Reducir imagen a 32x32
            reduce_image = resize(image,(32,32,1))
            X.append(reduce_image)

            print(filename)
            cant=cant+1

    X = np.array(X)

    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    y = np.array(y, dtype=float)

    return X, y

def load():
    X, y = generate_dataset()
    print("Data loaded, ", X.shape)  

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    

    return x_train, X_test, y_train, y_test
