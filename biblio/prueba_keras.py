from sklearn import datasets
digits = datasets.load_digits()

import matplotlib.pyplot as plt

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


data_n_instances, data_n_rows, data_n_cols = digits.images.shape
data_n_channels = 1

#Preparando el conjunto de datos para la convolucion
data = digits.images.reshape((data_n_instances, -1))
print('data.shape',data.shape)

print('digits.target.shape', digits.target.shape) #(1797,)
target = digits.target.reshape(-1,1) #Convirtiendo a vector columna (1797,1)
print('target.shape', target.shape) #(1797, 1)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
one_hot_target = enc.fit_transform(target)
print('one_hot_target.shape', one_hot_target.shape) #(1797, 10)
print("one_hot_target[0]", one_hot_target[0].shape)

import numpy as np

#Normalizando para que no se dispare tanto los valores de los kernels cuando se actualicen
data= (data - np.min(data))/(np.max(data) - np.min(data))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, one_hot_target, test_size=0.33, random_state=42)

print(f"X_train shape {X_train.shape} X_test shape {X_test.shape}")
print(f"y_train shape {y_train.shape} y_test shape {y_test.shape}")
print(f"y_train[0] shape {y_train[0].shape}")


import keras

#------Modelo 1
model = keras.Sequential()
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='SGD', loss='categorical_crossentropy')
# This builds the model for the first time:
model.fit(X_train, y_train, batch_size=32, epochs=10)
print(f"Evaluation Modelo 1: {model.evaluate(X_test)}")

#--------------Modelo 2
model2 = keras.Sequential()
model.add(keras.layers.Dense(30, activation='relu'))
model2.add(keras.layers.Dense(10, activation='softmax'))
model2.compile(optimizer='SGD', loss='categorical_crossentropy')
# This builds the model for the first time:
model2.fit(X_train, y_train, batch_size=32, epochs=10)
print(f"Evaluation Modelo 2: {model2.evaluate(X_test)}")


#--------------Modelo 3
data = digits.images

data = data.reshape(data.shape + (1,))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, one_hot_target, test_size=0.33, random_state=42)

print(f"X_train shape {X_train.shape} X_test shape {X_test.shape}")
print(f"y_train shape {y_train.shape} y_test shape {y_test.shape}")
print(f"y_train[0] shape {y_train[0].shape}")

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

model3 = keras.Sequential()
model3.add(keras.layers.Conv2D(2, (3,3), activation='relu'))
model3.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model3.add(keras.layers.Flatten())
model3.add(keras.layers.Dense(30, activation='relu'))
model3.add(keras.layers.Dense(10, activation='softmax'))
model3.compile(optimizer='SGD', loss='categorical_crossentropy')
# This builds the model for the first time:
model3.fit(X_train, y_train, batch_size=32, epochs=10)
print(f"Evaluation Modelo 3: {model3.evaluate(X_test)}")