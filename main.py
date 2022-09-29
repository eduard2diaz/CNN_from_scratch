from sklearn.datasets import load_digits
from classes.functions import *
from classes.Network import Network
from classes.layers.Convolution import Convolution
from classes.layers.pooling.MaxPooling import MaxPooling
from classes.layers.Flatten import Flatten
from classes.layers.Dense import Dense

digits = load_digits()
selected_instances = digits.target < 2
y = digits.target[selected_instances].reshape(-1,1)
X = digits.images[selected_instances]
X = X.reshape(X.shape + (1,))
del selected_instances

model = Network()
#model.appendLayer(Convolution(40, (3,3, 1), activation_function=relu))
#model.appendLayer(Convolution(32, (3,3, 40), activation_function=relu))
#model.appendLayer(MaxPooling(pooling_shape=(2,2), strides = 1 ))
model.appendLayer(Flatten())

topology = [8, 1]

for n_neurons in topology:
    model.appendLayer(Dense(n_neurons, sigm))
print(model)

model.train(X, y, l2_cost, lr = .05, epochs = 2500, error_range = 0.001)