from neural_network.network import Network
from neural_network.layers import Dense
from neural_network.activations import Tanh
from neural_network.losses import MSE

import numpy as np

# training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)

y_train = np.array([0, 1, 1, 0]).reshape(4, 1, 1)

# create neural network
network = Network(loss=MSE())
network.add_layers(Dense(2, 3, activation=Tanh()), Dense(3, 1, activation=Tanh()))

# train
network.fit(x_train, y_train, epochs=5000, learning_rate=0.01, verbose=True)

# predict
print(network.predict(x_train))
